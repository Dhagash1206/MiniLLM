"""Optional DPO on preference pairs (default: UltraFeedback binarized), starting from the SFT model.

Usage:
    torchrun --standalone --nproc_per_node=2 -m MiniLLM.Training.DPO --config Configs/Run125M.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from MiniLLM.Config import load_config
from MiniLLM.Data.ChatData import IGNORE_INDEX, build_dpo_dataset, pad_batch
from MiniLLM.Data.ShardLoader import ensure_tokenizer_available
from MiniLLM.Data.Tokenizer import PAD_TOKEN, load_tokenizer
from MiniLLM.Model import configure_optimizer
from MiniLLM.Training.TrainingUtils import (
    CheckpointManager,
    Timer,
    WandbLogger,
    all_reduce_mean,
    any_rank_true,
    cleanup_distributed,
    cosine_lr,
    load_model_weights,
    resolve_checkpoint,
    seed_everything,
    setup_distributed,
)
from MiniLLM.Training.SFT import chat_preview, checkpoint_payload, prepare_model_for_finetuning, restore_training_state


def sequence_logprobs(model, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log p(response tokens) per sequence."""
    logits, _ = model(inputs)
    mask = labels != IGNORE_INDEX
    safe_labels = labels.masked_fill(~mask, 0)
    token_logprobs = torch.gather(F.log_softmax(logits.float(), dim=-1), 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * mask).sum(-1)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta: float):
    chosen_reward = beta * (policy_chosen - reference_chosen)
    rejected_reward = beta * (policy_rejected - reference_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    accuracy = (chosen_reward > rejected_reward).float().mean()
    return loss, chosen_reward.mean(), rejected_reward.mean(), accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    section = config.dpo
    dist_info = setup_distributed()
    seed_everything(config.data.seed + dist_info.rank)
    use_cuda = dist_info.device.type == "cuda"

    def autocast_ctx():
        return torch.autocast("cuda", dtype=torch.float16) if use_cuda else contextlib.nullcontext()

    if dist_info.is_main:
        ensure_tokenizer_available(config.data)
        build_dpo_dataset(config.data.tokenizer_dir, section.dataset, section.split,
                          section.max_pairs + section.val_pairs, section.max_seq_len, config.data.seed)
    if dist.is_initialized():
        dist.barrier()
    pairs = build_dpo_dataset(config.data.tokenizer_dir, section.dataset, section.split,
                              section.max_pairs + section.val_pairs, section.max_seq_len, config.data.seed)
    tokenizer = load_tokenizer(config.data.tokenizer_dir)
    pad_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    val_pairs, train_pairs = pairs[: section.val_pairs], pairs[section.val_pairs :]

    pairs_per_step = section.micro_batch_size * section.grad_accum_steps * dist_info.world_size
    steps_per_epoch = len(train_pairs) // pairs_per_step
    total_steps = steps_per_epoch * section.epochs

    policy, _ = prepare_model_for_finetuning(section, dist_info, section.init_checkpoint)
    # The reference is the frozen starting policy; at 125M a full copy is cheaper than toggling adapters.
    reference, _ = load_model_weights(resolve_checkpoint(section.init_checkpoint), dist_info.device)
    reference.eval().requires_grad_(False)
    if use_cuda:
        reference.half()

    optimizer = configure_optimizer(policy, 0.0, section.max_lr, (0.9, 0.95), dist_info.device.type)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    checkpoints = CheckpointManager(section.output_dir, config.hub.checkpoint_repo, config.hub.push_checkpoints,
                                    config.hub.private, config.hub.squash_history, path_in_repo="dpo")
    start_step, wandb_run_id = 0, None
    if dist_info.is_main:
        checkpoints.find_resume_dir()
    if dist.is_initialized():
        dist.barrier()
    resume_dir = checkpoints.find_resume_dir()
    if resume_dir is not None:
        state = restore_training_state(resume_dir, policy, optimizer, scaler, section, dist_info.device)
        start_step, wandb_run_id = state["step"], state.get("wandb_run_id")

    raw_policy = policy
    if dist.is_initialized():
        policy = DistributedDataParallel(policy, device_ids=[dist_info.local_rank] if use_cuda else None)

    logger = None
    if dist_info.is_main:
        logger = WandbLogger(config.wandb.enabled, config.wandb.project,
                             (config.wandb.run_name or "run") + "-dpo", config.to_dict(), wandb_run_id)
        print(f"DPO: {len(train_pairs):,} train / {len(val_pairs):,} val pairs, {total_steps:,} steps", flush=True)

    def pair_batch(batch_pairs):
        # Chosen and rejected share one forward pass: first half chosen, second half rejected.
        examples = [pair["chosen"] for pair in batch_pairs] + [pair["rejected"] for pair in batch_pairs]
        inputs, labels = pad_batch(examples, pad_id)
        return inputs.to(dist_info.device), labels.to(dist_info.device)

    def compute_loss(model, batch_pairs):
        inputs, labels = pair_batch(batch_pairs)
        half = len(batch_pairs)
        with autocast_ctx():
            policy_logprobs = sequence_logprobs(model, inputs, labels)
            with torch.no_grad():
                reference_logprobs = sequence_logprobs(reference, inputs, labels)
        return dpo_loss(policy_logprobs[:half], policy_logprobs[half:],
                        reference_logprobs[:half], reference_logprobs[half:], section.beta)

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        raw_policy.eval()
        totals = torch.zeros(2, device=dist_info.device)
        batch_count = torch.zeros((), device=dist_info.device)
        shard = val_pairs[dist_info.rank :: dist_info.world_size]
        for start in range(0, len(shard), section.micro_batch_size):
            loss, _, _, accuracy = compute_loss(raw_policy, shard[start : start + section.micro_batch_size])
            totals += torch.stack([loss.detach().float(), accuracy.float()])
            batch_count += 1
        if dist.is_initialized():
            dist.all_reduce(totals)
            dist.all_reduce(batch_count)
        raw_policy.train()
        totals = totals / batch_count.clamp(min=1)
        return {"dpo/val_loss": totals[0].item(), "dpo/val_reward_accuracy": totals[1].item()}

    def save_checkpoint(next_step: int, force: bool = False) -> None:
        if dist_info.is_main:
            model_state, extra_files = checkpoint_payload(raw_policy, section)
            state = {"step": next_step, "total_steps": total_steps, "wandb_run_id": logger.run_id if logger else None}
            checkpoints.save(raw_policy, optimizer, scaler if use_cuda else None, state, config, force=force,
                             extra_files=extra_files, model_state=model_state)
        if dist.is_initialized():
            dist.barrier()

    timer = Timer(section.max_runtime_hours)
    policy.train()
    window_start = time.time()
    try:
        for step in range(start_step, total_steps):
            epoch, step_in_epoch = divmod(step, steps_per_epoch)
            order = torch.randperm(len(train_pairs), generator=torch.Generator().manual_seed(config.data.seed + epoch))
            learning_rate = cosine_lr(step, section.max_lr, 0.0, section.warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            stats = torch.zeros(4, device=dist_info.device)
            for micro_step in range(section.grad_accum_steps):
                first = ((step_in_epoch * section.grad_accum_steps + micro_step) * dist_info.world_size
                         + dist_info.rank) * section.micro_batch_size
                batch_pairs = [train_pairs[int(i)] for i in order[first : first + section.micro_batch_size]]
                is_last = micro_step == section.grad_accum_steps - 1
                sync_ctx = (policy.no_sync() if isinstance(policy, DistributedDataParallel) and not is_last
                            else contextlib.nullcontext())
                with sync_ctx:
                    loss, chosen_reward, rejected_reward, accuracy = compute_loss(policy, batch_pairs)
                    scaler.scale(loss / section.grad_accum_steps).backward()
                stats += torch.stack([loss.detach(), chosen_reward.detach(), rejected_reward.detach(),
                                      accuracy.detach()]).float() / section.grad_accum_steps

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for param in raw_policy.parameters() if param.requires_grad], section.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            completed_steps = step + 1

            if completed_steps % section.log_interval == 0:
                all_reduce_mean(stats)
                if dist_info.is_main:
                    metrics = {"dpo/loss": stats[0].item(), "dpo/chosen_reward": stats[1].item(),
                               "dpo/rejected_reward": stats[2].item(), "dpo/reward_accuracy": stats[3].item(),
                               "dpo/lr": learning_rate, "dpo/grad_norm": grad_norm.item(),
                               "dpo/sec_per_step": (time.time() - window_start) / section.log_interval}
                    print(f"dpo step {completed_steps}/{total_steps} loss {metrics['dpo/loss']:.4f} "
                          f"acc {metrics['dpo/reward_accuracy']:.3f}", flush=True)
                    logger.log(metrics, completed_steps)
                window_start = time.time()

            if completed_steps % section.eval_interval == 0 or completed_steps == total_steps:
                metrics = evaluate()
                if dist_info.is_main:
                    print("dpo eval " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
                    logger.log(metrics, completed_steps)
                    print(chat_preview(raw_policy, tokenizer, dist_info.device, autocast_ctx), flush=True)

            if completed_steps % section.checkpoint_interval == 0:
                save_checkpoint(completed_steps)
            if any_rank_true(timer.out_of_time(), dist_info.device):
                save_checkpoint(completed_steps, force=True)
                break
        else:
            if start_step < total_steps:
                save_checkpoint(total_steps, force=True)
    finally:
        if dist_info.is_main:
            checkpoints.wait_for_upload()
            if logger:
                logger.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
