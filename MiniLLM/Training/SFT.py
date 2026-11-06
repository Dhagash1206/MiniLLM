"""Supervised fine-tuning on the `sft.datasets` mix (default: smol-smoltalk + Magicoder-OSS-Instruct).

Loss is computed only on assistant tokens. Set `sft.lora_enabled=true` for LoRA.

Usage:
    torchrun --standalone --nproc_per_node=2 -m MiniLLM.Training.SFT --config Configs/Run125M.yaml
    # init from the Hub: sft.init_checkpoint=hub:$HF_USERNAME/minillm-125m-ckpt:pretrain/latest
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from MiniLLM.Config import load_config
from MiniLLM.Data.ChatData import build_sft_dataset, length_grouped_batches, pad_batch
from MiniLLM.Data.ShardLoader import ensure_tokenizer_available
from MiniLLM.Data.Tokenizer import PAD_TOKEN, load_tokenizer
from MiniLLM.LoRA import apply_lora, lora_state_dict, merged_state_dict
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

PREVIEW_QUESTIONS = [
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain what a hash table is in two sentences.",
]


def prepare_model_for_finetuning(section, dist_info, init_checkpoint: str):
    """Load init weights on every rank (rank 0 downloads first) and optionally inject LoRA."""
    if dist_info.is_main:
        resolve_checkpoint(init_checkpoint)
    if dist.is_initialized():
        dist.barrier()
    model, saved_config = load_model_weights(resolve_checkpoint(init_checkpoint), "cpu")
    model.config.gradient_checkpointing = True
    if section.lora_enabled:
        trainable = apply_lora(model, section.lora_rank, section.lora_alpha, section.lora_dropout)
        if dist_info.is_main:
            print(f"LoRA enabled: {trainable / 1e6:.2f}M trainable params", flush=True)
    return model.to(dist_info.device), saved_config


def checkpoint_payload(model, section) -> tuple[dict, dict]:
    """model.pt always holds plain merged weights; LoRA runs also keep the adapter for resuming."""
    if section.lora_enabled:
        return merged_state_dict(model), {"lora_adapter.pt": lora_state_dict(model)}
    return model.state_dict(), {}


def restore_training_state(resume_dir, model, optimizer, scaler, section, device) -> dict:
    if section.lora_enabled:
        model.load_state_dict(torch.load(resume_dir / "lora_adapter.pt", map_location=device), strict=False)
    else:
        model.load_state_dict(torch.load(resume_dir / "model.pt", map_location=device, weights_only=True))
    training_state = torch.load(resume_dir / "training_state.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(training_state["optimizer"])
    if training_state["scaler"] and device.type == "cuda":
        scaler.load_state_dict(training_state["scaler"])
    return json.loads((resume_dir / "state.json").read_text())


@torch.no_grad()
def chat_preview(model, tokenizer, device, autocast_ctx, questions=PREVIEW_QUESTIONS) -> str:
    model.eval()
    outputs = []
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for question in questions:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
        with autocast_ctx():
            generated = model.generate(input_ids, max_new_tokens=128, temperature=0.7, top_p=0.9,
                                       eos_token_ids=[im_end_id, tokenizer.eos_token_id])
        answer = tokenizer.decode(generated[0, input_ids.shape[1] :].tolist(), skip_special_tokens=True)
        outputs.append(f"Q: {question}\nA: {answer}")
    model.train()
    return "\n---\n".join(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    section = config.sft
    dist_info = setup_distributed()
    seed_everything(config.data.seed + dist_info.rank)
    use_cuda = dist_info.device.type == "cuda"

    def autocast_ctx():
        return torch.autocast("cuda", dtype=torch.float16) if use_cuda else contextlib.nullcontext()

    if dist_info.is_main:
        ensure_tokenizer_available(config.data)
        build_sft_dataset(config.data.tokenizer_dir, section.max_seq_len, section.datasets, config.data.seed)
    if dist.is_initialized():
        dist.barrier()
    examples = build_sft_dataset(config.data.tokenizer_dir, section.max_seq_len, section.datasets, config.data.seed)
    tokenizer = load_tokenizer(config.data.tokenizer_dir)
    pad_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)

    val_count = max(1, int(len(examples) * section.val_fraction))
    val_examples, train_examples = examples[:val_count], examples[val_count:]
    train_lengths = [len(input_ids) for input_ids, _ in train_examples]

    batches_per_epoch = len(length_grouped_batches(
        train_lengths, section.micro_batch_size, 0, dist_info.world_size)) // dist_info.world_size
    steps_per_epoch = batches_per_epoch // section.grad_accum_steps
    total_steps = steps_per_epoch * section.epochs

    model, _ = prepare_model_for_finetuning(section, dist_info, section.init_checkpoint)
    optimizer = configure_optimizer(model, section.weight_decay, section.max_lr, (0.9, 0.95), dist_info.device.type)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    checkpoints = CheckpointManager(section.output_dir, config.hub.checkpoint_repo, config.hub.push_checkpoints,
                                    config.hub.private, config.hub.squash_history, path_in_repo="sft",
                                    keep_last=config.train.keep_checkpoints)
    start_step, wandb_run_id = 0, None
    if dist_info.is_main:
        checkpoints.find_resume_dir()
    if dist.is_initialized():
        dist.barrier()
    resume_dir = checkpoints.find_resume_dir()
    if resume_dir is not None:
        state = restore_training_state(resume_dir, model, optimizer, scaler, section, dist_info.device)
        start_step, wandb_run_id = state["step"], state.get("wandb_run_id")
        if dist_info.is_main:
            print(f"Resumed SFT at step {start_step}", flush=True)

    raw_model = model
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[dist_info.local_rank] if use_cuda else None)

    logger = None
    if dist_info.is_main:
        logger = WandbLogger(config.wandb.enabled, config.wandb.project,
                             (config.wandb.run_name or "run") + "-sft", config.to_dict(), wandb_run_id)
        print(f"SFT: {len(train_examples):,} train / {len(val_examples):,} val examples, "
              f"{total_steps:,} steps", flush=True)

    @torch.no_grad()
    def evaluate() -> float:
        raw_model.eval()
        loss_sum = torch.zeros((), device=dist_info.device)
        token_count = torch.zeros((), device=dist_info.device)
        shard = val_examples[dist_info.rank :: dist_info.world_size]
        for start in range(0, len(shard), section.micro_batch_size):
            inputs, labels = pad_batch(shard[start : start + section.micro_batch_size], pad_id)
            inputs, labels = inputs.to(dist_info.device), labels.to(dist_info.device)
            with autocast_ctx():
                logits, _ = raw_model(inputs)
            # Token-weighted so the metric is independent of batch composition.
            loss_sum += torch.nn.functional.cross_entropy(
                logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")
            token_count += (labels != -100).sum()
        if dist.is_initialized():
            dist.all_reduce(loss_sum)
            dist.all_reduce(token_count)
        raw_model.train()
        return (loss_sum / token_count.clamp(min=1)).item()

    saved_steps: set[int] = set()

    def save_checkpoint(next_step: int, force: bool = False) -> None:
        # Skips a duplicate save when the interval save already covered the final step.
        if dist_info.is_main and next_step not in saved_steps:
            model_state, extra_files = checkpoint_payload(raw_model, section)
            state = {"step": next_step, "total_steps": total_steps, "wandb_run_id": logger.run_id if logger else None}
            if checkpoints.save(raw_model, optimizer, scaler if use_cuda else None, state, config, force=force,
                                extra_files=extra_files, model_state=model_state):
                saved_steps.add(next_step)
                print(f"Saved SFT checkpoint at step {next_step}", flush=True)
        if dist.is_initialized():
            dist.barrier()

    timer = Timer(section.max_runtime_hours)
    min_lr = section.max_lr * section.min_lr_ratio
    model.train()
    window_start = time.time()
    tokens_in_window = 0
    batches_by_epoch: dict[int, list[list[int]]] = {}
    try:
        for step in range(start_step, total_steps):
            epoch, step_in_epoch = divmod(step, steps_per_epoch)
            if epoch not in batches_by_epoch:
                batches_by_epoch.clear()
                batches_by_epoch[epoch] = length_grouped_batches(
                    train_lengths, section.micro_batch_size, config.data.seed + epoch, dist_info.world_size)
            epoch_batches = batches_by_epoch[epoch]
            learning_rate = cosine_lr(step, section.max_lr, min_lr, section.warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            loss_accum = torch.zeros((), device=dist_info.device)
            for micro_step in range(section.grad_accum_steps):
                batch_index = (step_in_epoch * section.grad_accum_steps + micro_step) * dist_info.world_size
                batch = [train_examples[i] for i in epoch_batches[batch_index + dist_info.rank]]
                inputs, labels = pad_batch(batch, pad_id)
                inputs, labels = inputs.to(dist_info.device), labels.to(dist_info.device)
                tokens_in_window += int((labels != -100).sum())
                is_last = micro_step == section.grad_accum_steps - 1
                sync_ctx = (model.no_sync() if isinstance(model, DistributedDataParallel) and not is_last
                            else contextlib.nullcontext())
                with sync_ctx, autocast_ctx():
                    _, loss = model(inputs, labels)
                    loss = loss / section.grad_accum_steps
                loss_accum += loss.detach()
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for param in raw_model.parameters() if param.requires_grad], section.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            completed_steps = step + 1

            if completed_steps % section.log_interval == 0:
                all_reduce_mean(loss_accum)
                if dist_info.is_main:
                    elapsed = time.time() - window_start
                    metrics = {"sft/loss": loss_accum.item(), "sft/lr": learning_rate,
                               "sft/grad_norm": grad_norm.item(), "sft/epoch": epoch + step_in_epoch / steps_per_epoch,
                               "sft/target_tokens_per_sec": tokens_in_window * dist_info.world_size / elapsed}
                    print(f"sft step {completed_steps}/{total_steps} loss {metrics['sft/loss']:.4f} "
                          f"lr {learning_rate:.2e}", flush=True)
                    logger.log(metrics, completed_steps)
                window_start, tokens_in_window = time.time(), 0

            if completed_steps % section.eval_interval == 0 or completed_steps == total_steps:
                val_loss = evaluate()
                if dist_info.is_main:
                    print(f"sft eval loss {val_loss:.4f} ppl {math.exp(min(val_loss, 20)):.2f}", flush=True)
                    logger.log({"sft/val_loss": val_loss, "sft/val_ppl": math.exp(min(val_loss, 20))},
                               completed_steps)
                    print(chat_preview(raw_model, tokenizer, dist_info.device, autocast_ctx), flush=True)

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
