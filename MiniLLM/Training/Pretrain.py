"""Pretraining entry point (single GPU, or DDP via torchrun).

Usage:
    torchrun --standalone --nproc_per_node=2 -m MiniLLM.Training.Pretrain --config Configs/Run125M.yaml
    python -m MiniLLM.Training.Pretrain --config Configs/SmokeTest.yaml          # CPU/MPS smoke test
Overrides: append `section.key=value`, e.g. `train.micro_batch_size=8`.
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
from MiniLLM.Data.ShardLoader import PretrainSampler, ValidationSet, ensure_tokens_available
from MiniLLM.Model import Transformer, configure_optimizer
from MiniLLM.Training.TrainingUtils import (
    CheckpointManager,
    Timer,
    WandbLogger,
    all_reduce_mean,
    any_rank_true,
    cleanup_distributed,
    cosine_lr,
    seed_everything,
    setup_distributed,
    unwrap,
)

PREVIEW_PROMPTS = ["The theory of evolution explains", "def fibonacci(n):\n", "SELECT name, COUNT(*)"]


@torch.no_grad()
def evaluate(model: Transformer, validation: ValidationSet, config, dist_info, autocast_ctx) -> dict[str, float]:
    model.eval()
    metrics = {}
    for domain in sorted(validation.domains):
        loss_sum = torch.zeros((), device=dist_info.device)
        batch_count = torch.zeros((), device=dist_info.device)
        for inputs, targets in validation.batches(
            domain, config.train.eval_batches, config.train.micro_batch_size, dist_info.rank,
            dist_info.world_size, dist_info.device,
        ):
            with autocast_ctx():
                _, loss = model(inputs, targets)
            loss_sum += loss.float()
            batch_count += 1
        if dist.is_initialized():
            dist.all_reduce(loss_sum)
            dist.all_reduce(batch_count)
        mean_loss = (loss_sum / batch_count.clamp(min=1)).item()
        metrics[f"val/{domain}_loss"] = mean_loss
        metrics[f"val/{domain}_ppl"] = math.exp(min(mean_loss, 20))
    model.train()
    return metrics


@torch.no_grad()
def preview_samples(model: Transformer, tokenizer, device, autocast_ctx) -> str:
    model.eval()
    outputs = []
    for prompt in PREVIEW_PROMPTS:
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        with autocast_ctx():
            generated = model.generate(
                input_ids, max_new_tokens=64, temperature=0.8, top_p=0.95,
                eos_token_ids=[tokenizer.eos_token_id],
            )
        outputs.append(tokenizer.decode(generated[0].tolist()))
    model.train()
    return "\n---\n".join(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    dist_info = setup_distributed()
    seed_everything(config.data.seed + dist_info.rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Rank 0 downloads shards once; others wait so nobody reads half-written files.
    if dist_info.is_main:
        ensure_tokens_available(config.data.tokens_dir, config.data.hub_repo)
    if dist.is_initialized():
        dist.barrier()
    tokens_dir = ensure_tokens_available(config.data.tokens_dir, config.data.hub_repo)

    seq_len = config.model.max_seq_len
    sampler = PretrainSampler(tokens_dir, seq_len, seed=config.data.seed)
    validation = ValidationSet(tokens_dir, seq_len)

    tokens_per_micro_step = config.train.micro_batch_size * seq_len * dist_info.world_size
    grad_accum_steps = max(1, config.train.tokens_per_step // tokens_per_micro_step)
    tokens_per_step = grad_accum_steps * tokens_per_micro_step
    total_steps = config.train.total_tokens // tokens_per_step

    use_cuda = dist_info.device.type == "cuda"
    # T4 has no bf16 tensor cores, so fp16 autocast + loss scaling.
    def autocast_ctx():
        if use_cuda:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    model = Transformer(config.model).to(dist_info.device)
    optimizer = configure_optimizer(
        model, config.train.weight_decay, config.train.max_lr,
        (config.train.beta1, config.train.beta2), dist_info.device.type,
    )

    checkpoints = CheckpointManager(
        f"{config.train.checkpoint_dir}/pretrain", config.hub.checkpoint_repo, config.hub.push_checkpoints,
        config.hub.private, config.hub.squash_history, path_in_repo="pretrain",
    )
    start_step, tokens_seen, wandb_run_id = 0, 0, None
    if dist_info.is_main:
        checkpoints.find_resume_dir()
    if dist.is_initialized():
        dist.barrier()
    resume_dir = checkpoints.find_resume_dir()
    if resume_dir is not None:
        state = json.loads((resume_dir / "state.json").read_text())
        model.load_state_dict(torch.load(resume_dir / "model.pt", map_location=dist_info.device, weights_only=True))
        training_state = torch.load(resume_dir / "training_state.pt", map_location=dist_info.device, weights_only=False)
        optimizer.load_state_dict(training_state["optimizer"])
        if training_state["scaler"] and use_cuda:
            scaler.load_state_dict(training_state["scaler"])
        start_step, tokens_seen, wandb_run_id = state["step"], state["tokens_seen"], state.get("wandb_run_id")
        if dist_info.is_main:
            print(f"Resumed from {resume_dir} at step {start_step}", flush=True)

    raw_model = model
    if config.train.compile:
        model = torch.compile(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[dist_info.local_rank] if use_cuda else None)

    tokenizer = None
    logger = None
    if dist_info.is_main:
        from MiniLLM.Data.Tokenizer import load_tokenizer

        try:
            tokenizer = load_tokenizer(config.data.tokenizer_dir)
        except Exception as error:
            print(f"Tokenizer not found ({error}); skipping sample previews", flush=True)
        logger = WandbLogger(config.wandb.enabled, config.wandb.project, config.wandb.run_name,
                             config.to_dict(), wandb_run_id)
        print(
            f"params={raw_model.num_parameters() / 1e6:.1f}M world={dist_info.world_size} "
            f"grad_accum={grad_accum_steps} tokens/step={tokens_per_step:,} steps={total_steps:,} "
            f"train_windows={sampler.num_windows:,}",
            flush=True,
        )

    def save_checkpoint(next_step: int, force: bool = False) -> None:
        if dist_info.is_main:
            state = {"step": next_step, "tokens_seen": tokens_seen, "wandb_run_id": logger.run_id if logger else None,
                     "total_steps": total_steps}
            if checkpoints.save(raw_model, optimizer, scaler if use_cuda else None, state, config, force=force):
                print(f"Saved checkpoint at step {next_step}", flush=True)
        if dist.is_initialized():
            dist.barrier()

    timer = Timer(config.train.max_runtime_hours)
    min_lr = config.train.max_lr * config.train.min_lr_ratio
    model.train()
    step = start_step
    window_start = time.time()
    try:
        for step in range(start_step, total_steps):
            learning_rate = cosine_lr(step, config.train.max_lr, min_lr, config.train.warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            loss_accum = torch.zeros((), device=dist_info.device)
            for micro_step in range(grad_accum_steps):
                inputs, targets = sampler.get_batch(
                    step * grad_accum_steps + micro_step, dist_info.rank, dist_info.world_size,
                    config.train.micro_batch_size, dist_info.device,
                )
                is_last_micro_step = micro_step == grad_accum_steps - 1
                # Skip the gradient all-reduce on all but the last micro-step.
                sync_ctx = (
                    model.no_sync() if isinstance(model, DistributedDataParallel) and not is_last_micro_step
                    else contextlib.nullcontext()
                )
                with sync_ctx, autocast_ctx():
                    _, loss = model(inputs, targets)
                    loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            tokens_seen += tokens_per_step
            completed_steps = step + 1

            if completed_steps % config.train.log_interval == 0:
                all_reduce_mean(loss_accum)
                if use_cuda:
                    torch.cuda.synchronize()
                elapsed = time.time() - window_start
                window_start = time.time()
                if dist_info.is_main:
                    tokens_per_second = tokens_per_step * config.train.log_interval / elapsed
                    remaining_hours = (total_steps - completed_steps) * tokens_per_step / tokens_per_second / 3600
                    metrics = {
                        "train/loss": loss_accum.item(), "train/lr": learning_rate,
                        "train/grad_norm": grad_norm.item(), "train/tokens_per_sec": tokens_per_second,
                        "train/tokens_seen": tokens_seen, "train/loss_scale": scaler.get_scale() if use_cuda else 1.0,
                    }
                    print(
                        f"step {completed_steps}/{total_steps} loss {metrics['train/loss']:.4f} "
                        f"lr {learning_rate:.2e} gnorm {metrics['train/grad_norm']:.2f} "
                        f"{tokens_per_second / 1e3:.1f}k tok/s eta {remaining_hours:.1f}h",
                        flush=True,
                    )
                    logger.log(metrics, completed_steps)

            if completed_steps % config.train.eval_interval == 0 or completed_steps == total_steps:
                metrics = evaluate(raw_model, validation, config, dist_info, autocast_ctx)
                if dist_info.is_main:
                    print("eval " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()), flush=True)
                    logger.log(metrics, completed_steps)

            if dist_info.is_main and tokenizer is not None and completed_steps % config.train.sample_interval == 0:
                samples = preview_samples(raw_model, tokenizer, dist_info.device, autocast_ctx)
                print(f"--- samples @ {completed_steps} ---\n{samples}\n", flush=True)
                if logger.run:
                    import wandb

                    logger.log({"samples": wandb.Html(f"<pre>{samples}</pre>")}, completed_steps)

            if completed_steps % config.train.checkpoint_interval == 0:
                save_checkpoint(completed_steps)

            if any_rank_true(timer.out_of_time(), dist_info.device):
                if dist_info.is_main:
                    print(f"Time budget reached after {timer.elapsed_hours():.2f}h; saving and exiting", flush=True)
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
