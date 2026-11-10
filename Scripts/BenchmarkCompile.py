"""Training throughput with and without torch.compile (roadmap: benchmark train.compile=true on a T4).

Uses the config's real model, optimizer, micro-batch and autocast setup on synthetic tokens, so no data is
needed. Warm-up steps (which include compilation) are timed separately from the measured steps.

Usage (one GPU, e.g. a Kaggle T4 cell):
    python Scripts/BenchmarkCompile.py --config Configs/Run125M.yaml
    python Scripts/BenchmarkCompile.py --config Configs/Proxy30M.yaml --steps 30 --out results/compile_proxy.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MiniLLM.Config import load_config  # noqa: E402
from MiniLLM.Model import Transformer, configure_optimizer  # noqa: E402
from MiniLLM.Training.TrainingUtils import setup_distributed  # noqa: E402


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    if device.type == "mps":
        return torch.autocast("mps", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(config, compile_model: bool, warmup_steps: int, steps: int, device: torch.device) -> dict:
    torch.manual_seed(0)
    model = Transformer(config.model).to(device).train()
    optimizer = configure_optimizer(model, config.train.weight_decay, config.train.max_lr,
                                    (config.train.beta1, config.train.beta2), device.type)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_model = torch.compile(model) if compile_model else model
    batch_size, seq_len = config.train.micro_batch_size, config.model.max_seq_len
    tokens = torch.randint(0, config.model.vocab_size, (batch_size, seq_len + 1), device=device)

    def train_step() -> None:
        with autocast_context(device):
            _, loss = train_model(tokens[:, :-1], tokens[:, 1:])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for _ in range(warmup_steps):
        train_step()
    synchronize(device)
    warmup_seconds = time.time() - start
    start = time.time()
    for _ in range(steps):
        train_step()
    synchronize(device)
    measured_seconds = time.time() - start
    result = {
        "compile": compile_model,
        "tokens_per_sec": round(steps * batch_size * seq_len / measured_seconds),
        "step_ms": round(1000 * measured_seconds / steps, 1),
        "warmup_seconds": round(warmup_seconds, 1),
    }
    if device.type == "cuda":
        result["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    del train_model, model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="Configs/Run125M.yaml")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", default="results/compile_benchmark.json")
    parser.add_argument("overrides", nargs="*", help="section.key=value config overrides")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    device = setup_distributed().device
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    print(f"{args.config} on {device_name}: micro_batch={config.train.micro_batch_size} "
          f"seq_len={config.model.max_seq_len} torch={torch.__version__}", flush=True)
    results = [benchmark(config, compile_model, args.warmup_steps, args.steps, device)
               for compile_model in (False, True)]

    eager, compiled = results
    speedup = compiled["tokens_per_sec"] / eager["tokens_per_sec"]
    print("| compile | tokens/s | step ms | warm-up s | peak GB |\n|---|---|---|---|---|")
    for result in results:
        print(f"| {result['compile']} | {result['tokens_per_sec']:,} | {result['step_ms']} | "
              f"{result['warmup_seconds']} | {result.get('peak_mem_gb', '-')} |")
    print(f"speedup with compile: {speedup:.2f}x", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"config": args.config, "device": device_name, "torch": torch.__version__,
                                          "results": results, "speedup": round(speedup, 3)}, indent=2))


if __name__ == "__main__":
    main()
