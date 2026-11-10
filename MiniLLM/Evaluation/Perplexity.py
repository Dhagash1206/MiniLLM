"""Perplexity on the held-out English and code validation shards.

Usage:
    python -m MiniLLM.Evaluation.Perplexity --model checkpoints/pretrain/latest --tokens-dir data/tokens
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from MiniLLM.Data.ShardLoader import ValidationSet
from MiniLLM.Evaluation.Common import autocast_context, load_eval_model, pick_device


@torch.no_grad()
def run(model_path: str, tokens_dir: str, max_batches: int, batch_size: int) -> dict[str, float]:
    device = pick_device()
    model = load_eval_model(model_path, device)
    validation = ValidationSet(Path(tokens_dir), model.config.max_seq_len)
    results = {}
    for domain in sorted(validation.domains):
        loss_sum, batch_count = 0.0, 0
        for inputs, targets in validation.batches(domain, max_batches, batch_size, 0, 1, device):
            with autocast_context(device):
                _, loss = model(inputs, targets)
            loss_sum += loss.item()
            batch_count += 1
        mean_loss = loss_sum / max(batch_count, 1)
        results[f"{domain}_loss"] = mean_loss
        results[f"{domain}_ppl"] = math.exp(mean_loss)
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens-dir", default="data/tokens")
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run(args.model, args.tokens_dir, args.max_batches, args.batch_size)


if __name__ == "__main__":
    main()
