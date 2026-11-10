"""HellaSwag (validation, 10,042 items): pick the ending with the lowest mean token loss (acc_norm).

Usage:
    python -m MiniLLM.Evaluation.HellaSwag --model checkpoints/pretrain/latest --tokenizer data/tokens/tokenizer
    python -m MiniLLM.Evaluation.HellaSwag --model hf:openai-community/gpt2        # baseline
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from MiniLLM.Evaluation.Common import autocast_context, load_eval_model, load_eval_tokenizer, pick_device


def load_examples() -> list[dict]:
    from datasets import load_dataset

    return list(load_dataset("Rowan/hellaswag", split="validation"))


@torch.no_grad()
def score_example(model, tokenizer, example: dict, device) -> tuple[int, int]:
    """Returns (predicted index by sum loss, predicted index by length-normalized loss)."""
    context_ids = tokenizer.encode(example["ctx"], add_special_tokens=False)
    candidates = [tokenizer.encode(" " + ending, add_special_tokens=False) for ending in example["endings"]]
    max_len = max(len(context_ids) + len(ending) for ending in candidates)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.bool)
    for row, ending_ids in enumerate(candidates):
        sequence = context_ids + ending_ids
        tokens[row, : len(sequence)] = torch.tensor(sequence)
        mask[row, len(context_ids) : len(sequence)] = True
    tokens, mask = tokens.to(device), mask.to(device)

    with autocast_context(device):
        logits, _ = model(tokens)
    shift_logits = logits[:, :-1].float()
    shift_targets = tokens[:, 1:]
    shift_mask = mask[:, 1:].float()
    token_losses = F.cross_entropy(shift_logits.transpose(1, 2), shift_targets, reduction="none")
    sum_loss = (token_losses * shift_mask).sum(1)
    mean_loss = sum_loss / shift_mask.sum(1)
    return int(sum_loss.argmin()), int(mean_loss.argmin())


def run(model_path: str, tokenizer_dir: str | None, limit: int | None = None) -> dict[str, float]:
    device = pick_device()
    model = load_eval_model(model_path, device)
    tokenizer = load_eval_tokenizer(model_path, tokenizer_dir)
    examples = load_examples()[:limit] if limit else load_examples()

    correct, correct_norm = 0, 0
    for index, example in enumerate(examples, start=1):
        prediction, prediction_norm = score_example(model, tokenizer, example, device)
        label = int(example["label"])
        correct += prediction == label
        correct_norm += prediction_norm == label
        if index % 1000 == 0:
            print(f"{index}/{len(examples)} acc_norm={correct_norm / index:.4f}", flush=True)
    results = {"hellaswag_acc": correct / len(examples), "hellaswag_acc_norm": correct_norm / len(examples),
               "hellaswag_n": len(examples)}
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None, help="Required for MiniLLM checkpoints")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.model, args.tokenizer, args.limit)


if __name__ == "__main__":
    main()
