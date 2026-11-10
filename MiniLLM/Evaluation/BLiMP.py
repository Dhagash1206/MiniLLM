"""BLiMP: 67 grammatical-vs-ungrammatical minimal pairs (1,000 each); accuracy = P(good) > P(bad).

Scores whole-sentence log-probability, so it needs no prompt format and works for base models.

Usage:
    python -m MiniLLM.Evaluation.BLiMP --model checkpoints/pretrain/latest --tokenizer data/tokens/tokenizer
    python -m MiniLLM.Evaluation.BLiMP --model hf:openai-community/gpt2 --limit-per-task 200
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from MiniLLM.Evaluation.Common import autocast_context, load_eval_model, load_eval_tokenizer, pick_device

BLIMP_REPO = "nyu-mll/blimp"


@torch.no_grad()
def sentence_log_probs(model, tokenizer, sentences: list[str], device) -> list[float]:
    # A leading EOS/BOS gives the first word a context to be scored against.
    start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    sequences = [[start_id] + tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]
    max_len = max(len(sequence) for sequence in sequences)
    tokens = torch.zeros((len(sequences), max_len), dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        tokens[row, : len(sequence)] = torch.tensor(sequence)
        mask[row, 1 : len(sequence)] = True
    tokens, mask = tokens.to(device), mask.to(device)
    with autocast_context(device):
        logits, _ = model(tokens)
    token_log_probs = -F.cross_entropy(logits[:, :-1].float().transpose(1, 2), tokens[:, 1:], reduction="none")
    return (token_log_probs * mask[:, 1:]).sum(1).tolist()


def run(model_path: str, tokenizer_dir: str | None, limit_per_task: int | None, batch_pairs: int = 16) -> dict:
    from datasets import get_dataset_config_names, load_dataset

    device = pick_device()
    model = load_eval_model(model_path, device)
    tokenizer = load_eval_tokenizer(model_path, tokenizer_dir)

    per_task, total_correct, total_pairs = {}, 0, 0
    for task in get_dataset_config_names(BLIMP_REPO):
        pairs = list(load_dataset(BLIMP_REPO, task, split="train"))
        if limit_per_task:
            pairs = pairs[:limit_per_task]
        correct = 0
        for start in range(0, len(pairs), batch_pairs):
            batch = pairs[start : start + batch_pairs]
            scores = sentence_log_probs(
                model, tokenizer, [pair["sentence_good"] for pair in batch] + [pair["sentence_bad"] for pair in batch],
                device,
            )
            good, bad = scores[: len(batch)], scores[len(batch) :]
            correct += sum(good_score > bad_score for good_score, bad_score in zip(good, bad))
        per_task[task] = correct / len(pairs)
        total_correct += correct
        total_pairs += len(pairs)
        print(f"{task}: {per_task[task]:.3f}", flush=True)

    results = {"blimp_acc": total_correct / total_pairs, "blimp_n": total_pairs, "blimp_per_task": per_task}
    print(json.dumps({key: value for key, value in results.items() if key != "blimp_per_task"}, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None, help="Required for MiniLLM checkpoints")
    parser.add_argument("--limit-per-task", type=int, default=None)
    args = parser.parse_args()
    run(args.model, args.tokenizer, args.limit_per_task)


if __name__ == "__main__":
    main()
