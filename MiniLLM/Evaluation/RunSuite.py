"""Run every benchmark for one model and save results/<name>.json (input for Evaluation.Report).

Code benchmarks (MBPP, HumanEval) execute model-written code, so they only run with --allow-code-exec;
use that on a disposable machine (Kaggle/Colab).

Usage:
    python -m MiniLLM.Evaluation.RunSuite --name minillm-pretrain --model checkpoints/pretrain/latest \
        --tokenizer data/tokens/tokenizer --tokens-dir data/tokens
    python -m MiniLLM.Evaluation.RunSuite --name gpt2 --model hf:openai-community/gpt2 --allow-code-exec
    python -m MiniLLM.Evaluation.RunSuite --name smollm2-135m --model hf:HuggingFaceTB/SmolLM2-135M --allow-code-exec
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ALL_TASKS = ["perplexity", "hellaswag", "blimp", "mbpp", "humaneval"]
CODE_TASKS = {"mbpp", "humaneval"}


def run_suite(args) -> dict:
    tasks = args.tasks
    if not args.allow_code_exec:
        skipped = [task for task in tasks if task in CODE_TASKS]
        if skipped:
            print(f"Skipping {skipped}: pass --allow-code-exec (disposable machine only) to run them", flush=True)
        tasks = [task for task in tasks if task not in CODE_TASKS]
    if args.model.startswith("hf:") or not args.tokens_dir:
        # Perplexity uses MiniLLM's own validation shards, so it is not comparable across tokenizers.
        tasks = [task for task in tasks if task != "perplexity"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores: dict = {}
    details: dict = {}
    timings: dict = {}
    for task in tasks:
        start = time.time()
        print(f"=== {task} ===", flush=True)
        if task == "perplexity":
            from MiniLLM.Evaluation import Perplexity

            scores.update(Perplexity.run(args.model, args.tokens_dir, max_batches=200, batch_size=8))
        elif task == "hellaswag":
            from MiniLLM.Evaluation import HellaSwag

            scores.update(HellaSwag.run(args.model, args.tokenizer, args.limit))
        elif task == "blimp":
            from MiniLLM.Evaluation import BLiMP

            result = BLiMP.run(args.model, args.tokenizer, args.limit)
            details["blimp_per_task"] = result.pop("blimp_per_task")
            scores.update(result)
        elif task == "mbpp":
            from MiniLLM.Evaluation import MBPP

            scores.update(MBPP.run(args.model, args.tokenizer, args.chat, args.limit,
                                   str(out_dir / f"{args.name}.mbpp_samples.json"), args.batch_size))
        elif task == "humaneval":
            from MiniLLM.Evaluation import HumanEval

            scores.update(HumanEval.run(args.model, args.tokenizer, args.chat, args.limit,
                                        str(out_dir / f"{args.name}.humaneval_samples.json"), args.batch_size))
        timings[task] = round(time.time() - start, 1)

    scores.pop("chat_prompt", None)
    report = {"name": args.name, "model": args.model, "chat": args.chat, "limit": args.limit,
              "scores": scores, "seconds": timings, "details": details}
    output = out_dir / f"{args.name}.json"
    output.write_text(json.dumps(report, indent=2))
    print(f"Saved {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Row name in the results table, e.g. minillm-sft or gpt2")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None, help="Required for MiniLLM checkpoints")
    parser.add_argument("--tokens-dir", default=None, help="MiniLLM token shards, enables validation perplexity")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--chat", action="store_true", help="Chat prompts for code tasks (SFT/DPO models)")
    parser.add_argument("--limit", type=int, default=None, help="Cap examples per task (per BLiMP subtask)")
    parser.add_argument("--batch-size", type=int, default=16, help="Prompts per generation batch (code tasks)")
    parser.add_argument("--allow-code-exec", action="store_true")
    parser.add_argument("--out-dir", default="results/eval")
    run_suite(parser.parse_args())


if __name__ == "__main__":
    main()
