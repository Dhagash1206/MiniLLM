"""HumanEval (164 problems) pass@1 with batched greedy decoding.

Generated code runs in a resource-capped subprocess with an empty environment (see Common.run_python),
which still allows network and file access: prefer a disposable machine (Kaggle/Colab).

Usage:
    python -m MiniLLM.Evaluation.HumanEval --model checkpoints/pretrain/latest --tokenizer data/tokens/tokenizer
    python -m MiniLLM.Evaluation.HumanEval --model checkpoints/sft/latest --tokenizer data/tokens/tokenizer --chat
    python -m MiniLLM.Evaluation.HumanEval --model hf:HuggingFaceTB/SmolLM2-135M      # baseline
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from MiniLLM.Evaluation.Common import (CODE_STOP_STRINGS, load_eval_model, load_eval_tokenizer, pick_device,
                                      run_python, truncate_completion)
from MiniLLM.Evaluation.TextGeneration import generate_texts

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|$)", re.DOTALL)


def build_program(problem: dict, completion: str, chat: bool) -> str:
    if chat:
        match = CODE_BLOCK.search(completion)
        code = match.group(1) if match else completion
        # Chat answers often restate only the body or a helper; keep the original signature available.
        if f"def {problem['entry_point']}" not in code:
            code = problem["prompt"] + truncate_completion(code)
    else:
        code = problem["prompt"] + truncate_completion(completion)
    return "\n".join([code, "", problem["test"], f"check({problem['entry_point']})"])


def run(model_path: str, tokenizer_dir: str | None, chat: bool, limit: int | None, output: str,
        batch_size: int = 16) -> dict:
    from datasets import load_dataset

    device = pick_device()
    model = load_eval_model(model_path, device)
    tokenizer = load_eval_tokenizer(model_path, tokenizer_dir)
    problems = list(load_dataset("openai/openai_humaneval", split="test"))
    if limit:
        problems = problems[:limit]

    records, passed = [], 0
    chunk_size = batch_size * 4
    for chunk_start in range(0, len(problems), chunk_size):
        chunk = problems[chunk_start : chunk_start + chunk_size]
        prompts = [f"Complete this Python function:\n```python\n{problem['prompt']}```" if chat
                   else problem["prompt"] for problem in chunk]
        completions = generate_texts(model, tokenizer, prompts, device, chat=chat, max_new_tokens=384,
                                     stop_strings=None if chat else CODE_STOP_STRINGS, batch_size=batch_size)
        for problem, completion in zip(chunk, completions):
            success = run_python(build_program(problem, completion, chat))
            passed += success
            records.append({"task_id": problem["task_id"], "passed": success, "completion": completion})
        print(f"{len(records)}/{len(problems)} pass@1={passed / len(records):.4f}", flush=True)

    results = {"humaneval_pass@1": passed / len(problems), "humaneval_n": len(problems), "chat_prompt": chat}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({"results": results, "samples": records}, indent=2))
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None, help="Required for MiniLLM checkpoints")
    parser.add_argument("--chat", action="store_true", help="Use the chat template (for SFT/DPO models)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="results/humaneval.json")
    args = parser.parse_args()
    run(args.model, args.tokenizer, args.chat, args.limit, args.output, args.batch_size)


if __name__ == "__main__":
    main()
