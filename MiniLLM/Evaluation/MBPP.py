"""MBPP (sanitized test split) pass@1 with batched greedy decoding.

Generated code runs in a resource-capped subprocess with an empty environment (see Common.run_python),
which still allows network and file access: prefer a disposable machine (Kaggle/Colab).

Usage:
    python -m MiniLLM.Evaluation.MBPP --model checkpoints/sft/latest --chat
    python -m MiniLLM.Evaluation.MBPP --model checkpoints/pretrain/latest        # base-model prompt
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


def build_prompt(problem: dict, chat: bool) -> str:
    tests = "\n".join(problem["test_list"])
    if chat:
        return f"{problem['prompt']}\nYour code should pass these tests:\n```python\n{tests}\n```"
    # Base models continue code better than they follow instructions, so prompt with a docstring.
    function_name = re.search(r"assert\s+(?:not\s+)?\(?\s*(\w+)\s*\(", problem["test_list"][0])
    signature_hint = function_name.group(1) if function_name else "solution"
    return f'"""\n{problem["prompt"]}\n{tests}\n"""\n\ndef {signature_hint}('


def extract_code(completion: str, prompt: str, chat: bool) -> str:
    if chat:
        match = CODE_BLOCK.search(completion)
        return match.group(1) if match else completion
    signature, _, body = (prompt.split('"""\n\n', 1)[1] + completion).partition("\n")
    return signature + "\n" + truncate_completion(body)


def passes_tests(code: str, problem: dict, timeout_seconds: float = 10.0) -> bool:
    imports = "\n".join(problem.get("test_imports") or [])
    program = "\n".join([imports, code, "", *problem["test_list"]])
    return run_python(program, timeout_seconds)


def run(model_path: str, tokenizer_dir: str | None, chat: bool, limit: int | None, output: str,
        batch_size: int = 16) -> dict:
    from datasets import load_dataset

    device = pick_device()
    model = load_eval_model(model_path, device)
    tokenizer = load_eval_tokenizer(model_path, tokenizer_dir)
    problems = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))
    if limit:
        problems = problems[:limit]

    records, passed = [], 0
    # Chunks of several batches keep progress visible while the length sort still has prompts to group.
    chunk_size = batch_size * 4
    for chunk_start in range(0, len(problems), chunk_size):
        chunk = problems[chunk_start : chunk_start + chunk_size]
        prompts = [build_prompt(problem, chat) for problem in chunk]
        completions = generate_texts(model, tokenizer, prompts, device, chat=chat, max_new_tokens=384,
                                     stop_strings=None if chat else CODE_STOP_STRINGS, batch_size=batch_size)
        for problem, prompt, completion in zip(chunk, prompts, completions):
            code = extract_code(completion, prompt, chat)
            success = passes_tests(code, problem)
            passed += success
            records.append({"task_id": problem["task_id"], "passed": success, "code": code})
        print(f"{len(records)}/{len(problems)} pass@1={passed / len(records):.4f}", flush=True)

    results = {"mbpp_pass@1": passed / len(problems), "mbpp_n": len(problems), "chat_prompt": chat}
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
    parser.add_argument("--output", default="results/mbpp.json")
    args = parser.parse_args()
    run(args.model, args.tokenizer, args.chat, args.limit, args.output, args.batch_size)


if __name__ == "__main__":
    main()
