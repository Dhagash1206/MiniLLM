"""MBPP (sanitized test split) pass@1 with greedy decoding.

Generated code is executed in a subprocess with a timeout. Run this only on a disposable
machine (Kaggle/Colab), never on a machine with secrets you care about.

Usage:
    python -m MiniLLM.Evaluation.MBPP --model checkpoints/sft/latest --chat
    python -m MiniLLM.Evaluation.MBPP --model checkpoints/pretrain/latest        # base-model prompt
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from MiniLLM.Evaluation.TextGeneration import generate_text
from MiniLLM.HFExport import load_any_model

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
    code = prompt.split('"""\n\n', 1)[1] + completion
    # Stop at the first top-level statement that is not part of the function body.
    lines = code.splitlines()
    kept = [lines[0]]
    for line in lines[1:]:
        if line and not line[0].isspace() and not line.startswith(("def ", "class ", "import ", "from ", "#", "@")):
            break
        kept.append(line)
    return "\n".join(kept)


def passes_tests(code: str, problem: dict, timeout_seconds: float = 10.0) -> bool:
    imports = "\n".join(problem.get("test_imports") or [])
    program = "\n".join([imports, code, "", *problem["test_list"]])
    with tempfile.TemporaryDirectory() as work_dir:
        script = Path(work_dir) / "candidate.py"
        script.write_text(program)
        try:
            result = subprocess.run([sys.executable, "-I", str(script)], cwd=work_dir, capture_output=True,
                                    timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def run(model_path: str, tokenizer_dir: str, chat: bool, limit: int | None, output: str) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_any_model(model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    problems = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))
    if limit:
        problems = problems[:limit]

    records, passed = [], 0
    for index, problem in enumerate(problems, start=1):
        prompt = build_prompt(problem, chat)
        completion = generate_text(model, tokenizer, prompt, device, chat=chat, max_new_tokens=384)
        code = extract_code(completion, prompt, chat)
        success = passes_tests(code, problem)
        passed += success
        records.append({"task_id": problem["task_id"], "passed": success, "code": code})
        if index % 50 == 0:
            print(f"{index}/{len(problems)} pass@1={passed / index:.4f}", flush=True)

    results = {"mbpp_pass@1": passed / len(problems), "mbpp_n": len(problems), "chat_prompt": chat}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({"results": results, "samples": records}, indent=2))
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="data/tokens/tokenizer")
    parser.add_argument("--chat", action="store_true", help="Use the chat template (for SFT/DPO models)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/mbpp.json")
    args = parser.parse_args()
    run(args.model, args.tokenizer, args.chat, args.limit, args.output)


if __name__ == "__main__":
    main()
