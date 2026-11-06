"""Run your own prompts (EvalPrompts/CustomPrompts.jsonl) and write a Markdown report for manual review.

Usage:
    python -m MiniLLM.Evaluation.PromptReport --model checkpoints/sft/latest --chat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from MiniLLM.Evaluation.TextGeneration import generate_text
from MiniLLM.HFExport import load_any_model


def run(model_path: str, tokenizer_dir: str, prompts_path: str, chat: bool, output: str, temperature: float) -> None:
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_any_model(model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    prompts = [json.loads(line) for line in Path(prompts_path).read_text().splitlines() if line.strip()]

    sections = [f"# Prompt eval: `{model_path}`\n"]
    for item in prompts:
        answer = generate_text(model, tokenizer, item["prompt"], device, chat=chat, max_new_tokens=384,
                               temperature=temperature, top_p=0.9, repetition_penalty=1.1)
        sections.append(f"## [{item.get('category', 'general')}] {item['prompt']}\n\n```\n{answer.strip()}\n```\n")
        print(f"> {item['prompt']}\n{answer.strip()}\n", flush=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(sections))
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="data/tokens/tokenizer")
    parser.add_argument("--prompts", default="EvalPrompts/CustomPrompts.jsonl")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output", default="results/prompts.md")
    args = parser.parse_args()
    run(args.model, args.tokenizer, args.prompts, args.chat, args.output, args.temperature)


if __name__ == "__main__":
    main()
