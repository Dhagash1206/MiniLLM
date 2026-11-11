"""Measure what retrieval adds: answer a question set with and without Wikipedia passages.

    python -m MiniLLM.RAG.Compare                                   # EvalPrompts/RagQuestions.jsonl
    python -m MiniLLM.RAG.Compare --model checkpoints/sft/latest --out results/rag_comparison.md
    python -m MiniLLM.RAG.Compare --offline                         # rerun on the cached pages only

An answer is correct when it contains one of the accepted answers as whole words (case-insensitive).
"Retrieval hit" says whether the passages in the prompt contained one, which separates retrieval failures
(the passages missed it) from generation failures (the model ignored or garbled them).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from MiniLLM.RAG.Answer import add_pipeline_arguments, load_pipeline

DEFAULT_QUESTIONS = "EvalPrompts/RagQuestions.jsonl"
DEFAULT_REPORT = "results/rag_comparison.md"


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def contains_answer(text: str, accepted_answers: list[str]) -> bool:
    normalized = f" {normalize(text)} "
    return any(f" {normalize(answer)} " in normalized for answer in accepted_answers)


def load_questions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as question_file:
        return [json.loads(line) for line in question_file if line.strip()]


def mark(correct: bool) -> str:
    return "ok" if correct else "x"


def markdown_cell(text: str, limit: int = 90) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[:limit - 3] + "..."


def build_report(rows: list[dict], model_description: str) -> str:
    total = len(rows)

    def rate(key: str) -> str:
        hits = sum(row[key] for row in rows)
        return f"{hits}/{total} ({100 * hits / max(total, 1):.0f}%)"

    lines = [
        "# RAG comparison: answers with and without Wikipedia retrieval",
        "",
        f"Model: {model_description}",
        "",
        "| | Correct |",
        "|---|---|",
        f"| Without retrieval | {rate('correct_without')} |",
        f"| With retrieval | {rate('correct_with')} |",
        f"| Retrieval hit (answer in the prompt passages) | {rate('retrieval_hit')} |",
        "",
        "| Question | Expected | Hit | Without retrieval | With retrieval |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {markdown_cell(row['question'], 60)} | {markdown_cell(row['answers'][0], 20)} "
            f"| {mark(row['retrieval_hit'])} "
            f"| {mark(row['correct_without'])} {markdown_cell(row['answer_without'])} "
            f"| {mark(row['correct_with'])} {markdown_cell(row['answer_with'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=None)
    add_pipeline_arguments(parser)
    args = parser.parse_args()

    questions = load_questions(args.questions)[:args.limit]
    pipeline = load_pipeline(args)
    started = time.perf_counter()
    rows = []
    for number, item in enumerate(questions, start=1):
        with_retrieval = pipeline.answer(item["question"])
        without_retrieval = pipeline.answer(item["question"], use_retrieval=False)
        passages_text = " ".join(passage.text for passage in with_retrieval.passages)
        row = {
            "question": item["question"],
            "answers": item["answers"],
            "answer_with": with_retrieval.answer,
            "answer_without": without_retrieval.answer,
            "correct_with": contains_answer(with_retrieval.answer, item["answers"]),
            "correct_without": contains_answer(without_retrieval.answer, item["answers"]),
            "retrieval_hit": contains_answer(passages_text, item["answers"]),
            "sources": [passage.label for passage in with_retrieval.passages],
        }
        rows.append(row)
        print(f"[{number}/{len(questions)}] hit={row['retrieval_hit']!s:5} without={row['correct_without']!s:5} "
              f"with={row['correct_with']!s:5} {item['question']}", flush=True)

    model_description = (f"`{args.model or 'default checkpoint'}`, {'ChatML' if pipeline.chat else 'base'} prompt, "
                         f"{pipeline.token_budget}-token budget, top {pipeline.top_k} of {pipeline.pages} pages")
    report = build_report(rows, model_description)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    Path(args.out).with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(report)
    print(f"Wrote {args.out} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
