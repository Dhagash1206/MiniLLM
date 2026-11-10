"""Turn training logs + RunSuite results into publishable artifacts (no plotting dependency needed).

Writes into --out-dir:
    loss_curve.svg   train loss and eval losses vs. tokens seen, parsed from the pretraining log
    loss_curve.csv   the same points as data
    RESULTS.md       benchmark table: every results/eval/*.json, baselines included
    MODEL_CARD.md    model card filled from the config and the scores of --model-name

Usage:
    python -m MiniLLM.Evaluation.Report --pretrain-log data/pretrain_local.log --config Configs/RunLocalMac.yaml \
        --model-name minillm-sft
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STEP_LINE = re.compile(r"^step (\d+)/(\d+) loss ([\d.]+)")
EVAL_LINE = re.compile(r"^eval (.*)$")
METRIC = re.compile(r"(\S+)=([\d.]+)")

# Columns shown in RESULTS.md, in order; missing scores render as "-".
COLUMNS = [
    ("hellaswag_acc_norm", "HellaSwag acc_norm"),
    ("blimp_acc", "BLiMP acc"),
    ("mbpp_pass@1", "MBPP pass@1"),
    ("humaneval_pass@1", "HumanEval pass@1"),
    ("english_ppl", "Val PPL (English)"),
    ("code_ppl", "Val PPL (code)"),
]


def parse_training_log(log_text: str, tokens_per_step: int) -> tuple[list[tuple], list[tuple]]:
    """Returns (train points [(step, tokens, loss)], eval points [(step, tokens, {metric: value})])."""
    train_points, eval_points = [], []
    last_step = 0
    for line in log_text.splitlines():
        if match := STEP_LINE.match(line):
            last_step = int(match.group(1))
            train_points.append((last_step, last_step * tokens_per_step, float(match.group(3))))
        elif match := EVAL_LINE.match(line):
            metrics = {key.split("/")[-1]: float(value) for key, value in METRIC.findall(match.group(1))}
            eval_points.append((last_step, last_step * tokens_per_step, metrics))
    return train_points, eval_points


def loss_curve_svg(train_points: list[tuple], eval_points: list[tuple], width: int = 720, height: int = 400) -> str:
    series = {"train": [(tokens, loss) for _, tokens, loss in train_points]}
    for _, tokens, metrics in eval_points:
        for key, value in metrics.items():
            if key.endswith("_loss"):
                series.setdefault(f"val {key.removesuffix('_loss')}", []).append((tokens, value))
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        return ""
    margin_left, margin_bottom, margin_top, margin_right = 56, 40, 16, 150
    plot_width, plot_height = width - margin_left - margin_right, height - margin_top - margin_bottom
    max_tokens = max(tokens for tokens, _ in all_points) or 1
    min_loss, max_loss = min(loss for _, loss in all_points), max(loss for _, loss in all_points)
    loss_span = (max_loss - min_loss) or 1.0

    def to_xy(tokens: float, loss: float) -> str:
        x = margin_left + tokens / max_tokens * plot_width
        y = margin_top + (max_loss - loss) / loss_span * plot_height
        return f"{x:.1f},{y:.1f}"

    colors = ["#2a6fdb", "#e0663a", "#2e9d5b", "#9b51e0"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" '
        f'font-size="12"><rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" '
        f'y2="{margin_top + plot_height}" stroke="#888"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#888"/>',
        f'<text x="{margin_left + plot_width / 2}" y="{height - 8}" text-anchor="middle">tokens seen '
        f'(max {max_tokens / 1e6:,.0f}M)</text>',
        f'<text x="{margin_left - 8}" y="{margin_top + 10}" text-anchor="end">{max_loss:.2f}</text>',
        f'<text x="{margin_left - 8}" y="{margin_top + plot_height}" text-anchor="end">{min_loss:.2f}</text>',
    ]
    for index, (name, points) in enumerate(series.items()):
        color = colors[index % len(colors)]
        polyline = " ".join(to_xy(tokens, loss) for tokens, loss in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{polyline}"/>')
        legend_y = margin_top + 16 + index * 18
        parts.append(f'<line x1="{width - margin_right + 12}" y1="{legend_y - 4}" x2="{width - margin_right + 32}" '
                     f'y2="{legend_y - 4}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{width - margin_right + 38}" y="{legend_y}">{name}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def format_score(key: str, value) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if key.endswith("_ppl") else f"{100 * value:.1f}"


def results_table(results: list[dict]) -> str:
    header = "| Model | " + " | ".join(label for _, label in COLUMNS) + " |"
    divider = "|---" * (len(COLUMNS) + 1) + "|"
    rows = [header, divider]
    for result in sorted(results, key=lambda item: item["name"]):
        cells = [format_score(key, result["scores"].get(key)) for key, _ in COLUMNS]
        rows.append(f"| {result['name']} | " + " | ".join(cells) + " |")
    notes = sorted({f"`{result['name']}`: first {result['limit']} examples per task"
                    for result in results if result.get("limit")})
    footer = ("\n\nAccuracies and pass@1 in %. Perplexity uses MiniLLM's own tokenizer, so it is only "
              "comparable between MiniLLM checkpoints.")
    if notes:
        footer += "\n\nPartial runs: " + "; ".join(notes) + "."
    return "\n".join(rows) + footer


def model_card(config_path: str | None, model_name: str, result: dict | None, has_curve: bool) -> str:
    model_lines, train_lines = [], []
    if config_path:
        from MiniLLM.Config import load_config
        from MiniLLM.Model import Transformer

        config = load_config(config_path)
        parameters = sum(parameter.numel() for parameter in Transformer(config.model).parameters())
        model_cfg = config.model
        model_lines = [
            f"- Parameters: {parameters / 1e6:.1f}M (tied embeddings: {model_cfg.tie_embeddings})",
            f"- Layers / dim / heads / KV heads: {model_cfg.n_layers} / {model_cfg.dim} / {model_cfg.n_heads} / "
            f"{model_cfg.n_kv_heads}",
            f"- Context length: {model_cfg.max_seq_len}, vocab: {model_cfg.vocab_size} (byte-level BPE)",
        ]
        train_lines = [
            f"- Pretraining tokens: {config.train.total_tokens / 1e6:,.0f}M "
            f"({config.train.total_tokens / parameters:.0f} tokens/parameter)",
            f"- Batch: {config.train.tokens_per_step:,} tokens/step, peak LR {config.train.max_lr}",
        ]
    score_lines = ["Not evaluated yet: run `python -m MiniLLM.Evaluation.RunSuite`."]
    if result:
        score_lines = [f"- {label}: {format_score(key, result['scores'][key])}"
                       for key, label in COLUMNS if key in result["scores"]]
    curve = "![Loss curve](loss_curve.svg)\n\n" if has_curve else ""
    return f"""# {model_name}

A small Llama-style language model trained from scratch with the MiniLLM pipeline.

## Architecture
{chr(10).join(model_lines) or "- See the training config."}
- RMSNorm, RoPE, SwiGLU, grouped-query attention support

## Training
{chr(10).join(train_lines) or "- See the training config."}
- Data: FineWeb-Edu (English) and permissively licensed code from The Stack (dedup), filtered for quality,
  secrets and licenses. Local runs without a Hugging Face login use English only.

{curve}## Evaluation
{chr(10).join(score_lines)}

See `RESULTS.md` for the comparison with GPT-2 small and SmolLM2-135M on the same harness.

## Limitations
- Small model: expect factual errors, repetition and weak reasoning. Not for production or advice.
- English and code only; no safety tuning beyond the SFT/DPO data.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrain-log", default=None)
    parser.add_argument("--config", default=None, help="Training config (for tokens/step and the model card)")
    parser.add_argument("--results-dir", default="results/eval")
    parser.add_argument("--model-name", default="minillm", help="RunSuite --name whose scores go in the card")
    parser.add_argument("--out-dir", default="results/report")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    has_curve = False
    if args.pretrain_log:
        tokens_per_step = 524_288
        if args.config:
            from MiniLLM.Config import load_config

            tokens_per_step = load_config(args.config).train.tokens_per_step
        train_points, eval_points = parse_training_log(Path(args.pretrain_log).read_text(), tokens_per_step)
        svg = loss_curve_svg(train_points, eval_points)
        if svg:
            (out_dir / "loss_curve.svg").write_text(svg)
            rows = ["step,tokens,train_loss"] + [f"{step},{tokens},{loss}" for step, tokens, loss in train_points]
            (out_dir / "loss_curve.csv").write_text("\n".join(rows) + "\n")
            has_curve = True

    results = [json.loads(path.read_text()) for path in sorted(Path(args.results_dir).glob("*.json"))
               if not path.name.endswith("_samples.json")]
    if results:
        (out_dir / "RESULTS.md").write_text("# Results\n\n" + results_table(results) + "\n")
    own_result = next((result for result in results if result["name"] == args.model_name), None)
    (out_dir / "MODEL_CARD.md").write_text(model_card(args.config, args.model_name, own_result, has_curve))
    print(f"Wrote report to {out_dir}")


if __name__ == "__main__":
    main()
