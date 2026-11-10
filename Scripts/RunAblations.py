"""Cheap ablations on the ~27M proxy (roadmap Phase 3), compared by final validation loss at a fixed token budget.

One variable changes per run: English/code mix, peak learning rate, grouped-query attention, and (with
--dedup-tokens-dir) MinHash near-dedup. Each variant is a normal pretraining run with config overrides,
logged to results/ablations/<name>.log; finished variants are skipped on rerun, so the sweep can be split
across Kaggle sessions. --wandb logs every variant to the same W&B project so the curves are comparable.

Usage:
    python Scripts/RunAblations.py                                  # all variants, Configs/Proxy30M.yaml
    python Scripts/RunAblations.py --only base gqa_kv2 --total-tokens 5e7
    python Scripts/RunAblations.py --dedup-tokens-dir data/tokens_neardedup --wandb
    python Scripts/RunAblations.py --summary-only                   # just rebuild SUMMARY.md
    python Scripts/RunAblations.py --config Configs/Proxy30M.yaml data.tokens_dir=data/local_step1/tokens_fineweb_edu
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# base = natural 70/30 mix, peak LR 1e-3, 6 KV heads. The proxy has 6 heads, so 2 KV heads is the same
# 3:1 grouping as the roadmap's 12 -> 4 at full size.
VARIANTS: dict[str, list[str]] = {
    "base": [],
    "mix_50_50": ["data.domain_mix=english:0.5,code:0.5"],
    "mix_85_15": ["data.domain_mix=english:0.85,code:0.15"],
    "lr_4e-4": ["train.max_lr=4e-4"],
    "lr_6e-4": ["train.max_lr=6e-4"],
    "gqa_kv2": ["model.n_kv_heads=2"],
    "near_dedup": [],  # data.tokens_dir comes from --dedup-tokens-dir
}
MIX_VARIANTS = {"mix_50_50", "mix_85_15"}
EVAL_LINE = re.compile(r"^eval (.*)$", re.MULTILINE)
METRIC = re.compile(r"val/(\w+)_loss=([\d.]+)")


def final_val_losses(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    evals = EVAL_LINE.findall(log_path.read_text())
    return {domain: float(value) for domain, value in METRIC.findall(evals[-1])} if evals else {}


def is_finished(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    log_text = log_path.read_text()
    # The final step is not always a log_interval multiple, but the final checkpoint save is always printed.
    totals = re.findall(r" steps=([\d,]+) ", log_text)
    return bool(totals) and f"Saved checkpoint at step {totals[-1].replace(',', '')}\n" in log_text


def has_code_domain(tokens_dir: str) -> bool:
    import json

    return any(json.loads(path.read_text()).get("domain") == "code" for path in Path(tokens_dir).glob("*.json"))


def write_summary(out_dir: Path) -> None:
    rows = []
    for name, overrides in VARIANTS.items():
        losses = final_val_losses(out_dir / f"{name}.log")
        if losses:
            described = " ".join(overrides) or ("near-dedup shards" if name == "near_dedup" else "(defaults)")
            rows.append((name, described, losses))
    if not rows:
        return
    domains = sorted({domain for _, _, losses in rows for domain in losses})
    base = next((losses for name, _, losses in rows if name == "base"), None)
    lines = ["# Ablations (proxy model, final validation loss; lower is better)", "",
             "| Variant | Overrides | " + " | ".join(f"val {domain}" for domain in domains) + " |",
             "|---" * (len(domains) + 2) + "|"]
    for name, overrides, losses in rows:
        cells = []
        for domain in domains:
            value = losses.get(domain)
            delta = f" ({value - base[domain]:+.3f})" if base and value is not None and name != "base" else ""
            cells.append(f"{value:.3f}{delta}" if value is not None else "-")
        lines.append(f"| {name} | `{overrides}` | " + " | ".join(cells) + " |")
    lines += ["", "Mix variants change the training mix, so compare them on each domain separately, not on the mean.",
              "near_dedup trains on different shards, so its validation sets differ too: compare it on the "
              "downstream evals (RunSuite), not on this loss alone."]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print((out_dir / "SUMMARY.md").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="Configs/Proxy30M.yaml")
    parser.add_argument("--only", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--total-tokens", type=float, default=None, help="Override train.total_tokens for all")
    parser.add_argument("--out-dir", default="results/ablations")
    parser.add_argument("--dedup-tokens-dir", default=None,
                        help="Shards prepared with PrepareShards --near-dedup; enables the near_dedup variant")
    parser.add_argument("--wandb", action="store_true", help="Log every variant to the config's W&B project")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Extra section.key=value overrides applied to every variant")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        from MiniLLM.Config import load_config

        tokens_dir = load_config(args.config, args.overrides).data.tokens_dir
        for name in args.only:
            log_path = out_dir / f"{name}.log"
            if is_finished(log_path):
                print(f"[{name}] already finished, skipping")
                continue
            if name in MIX_VARIANTS and Path(tokens_dir).exists() and not has_code_domain(tokens_dir):
                print(f"[{name}] skipped: no code shards in {tokens_dir}")
                continue
            variant_overrides = list(VARIANTS[name])
            if name == "near_dedup":
                if not args.dedup_tokens_dir:
                    print(f"[{name}] skipped: pass --dedup-tokens-dir (shards built with --near-dedup)")
                    continue
                variant_overrides.append(f"data.tokens_dir={args.dedup_tokens_dir}")
            overrides = [*args.overrides, *variant_overrides, f"train.checkpoint_dir=checkpoints_ablations/{name}",
                         f"wandb.run_name=ablation-{name}"]
            if args.wandb:
                overrides.append("wandb.enabled=true")
            if args.total_tokens:
                overrides.append(f"train.total_tokens={int(args.total_tokens)}")
            command = [sys.executable, "-u", "-m", "MiniLLM.Training.Pretrain", "--config", args.config, *overrides]
            print(f"[{name}] {' '.join(command)}", flush=True)
            with log_path.open("a") as log_file:
                result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                print(f"[{name}] failed (exit {result.returncode}); see {log_path}", flush=True)
    write_summary(out_dir)


if __name__ == "__main__":
    main()
