#!/usr/bin/env bash
# Score GPT-2 small and SmolLM2-135M on the same harness as MiniLLM, then rebuild the results table.
# Code benchmarks execute model-written code: pass --allow-code-exec only on a disposable machine (Kaggle/Colab).
#   Scripts/RunBaselines.sh                       # HellaSwag + BLiMP
#   Scripts/RunBaselines.sh --allow-code-exec     # + MBPP + HumanEval
#   Scripts/RunBaselines.sh --limit 200           # quick partial run (marked as partial in RESULTS.md)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv/bin/python}

$PYTHON -m MiniLLM.Evaluation.RunSuite --name gpt2-small --model hf:openai-community/gpt2 "$@"
$PYTHON -m MiniLLM.Evaluation.RunSuite --name smollm2-135m --model hf:HuggingFaceTB/SmolLM2-135M "$@"
$PYTHON -m MiniLLM.Evaluation.Report
