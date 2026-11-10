#!/usr/bin/env bash
# Phase 1 evidence on an Apple Silicon Mac: pretrain -> SFT -> eval suite for both, the GPT-2 small and
# SmolLM2-135M baselines on the same harness, then RESULTS.md / MODEL_CARD.md / loss curve in results/report.
# Prepare data first:   TOTAL_TOKENS=1.6e8 Scripts/PrepareDataLocal.sh
# Every stage resumes or is skipped when already done, so rerun after an interruption.
# ALLOW_CODE_EXEC=1 adds MBPP + HumanEval, which execute model-written code in the sandbox of
# MiniLLM/Evaluation/Common.run_python (resource caps, empty env; network and files stay reachable).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
CONFIG=${CONFIG:-Configs/RunLocalMac.yaml}
TOKENS_DIR=${TOKENS_DIR:-data/local_step1/tokens_fineweb_edu}
TOKENIZER="$TOKENS_DIR/tokenizer"
NAME=${NAME:-minillm-42m}
EVAL_ARGS=()
if [ "${ALLOW_CODE_EXEC:-0}" = 1 ]; then EVAL_ARGS+=(--allow-code-exec); fi

stage() { echo "=== $(date '+%H:%M:%S') $* ==="; }

stage pretrain
$PYTHON -u -m MiniLLM.Training.Pretrain --config "$CONFIG" 2>&1 | tee -a data/pretrain_local.log
stage sft
$PYTHON -u -m MiniLLM.Training.SFT --config "$CONFIG" 2>&1 | tee -a data/sft_local.log

suite() {  # name, then RunSuite arguments; skipped when results/eval/<name>.json exists
    local name=$1; shift
    if [ -f "results/eval/$name.json" ]; then echo "results/eval/$name.json exists, skipping"; return; fi
    stage "eval $name"
    $PYTHON -u -m MiniLLM.Evaluation.RunSuite --name "$name" "$@" ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}
}
suite "$NAME-base" --model checkpoints_local/pretrain/latest --tokenizer "$TOKENIZER" --tokens-dir "$TOKENS_DIR"
suite "$NAME-sft" --model checkpoints_local/sft/latest --tokenizer "$TOKENIZER" --tokens-dir "$TOKENS_DIR" --chat
suite gpt2-small --model hf:openai-community/gpt2
suite smollm2-135m --model hf:HuggingFaceTB/SmolLM2-135M

stage report
$PYTHON -m MiniLLM.Evaluation.Report --pretrain-log data/pretrain_local.log --config "$CONFIG" \
    --model-name "$NAME-sft"
