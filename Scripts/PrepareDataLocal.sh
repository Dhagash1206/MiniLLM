#!/usr/bin/env bash
# Local replacement for Notebooks/Step1PrepareData.ipynb: trains the tokenizer, then prepares every
# source in its own process (in parallel).
# Logged in to HF (`hf auth login`, write token): all sources, each uploaded to the dataset as it finishes;
# rerun to skip sources already on the Hub.
# Not logged in: public English source only (the-stack is gated), shards stay local in $WORK_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
HF_USERNAME=${HF_USERNAME:-Dhagash1206}
DATASET_REPO="$HF_USERNAME/minillm-tokens"
WORK_DIR=${WORK_DIR:-data/local_step1}
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

if .venv/bin/hf auth whoami > /dev/null 2>&1; then
    SOURCES=(fineweb_edu python javascript sql)
    TOKENIZER_ARGS=(--reuse-from-hub "$DATASET_REPO")
    PUSH_ARGS=(--push-to-hub "$DATASET_REPO")
else
    echo "Not logged in to HF: English only, no upload."
    SOURCES=(fineweb_edu)
    TOKENIZER_ARGS=(--sources fineweb_edu)
    PUSH_ARGS=()
fi

if [ ! -f artifacts/tokenizer/tokenizer.json ]; then
    $PYTHON -u -m MiniLLM.Data.Tokenizer --out artifacts/tokenizer --docs-per-source 150000 \
        "${TOKENIZER_ARGS[@]}" 2>&1 | tee "$LOG_DIR/tokenizer.log"
fi

if [ ${#PUSH_ARGS[@]} -gt 0 ]; then
    # Create the repo and upload the tokenizer once, so the parallel workers don't race on it.
    $PYTHON - "$DATASET_REPO" <<'EOF'
import sys
from huggingface_hub import HfApi
repo_id = sys.argv[1]
api = HfApi()
api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
if "tokenizer/tokenizer.json" not in api.list_repo_files(repo_id, repo_type="dataset"):
    api.upload_folder(folder_path="artifacts/tokenizer", path_in_repo="tokenizer", repo_id=repo_id,
                      repo_type="dataset", commit_message="Add tokenizer")
print(f"Tokenizer is on datasets/{repo_id}")
EOF
fi

pids=()
for source in "${SOURCES[@]}"; do
    $PYTHON -u -m MiniLLM.Data.PrepareShards --tokenizer artifacts/tokenizer --out "$WORK_DIR/tokens_$source" \
        --download-dir "$WORK_DIR/hf_files_$source" --total-tokens "${TOTAL_TOKENS:-2.6e9}" --sources "$source" \
        ${PUSH_ARGS[@]+"${PUSH_ARGS[@]}"} > "$LOG_DIR/$source.log" 2>&1 &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "All sources finished (exit status $status). Logs: $LOG_DIR"
exit $status
