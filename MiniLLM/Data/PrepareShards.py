"""Stream, filter, dedup, tokenize, and write uint16 token shards (optionally push to the HF Hub).

Run on a Kaggle/Colab CPU session; training sessions then just download the shards.
With --push-to-hub, the tokenizer is uploaded first and each source right after it finishes, and sources
already on the Hub are skipped, so a session cut off by the 12 h cap can simply be rerun to resume.

Usage:
    python -m MiniLLM.Data.PrepareShards --tokenizer artifacts/tokenizer --out data/tokens \
        --total-tokens 2.6e9 --push-to-hub $HF_USERNAME/minillm-tokens
    # Split the work across sessions with --sources fineweb_edu  /  --sources python javascript sql
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from MiniLLM.Data.PretrainSources import DEFAULT_DOWNLOAD_DIR, PRETRAIN_SOURCES, SOURCES_BY_NAME, FilterStats, iter_documents
from MiniLLM.Data.Tokenizer import EOS_TOKEN

TOKEN_DTYPE = np.uint16  # vocab 32768 < 65536, halves disk vs int32


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, shard_tokens: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.buffer: list[np.ndarray] = []
        self.buffered_tokens = 0
        self.shard_index = 0
        self.total_tokens = 0
        self.files: list[str] = []

    def add(self, token_ids: np.ndarray) -> None:
        self.buffer.append(token_ids)
        self.buffered_tokens += len(token_ids)
        self.total_tokens += len(token_ids)
        if self.buffered_tokens >= self.shard_tokens:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        tokens = np.concatenate(self.buffer).astype(TOKEN_DTYPE)
        file_name = f"{self.prefix}_{self.shard_index:04d}.bin"
        tokens.tofile(self.out_dir / file_name)
        self.files.append(file_name)
        self.shard_index += 1
        self.buffer = []
        self.buffered_tokens = 0


def prepare_source(
    source_name: str,
    tokenizer: Tokenizer,
    out_dir: Path,
    train_tokens: int,
    val_tokens: int,
    shard_tokens: int,
    batch_docs: int,
    seed: int,
    download_dir: str,
) -> dict:
    spec = SOURCES_BY_NAME[source_name]
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    stats = FilterStats()
    val_writer = ShardWriter(out_dir, f"{source_name}_val", shard_tokens=10**12)
    train_writer = ShardWriter(out_dir, f"{source_name}_train", shard_tokens=shard_tokens)
    start_time = time.time()
    last_report = start_time

    def write_batch(texts: list[str]) -> None:
        for encoding in tokenizer.encode_batch(texts, add_special_tokens=False):
            token_ids = np.asarray(encoding.ids + [eos_id], dtype=TOKEN_DTYPE)
            # The first documents of each source form its held-out validation split.
            writer = val_writer if val_writer.total_tokens < val_tokens else train_writer
            writer.add(token_ids)

    pending_texts: list[str] = []
    for text in iter_documents(spec, stats=stats, seed=seed, download_dir=download_dir):
        pending_texts.append(text)
        if len(pending_texts) >= batch_docs:
            write_batch(pending_texts)
            pending_texts = []
            if train_writer.total_tokens >= train_tokens:
                break
            if time.time() - last_report > 60:
                last_report = time.time()
                rate = train_writer.total_tokens / (last_report - start_time)
                print(
                    f"[{source_name}] {train_writer.total_tokens / 1e6:,.1f}M / {train_tokens / 1e6:,.0f}M tokens "
                    f"({rate / 1e3:,.0f}k tok/s) {stats.as_dict()}",
                    flush=True,
                )
    if pending_texts:
        write_batch(pending_texts)
    val_writer.flush()
    train_writer.flush()

    meta = {
        "source": source_name,
        "domain": spec.domain,
        "train_tokens": train_writer.total_tokens,
        "val_tokens": val_writer.total_tokens,
        "train_files": train_writer.files,
        "val_files": val_writer.files,
        "filter_stats": stats.as_dict(),
        "dtype": "uint16",
    }
    (out_dir / f"{source_name}.json").write_text(json.dumps(meta, indent=2))
    if train_writer.total_tokens < train_tokens:
        print(f"Warning: {source_name} ran out of data at {train_writer.total_tokens:,} tokens")
    return meta


def upload_source(api, repo_id: str, out_dir: Path, source_name: str, meta: dict) -> None:
    file_names = meta["train_files"] + meta["val_files"] + [f"{source_name}.json"]
    # The .json goes last so a source only counts as done once all of its shards are up.
    api.upload_folder(
        folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset",
        allow_patterns=meta["train_files"] + meta["val_files"],
        commit_message=f"Add {source_name} shards",
    )
    api.upload_file(
        path_or_fileobj=str(out_dir / f"{source_name}.json"), path_in_repo=f"{source_name}.json",
        repo_id=repo_id, repo_type="dataset", commit_message=f"Mark {source_name} done",
    )
    for file_name in file_names[:-1]:
        (out_dir / file_name).unlink(missing_ok=True)
    print(f"Uploaded {source_name} to datasets/{repo_id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--out", default="data/tokens")
    parser.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR, help="Scratch space for source parquet files")
    parser.add_argument("--total-tokens", type=float, default=2.6e9, help="Train tokens across all sources")
    parser.add_argument("--val-tokens-per-domain", type=float, default=5e6)
    parser.add_argument("--shard-tokens", type=float, default=1e8)
    parser.add_argument("--batch-docs", type=int, default=1000)
    parser.add_argument("--sources", nargs="+", default=[spec.name for spec in PRETRAIN_SOURCES])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--push-to-hub", default=None, help="HF dataset repo id for the shards")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    # Ship the tokenizer with the shards so a training session needs a single download.
    shutil.copytree(args.tokenizer, out_dir / "tokenizer", dirs_exist_ok=True)

    api = None
    done_sources: set[str] = set()
    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="dataset", private=True, exist_ok=True)
        hub_files = set(api.list_repo_files(args.push_to_hub, repo_type="dataset"))
        if "tokenizer/tokenizer.json" not in hub_files:
            api.upload_folder(
                folder_path=str(out_dir / "tokenizer"), path_in_repo="tokenizer", repo_id=args.push_to_hub,
                repo_type="dataset", commit_message="Add tokenizer",
            )
        done_sources = {name for name in args.sources if f"{name}.json" in hub_files}
        if done_sources:
            print(f"Already on the Hub, skipping: {sorted(done_sources)}")

    domain_fraction: dict[str, float] = {}
    for spec in PRETRAIN_SOURCES:
        domain_fraction[spec.domain] = domain_fraction.get(spec.domain, 0.0) + spec.mix_fraction

    for source_name in args.sources:
        if source_name in done_sources:
            continue
        spec = SOURCES_BY_NAME[source_name]
        train_tokens = int(args.total_tokens * spec.mix_fraction)
        # Each domain's validation budget is split across its sources by mix weight.
        val_tokens = int(args.val_tokens_per_domain * spec.mix_fraction / domain_fraction[spec.domain])
        print(f"Preparing {source_name}: {train_tokens:,} train / {val_tokens:,} val tokens", flush=True)
        meta = prepare_source(
            source_name, tokenizer, out_dir, train_tokens, val_tokens,
            int(args.shard_tokens), args.batch_docs, args.seed, args.download_dir,
        )
        print(json.dumps(meta["filter_stats"]), f"train={meta['train_tokens']:,} val={meta['val_tokens']:,}", flush=True)
        if api is not None:
            upload_source(api, args.push_to_hub, out_dir, source_name, meta)

    if api is not None:
        print(f"Uploaded shards to datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
