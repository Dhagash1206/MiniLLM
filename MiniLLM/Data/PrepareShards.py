"""Stream, filter, dedup, tokenize, and write uint16 token shards (optionally push to the HF Hub).

Run on a Kaggle/Colab CPU session once; training sessions then just download the shards.

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

from MiniLLM.Data.PretrainSources import PRETRAIN_SOURCES, SOURCES_BY_NAME, FilterStats, iter_documents
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
    for text in iter_documents(spec, stats=stats, seed=seed):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--out", default="data/tokens")
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

    domain_fraction: dict[str, float] = {}
    for spec in PRETRAIN_SOURCES:
        domain_fraction[spec.domain] = domain_fraction.get(spec.domain, 0.0) + spec.mix_fraction

    for source_name in args.sources:
        spec = SOURCES_BY_NAME[source_name]
        train_tokens = int(args.total_tokens * spec.mix_fraction)
        # Each domain's validation budget is split across its sources by mix weight.
        val_tokens = int(args.val_tokens_per_domain * spec.mix_fraction / domain_fraction[spec.domain])
        print(f"Preparing {source_name}: {train_tokens:,} train / {val_tokens:,} val tokens")
        meta = prepare_source(
            source_name, tokenizer, out_dir, train_tokens, val_tokens,
            int(args.shard_tokens), args.batch_docs, args.seed,
        )
        print(json.dumps(meta["filter_stats"]), f"train={meta['train_tokens']:,} val={meta['val_tokens']:,}")

    # Ship the tokenizer with the shards so a training session needs a single download.
    shutil.copytree(args.tokenizer, out_dir / "tokenizer", dirs_exist_ok=True)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="dataset", private=True, exist_ok=True)
        api.upload_large_folder(folder_path=str(out_dir), repo_id=args.push_to_hub, repo_type="dataset")
        print(f"Uploaded shards to datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
