"""Pretraining sources: whole parquet files are downloaded in parallel, then filtered and deduped on the fly.

Row-by-row HF streaming managed ~22 docs/s, which made Step1 overrun Kaggle's 12 h cap; downloading
whole files with a prefetch thread pool is network-bound at full bandwidth instead.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from MiniLLM.Data.DocumentFilters import Deduplicator, code_document_ok, english_quality_ok

DEFAULT_DOWNLOAD_DIR = os.environ.get("MINILLM_DOWNLOAD_DIR", "data/hf_files")

# Metadata columns the filters read; only those present in a file are loaded.
FILTER_COLUMNS = {
    "english": ["language_score", "int_score"],
    "code": ["max_stars_repo_licenses", "licenses", "max_line_length", "avg_line_length", "alphanum_fraction"],
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    domain: str  # "english" or "code"; validation loss is tracked per domain
    dataset_path: str
    file_prefix: str  # folder of parquet files inside the dataset repo
    text_field: str
    mix_fraction: float


# 70% English / 30% code, matching the roadmap.
PRETRAIN_SOURCES = [
    SourceSpec("fineweb_edu", "english", "HuggingFaceFW/fineweb-edu", "sample/10BT/", "text", 0.70),
    SourceSpec("python", "code", "bigcode/the-stack-dedup", "data/python/", "content", 0.15),
    SourceSpec("javascript", "code", "bigcode/the-stack-dedup", "data/javascript/", "content", 0.10),
    SourceSpec("sql", "code", "bigcode/the-stack-dedup", "data/sql/", "content", 0.05),
]

SOURCES_BY_NAME = {spec.name: spec for spec in PRETRAIN_SOURCES}


class FilterStats:
    def __init__(self) -> None:
        self.seen = 0
        self.kept = 0
        self.duplicates = 0
        self.rejected = 0

    def as_dict(self) -> dict[str, int]:
        return {"seen": self.seen, "kept": self.kept, "duplicates": self.duplicates, "rejected": self.rejected}


def list_source_files(spec: SourceSpec, seed: int) -> list[str]:
    from huggingface_hub import HfApi

    files = sorted(
        path for path in HfApi().list_repo_files(spec.dataset_path, repo_type="dataset")
        if path.startswith(spec.file_prefix) and path.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No parquet files under {spec.dataset_path}/{spec.file_prefix}")
    random.Random(seed).shuffle(files)
    return files


def download_file(spec: SourceSpec, file_path: str, download_dir: str | Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(spec.dataset_path, file_path, repo_type="dataset", local_dir=download_dir))


def iter_parquet_records(spec: SourceSpec, local_path: Path) -> Iterator[dict]:
    parquet_file = pq.ParquetFile(local_path)
    available = set(parquet_file.schema_arrow.names)
    columns = [spec.text_field] + [name for name in FILTER_COLUMNS[spec.domain] if name in available]
    for batch in parquet_file.iter_batches(batch_size=1024, columns=columns):
        yield from batch.to_pylist()


def iter_source_records(
    spec: SourceSpec,
    seed: int,
    download_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
    prefetch_files: int = 2,
    keep_files: bool = False,
) -> Iterator[dict]:
    """Yield raw records file by file while the next `prefetch_files` files download in the background."""
    files = list_source_files(spec, seed)
    pool = ThreadPoolExecutor(max_workers=prefetch_files)
    pending: list[Future] = [pool.submit(download_file, spec, path, download_dir) for path in files[:prefetch_files]]
    next_index = len(pending)
    try:
        while pending:
            local_path = pending.pop(0).result()
            if next_index < len(files):
                pending.append(pool.submit(download_file, spec, files[next_index], download_dir))
                next_index += 1
            try:
                yield from iter_parquet_records(spec, local_path)
            finally:
                # Fineweb files are ~2 GB each; deleting after use keeps Kaggle's disk free.
                if not keep_files:
                    local_path.unlink(missing_ok=True)
    finally:
        # Reached when the consumer stops early (token budget met): drop prefetched files too.
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True)
        if not keep_files:
            for future in pending:
                if not future.cancelled() and future.exception() is None:
                    future.result().unlink(missing_ok=True)


def shuffled(records: Iterator[dict], buffer_size: int, seed: int) -> Iterator[dict]:
    if buffer_size <= 1:
        yield from records
        return
    rng = random.Random(seed)
    buffer: list[dict] = []
    for record in records:
        if len(buffer) < buffer_size:
            buffer.append(record)
            continue
        index = rng.randrange(buffer_size)
        yield buffer[index]
        buffer[index] = record
    rng.shuffle(buffer)
    yield from buffer


def iter_documents(
    spec: SourceSpec,
    deduplicator: Deduplicator | None = None,
    stats: FilterStats | None = None,
    shuffle_buffer: int = 10_000,
    seed: int = 1234,
    download_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
) -> Iterator[str]:
    """Yield cleaned text for one source. A different `seed` gives a different file order (used by the tokenizer)."""
    deduplicator = deduplicator or Deduplicator()
    stats = stats or FilterStats()
    records = shuffled(iter_source_records(spec, seed, download_dir), shuffle_buffer, seed)
    for record in records:
        stats.seen += 1
        text = record.get(spec.text_field) or ""
        if spec.domain == "english":
            keep = english_quality_ok(text, record)
        else:
            keep = code_document_ok(text, record)
        if not keep:
            stats.rejected += 1
            continue
        if deduplicator.is_duplicate(text):
            stats.duplicates += 1
            continue
        stats.kept += 1
        yield text
