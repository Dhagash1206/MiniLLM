"""Streaming pretraining sources with filtering and dedup applied on the fly."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from datasets import load_dataset

from MiniLLM.Data.DocumentFilters import Deduplicator, code_document_ok, english_quality_ok


@dataclass(frozen=True)
class SourceSpec:
    name: str
    domain: str  # "english" or "code"; validation loss is tracked per domain
    dataset_path: str
    dataset_name: str | None
    data_dir: str | None
    text_field: str
    mix_fraction: float


# 70% English / 30% code, matching the roadmap.
PRETRAIN_SOURCES = [
    SourceSpec("fineweb_edu", "english", "HuggingFaceFW/fineweb-edu", "sample-10BT", None, "text", 0.70),
    SourceSpec("python", "code", "bigcode/the-stack-dedup", None, "data/python", "content", 0.15),
    SourceSpec("javascript", "code", "bigcode/the-stack-dedup", None, "data/javascript", "content", 0.10),
    SourceSpec("sql", "code", "bigcode/the-stack-dedup", None, "data/sql", "content", 0.05),
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


def iter_documents(
    spec: SourceSpec,
    deduplicator: Deduplicator | None = None,
    stats: FilterStats | None = None,
    shuffle_buffer: int = 10_000,
    seed: int = 1234,
    skip_records: int = 0,
) -> Iterator[str]:
    """Yield cleaned text for one source. `skip_records` lets the tokenizer sample differ from pretraining order."""
    dataset = load_dataset(
        spec.dataset_path,
        name=spec.dataset_name,
        data_dir=spec.data_dir,
        split="train",
        streaming=True,
    )
    if shuffle_buffer:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if skip_records:
        dataset = dataset.skip(skip_records)

    deduplicator = deduplicator or Deduplicator()
    stats = stats or FilterStats()
    for record in dataset:
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
