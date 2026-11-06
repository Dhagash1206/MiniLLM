"""Deterministic, resumable batch sampling over memory-mapped uint16 token shards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def ensure_tokens_available(tokens_dir: str, hub_repo: str | None) -> Path:
    tokens_path = Path(tokens_dir)
    if any(tokens_path.glob("*.json")):
        return tokens_path
    if not hub_repo:
        raise FileNotFoundError(f"No token shards in {tokens_dir} and data.hub_repo is not set")
    from huggingface_hub import snapshot_download

    print(f"Downloading token shards from datasets/{hub_repo} ...", flush=True)
    snapshot_download(repo_id=hub_repo, repo_type="dataset", local_dir=str(tokens_path))
    return tokens_path


def ensure_tokenizer_available(data_config) -> Path:
    """Fetch only the tokenizer folder from the token dataset repo when it is missing locally."""
    tokenizer_path = Path(data_config.tokenizer_dir)
    if (tokenizer_path / "tokenizer.json").exists():
        return tokenizer_path
    if not data_config.hub_repo:
        raise FileNotFoundError(f"No tokenizer at {tokenizer_path} and data.hub_repo is not set")
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id=data_config.hub_repo, repo_type="dataset", allow_patterns=["tokenizer/*"],
                             local_dir=str(Path(data_config.tokens_dir)))
    downloaded = Path(root) / "tokenizer"
    if downloaded.resolve() != tokenizer_path.resolve():
        import shutil

        shutil.copytree(downloaded, tokenizer_path, dirs_exist_ok=True)
    return tokenizer_path


def load_source_metas(tokens_dir: Path) -> list[dict]:
    metas = [json.loads(path.read_text()) for path in sorted(tokens_dir.glob("*.json"))]
    if not metas:
        raise FileNotFoundError(f"No source metadata (*.json) in {tokens_dir}")
    return metas


def _memmap(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


class PretrainSampler:
    """Splits every train shard into fixed windows and visits them in a seeded random order.

    Because each source contributes windows in proportion to its token count, a global
    shuffle reproduces the 70/30 English/code mix in every batch on average. A batch is a
    pure function of (micro-step, rank), so resuming only needs the step counter.
    """

    def __init__(self, tokens_dir: Path, seq_len: int, seed: int):
        self.seq_len = seq_len
        self.seed = seed
        self.shards: list[np.memmap] = []
        shard_ids, starts = [], []
        for meta in load_source_metas(tokens_dir):
            for file_name in meta["train_files"]:
                shard = _memmap(tokens_dir / file_name)
                shard_id = len(self.shards)
                self.shards.append(shard)
                # +1 token per window for the shifted target.
                window_count = (len(shard) - 1) // seq_len
                shard_ids.append(np.full(window_count, shard_id, dtype=np.int32))
                starts.append(np.arange(window_count, dtype=np.int64) * seq_len)
        self.window_shard = np.concatenate(shard_ids)
        self.window_start = np.concatenate(starts)
        self.num_windows = len(self.window_shard)
        self._epoch = -1
        self._order: np.ndarray | None = None

    def _order_for_epoch(self, epoch: int) -> np.ndarray:
        if epoch != self._epoch:
            self._order = np.random.default_rng(self.seed + epoch).permutation(self.num_windows)
            self._epoch = epoch
        return self._order

    def window(self, global_index: int) -> np.ndarray:
        epoch, position = divmod(global_index, self.num_windows)
        window_id = self._order_for_epoch(epoch)[position]
        shard = self.shards[self.window_shard[window_id]]
        start = self.window_start[window_id]
        return np.asarray(shard[start : start + self.seq_len + 1], dtype=np.int64)

    def get_batch(
        self, micro_step: int, rank: int, world_size: int, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_index = (micro_step * world_size + rank) * batch_size
        windows = np.stack([self.window(first_index + offset) for offset in range(batch_size)])
        tokens = torch.from_numpy(windows)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        if device.type == "cuda":
            return inputs.pin_memory().to(device, non_blocking=True), targets.pin_memory().to(device, non_blocking=True)
        return inputs.to(device), targets.to(device)


class ValidationSet:
    """Fixed sequential windows per domain ("english", "code") for comparable loss curves."""

    def __init__(self, tokens_dir: Path, seq_len: int):
        self.seq_len = seq_len
        domain_arrays: dict[str, list[np.ndarray]] = {}
        for meta in load_source_metas(tokens_dir):
            for file_name in meta["val_files"]:
                domain_arrays.setdefault(meta["domain"], []).append(_memmap(tokens_dir / file_name))
        self.domains: dict[str, np.ndarray] = {}
        for domain, arrays in domain_arrays.items():
            # Interleave sources within a domain (python/js/sql) by striding windows across them.
            windows = []
            for array in arrays:
                count = (len(array) - 1) // seq_len
                windows.append([array[i * seq_len : i * seq_len + seq_len + 1] for i in range(count)])
            merged = [window for group in zip(*windows) for window in group] if len(windows) > 1 else windows[0]
            leftovers = [window for group in windows for window in group[min(map(len, windows)) :]]
            self.domains[domain] = np.stack(merged + leftovers).astype(np.int64)

    def batches(self, domain: str, num_batches: int, batch_size: int, rank: int, world_size: int, device):
        windows = self.domains[domain]
        for batch_index in range(num_batches):
            start = (batch_index * world_size + rank) * batch_size
            if start + batch_size > len(windows):
                break
            tokens = torch.from_numpy(windows[start : start + batch_size]).to(device)
            yield tokens[:, :-1], tokens[:, 1:]
