"""Shared training plumbing: DDP setup, LR schedule, checkpoints (local + HF Hub), logging."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from MiniLLM.Config import ModelConfig, RunConfig
from MiniLLM.Model import Transformer


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistInfo:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        rank, local_rank, world_size = 0, 0, 1

    forced_device = os.environ.get("MINILLM_DEVICE")  # e.g. "cpu" to test DDP with gloo on a laptop
    if forced_device:
        device = torch.device(forced_device)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return DistInfo(rank, local_rank, world_size, device)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value


def any_rank_true(flag: bool, device: torch.device) -> bool:
    # Keeps all ranks in agreement on stop decisions so nobody hangs in a collective.
    tensor = torch.tensor(1.0 if flag else 0.0, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item() > 0)


def unwrap(model: torch.nn.Module) -> Transformer:
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


class CheckpointManager:
    """Saves `<dir>/latest` atomically and mirrors it to a HF model repo in a background thread."""

    def __init__(self, local_dir: str | Path, hub_repo: str | None, push: bool, private: bool, squash: bool,
                 path_in_repo: str):
        self.local_dir = Path(local_dir)
        self.hub_repo = hub_repo if push else None
        self.private = private
        self.squash = squash
        self.path_in_repo = path_in_repo
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_upload: Future | None = None

    @property
    def latest_dir(self) -> Path:
        return self.local_dir / "latest"

    def save(self, model, optimizer, scaler, state: dict, config: RunConfig, force: bool = False,
             extra_files: dict[str, object] | None = None, model_state: dict | None = None) -> bool:
        """Returns False when skipped because the previous upload is still reading `latest`."""
        self.local_dir.mkdir(parents=True, exist_ok=True)
        if self.pending_upload is not None and not self.pending_upload.done() and not force:
            # Waiting here could exceed the NCCL barrier timeout on slow uploads; just skip this one.
            print("[ckpt] previous upload still running; skipping this checkpoint", flush=True)
            return False
        self.wait_for_upload()
        staging_dir = self.local_dir / "latest.tmp"
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True)
        torch.save(model_state if model_state is not None else model.state_dict(), staging_dir / "model.pt")
        for file_name, payload in (extra_files or {}).items():
            torch.save(payload, staging_dir / file_name)
        training_state = {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if scaler else None}
        torch.save(training_state, staging_dir / "training_state.pt")
        (staging_dir / "state.json").write_text(json.dumps(state, indent=2))
        (staging_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
        shutil.rmtree(self.latest_dir, ignore_errors=True)
        staging_dir.rename(self.latest_dir)

        if self.hub_repo:
            self.pending_upload = self.executor.submit(self._upload, state.get("step", 0))
        return True

    def _upload(self, step: int) -> None:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            api.create_repo(self.hub_repo, repo_type="model", private=self.private, exist_ok=True)
            api.upload_folder(
                folder_path=str(self.latest_dir),
                repo_id=self.hub_repo,
                path_in_repo=f"{self.path_in_repo}/latest",
                commit_message=f"{self.path_in_repo} checkpoint step {step}",
            )
            if self.squash:
                api.super_squash_history(repo_id=self.hub_repo, repo_type="model")
            print(f"[hub] uploaded {self.path_in_repo} step {step} to {self.hub_repo}", flush=True)
        except Exception as error:  # A failed upload must not kill a long training run.
            print(f"[hub] upload failed at step {step}: {error}", flush=True)

    def wait_for_upload(self) -> None:
        if self.pending_upload is not None:
            self.pending_upload.result()
            self.pending_upload = None

    def find_resume_dir(self) -> Path | None:
        if (self.latest_dir / "state.json").exists():
            return self.latest_dir
        if not self.hub_repo:
            return None
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import RepositoryNotFoundError

        try:
            download_root = snapshot_download(
                repo_id=self.hub_repo,
                allow_patterns=[f"{self.path_in_repo}/latest/*"],
                local_dir=str(self.local_dir / "hub_download"),
            )
        except RepositoryNotFoundError:
            return None
        candidate = Path(download_root) / self.path_in_repo / "latest"
        return candidate if (candidate / "state.json").exists() else None


def load_model_weights(checkpoint_dir: str | Path, device: torch.device | str = "cpu") -> tuple[Transformer, dict]:
    checkpoint_dir = Path(checkpoint_dir)
    saved_config = json.loads((checkpoint_dir / "config.json").read_text())
    model_config = ModelConfig(**saved_config["model"])
    model = Transformer(model_config)
    state_dict = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device), saved_config


def resolve_checkpoint(path_or_repo: str, path_in_repo: str = "pretrain/latest") -> Path:
    """Accepts a local checkpoint dir or `hub:<repo_id>[:<path_in_repo>]`."""
    if not path_or_repo.startswith("hub:"):
        return Path(path_or_repo)
    from huggingface_hub import snapshot_download

    parts = path_or_repo[len("hub:") :].split(":", 1)
    repo_id = parts[0]
    subdir = parts[1] if len(parts) > 1 else path_in_repo
    root = snapshot_download(repo_id=repo_id, allow_patterns=[f"{subdir}/*"], local_dir=f"checkpoints/hub/{repo_id}")
    return Path(root) / subdir


class WandbLogger:
    def __init__(self, enabled: bool, project: str, run_name: str | None, config: dict, run_id: str | None):
        self.run = None
        if not enabled:
            return
        try:
            import wandb

            self.run = wandb.init(project=project, name=run_name, config=config, id=run_id, resume="allow")
        except Exception as error:
            print(f"[wandb] disabled: {error}", flush=True)

    @property
    def run_id(self) -> str | None:
        return self.run.id if self.run else None

    def log(self, metrics: dict, step: int) -> None:
        if self.run:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run:
            self.run.finish()


class Timer:
    def __init__(self, max_runtime_hours: float):
        self.start = time.time()
        self.deadline = self.start + max_runtime_hours * 3600

    def out_of_time(self) -> bool:
        return time.time() >= self.deadline

    def elapsed_hours(self) -> float:
        return (time.time() - self.start) / 3600
