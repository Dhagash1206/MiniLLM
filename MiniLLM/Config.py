"""YAML-backed dataclass configs with `section.key=value` CLI overrides."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    ffn_hidden_dim: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    # Off by default: recomputing activations costs ~33% more compute; 125M fits a T4 without it.
    gradient_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


@dataclass
class DataConfig:
    tokens_dir: str = "data/tokens"
    # HF dataset repo holding the tokenized shards; downloaded when tokens_dir is empty.
    hub_repo: str | None = None
    tokenizer_dir: str = "artifacts/tokenizer"
    seed: int = 1234
    # Ablations: resample training windows to this domain mix, e.g. "english:0.5,code:0.5".
    # None keeps the natural mix (each source in proportion to its token count).
    domain_mix: str | None = None


@dataclass
class TrainConfig:
    total_tokens: int = 2_500_000_000
    micro_batch_size: int = 8
    tokens_per_step: int = 524_288
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 250
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 40
    log_interval: int = 10
    sample_interval: int = 500
    checkpoint_interval: int = 250
    # Stop cleanly before Kaggle's 12h session limit so the last checkpoint is uploaded.
    max_runtime_hours: float = 11.25
    checkpoint_dir: str = "checkpoints"
    keep_checkpoints: int = 2  # newest full step-XXXXX checkpoints kept locally and on the Hub
    milestone_interval: int = 1000  # fp16 weights-only snapshots kept forever (model versions)
    # A step whose loss exceeds its running average by this much blocks checkpoint saves for a while.
    loss_spike_threshold: float = 0.5
    compile: bool = False


@dataclass
class HubConfig:
    checkpoint_repo: str | None = None
    push_checkpoints: bool = True
    private: bool = True
    # Keeps the checkpoint repo small by dropping old commits after each upload.
    squash_history: bool = True


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "minillm"
    run_name: str | None = None


@dataclass
class SFTConfig:
    init_checkpoint: str = "checkpoints/pretrain/latest"
    output_dir: str = "checkpoints/sft"
    epochs: int = 2
    micro_batch_size: int = 16
    grad_accum_steps: int = 4
    max_lr: float = 6e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_seq_len: int = 1024
    # Each entry: Hub dataset, split, and how many examples to keep after the length filter (0 = all).
    datasets: list[dict] = field(default_factory=lambda: [
        {"name": "HuggingFaceTB/smol-smoltalk", "split": "train", "samples": 200_000},
        {"name": "ise-uiuc/Magicoder-OSS-Instruct-75K", "split": "train", "samples": 0},
    ])
    val_fraction: float = 0.01
    eval_interval: int = 200
    log_interval: int = 10
    checkpoint_interval: int = 500
    max_runtime_hours: float = 11.25
    # Looser than pretraining: SFT batches mix chat and code, so per-step loss is noisier.
    loss_spike_threshold: float = 1.0
    lora_enabled: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05


@dataclass
class DPOConfig:
    init_checkpoint: str = "checkpoints/sft/latest"
    output_dir: str = "checkpoints/dpo"
    dataset: str = "HuggingFaceH4/ultrafeedback_binarized"
    split: str = "train_prefs"
    max_pairs: int = 20_000
    beta: float = 0.1
    epochs: int = 1
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    max_lr: float = 2e-6
    warmup_steps: int = 50
    grad_clip: float = 1.0
    max_seq_len: int = 1024
    val_pairs: int = 500
    eval_interval: int = 100
    log_interval: int = 10
    checkpoint_interval: int = 200
    max_runtime_hours: float = 11.25
    lora_enabled: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(raw_value: str, current: Any) -> Any:
    if isinstance(current, list):
        raise ValueError("List settings (e.g. sft.datasets) can only be set in the YAML file")
    if isinstance(current, bool):
        return raw_value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(float(raw_value))
    if isinstance(current, float):
        return float(raw_value)
    if raw_value.lower() in {"none", "null"}:
        return None
    return raw_value


def _expand_env(value: Any) -> Any:
    # Lets YAML say `${HF_USERNAME}/repo` so the same config works for any account.
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded or "$HF_" in expanded:
            raise ValueError(f"Unset environment variable in config value {value!r} (export HF_USERNAME=...)")
        return expanded
    return value


def _apply_section(section_obj: Any, values: dict[str, Any], section_name: str) -> None:
    known = {f.name for f in dataclasses.fields(section_obj)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Unknown config key: {section_name}.{key}")
        setattr(section_obj, key, _expand_env(value))


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> RunConfig:
    config = RunConfig()
    if path:
        with open(path) as handle:
            raw = yaml.safe_load(handle) or {}
        for section_name, values in raw.items():
            if not hasattr(config, section_name):
                raise KeyError(f"Unknown config section: {section_name}")
            _apply_section(getattr(config, section_name), values or {}, section_name)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must look like section.key=value, got {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        section_name, key = dotted_key.split(".", 1)
        section_obj = getattr(config, section_name)
        if not hasattr(section_obj, key):
            raise KeyError(f"Unknown config key: {dotted_key}")
        setattr(section_obj, key, _expand_env(_coerce(raw_value, getattr(section_obj, key))))
    return config
