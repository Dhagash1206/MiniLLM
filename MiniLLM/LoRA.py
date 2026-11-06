"""Minimal LoRA: wrap selected nn.Linear layers with trainable low-rank adapters."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        # B starts at zero so the wrapped layer initially equals the base layer exactly.
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        lora_out = (self.dropout(hidden) @ self.lora_A.t().to(hidden.dtype)) @ self.lora_B.t().to(hidden.dtype)
        return self.base(hidden) + lora_out * self.scaling

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scaling


def apply_lora(
    model: nn.Module, rank: int, alpha: float, dropout: float, targets: tuple[str, ...] = DEFAULT_TARGETS
) -> int:
    """Freeze the model, inject adapters in place, and return the trainable parameter count."""
    for param in model.parameters():
        param.requires_grad_(False)
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor for name, tensor in model.state_dict().items() if "lora_" in name}


def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Plain Transformer state dict with adapters folded in; the model itself is not modified."""
    merged: dict[str, torch.Tensor] = {}
    lora_modules = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    for name, tensor in model.state_dict().items():
        if "lora_" in name:
            continue
        if name.endswith(".base.weight"):
            module_name = name[: -len(".base.weight")]
            merged[f"{module_name}.weight"] = lora_modules[module_name].merged_weight().detach().clone()
        else:
            merged[name] = tensor
    return merged
