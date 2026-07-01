from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse import SparseTensor


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        device = linear.weight.device
        self.lora_down = nn.Parameter(torch.zeros(rank, linear.in_features, device=device))
        self.lora_up = nn.Parameter(torch.zeros(linear.out_features, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_down, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up)

        for p in self.linear.parameters():
            p.requires_grad = False

    def _forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.linear.weight, self.linear.bias)
        lora_x = self.dropout(x).to(self.lora_down.dtype)
        lora_y = F.linear(F.linear(lora_x, self.lora_down), self.lora_up) * self.scale
        return y + lora_y.to(y.dtype)

    def forward(self, x):
        if isinstance(x, SparseTensor):
            return x.replace(self._forward_tensor(x.feats))
        return self._forward_tensor(x)


def _matches(name: str, patterns: Optional[Iterable[str]]) -> bool:
    if not patterns:
        return True
    return any(name.startswith(pattern) if pattern.endswith(".") else pattern in name for pattern in patterns)


def _replace_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_patterns: Optional[Iterable[str]] = None,
) -> int:
    for p in model.parameters():
        p.requires_grad = False

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear) and _matches(name, target_patterns)
    ]
    for name, module in targets:
        _replace_module(model, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(module, LoRALinear) for module in model.modules())


def lora_state_dict(model: nn.Module) -> dict:
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if ".lora_down" in name or ".lora_up" in name
    }
