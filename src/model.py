"""Step 6D/6E — multi-timeframe transformer with multi-head targets.

Three parallel branches consume the same feature window at different
resolutions (1m / 5m / 15m), each passed through a positional encoding + a
``TransformerEncoder`` and mean-pooled.  The branch embeddings are then fused
with a cross-branch transformer layer and fed to a multi-head output:

    direction   : 3-class UP / DOWN / NO-MOVE  (cross-entropy)
    returns     : future_ret_1..future_ret_N   (Huber)
    vol         : future_vol                    (Huber)
    max_up      : future_max_up                 (Huber)
    max_down    : future_max_down               (Huber)

The weighted sum of those losses is the training objective.  Everything here is
plain PyTorch and device-agnostic (CUDA if available, else CPU).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# name -> timeframe (minutes) used to resample the aligned branch windows
BRANCH_TIMEFRAMES = {"1m": 1, "5m": 5, "15m": 15}


@dataclass
class ModelConfig:
    n_features: int = 83
    d_model: int = 512
    layers: int = 8
    heads: int = 8
    dropout: float = 0.15
    seq_len: int = 30                       # 1m branch length (config key)
    branches: dict = field(default_factory=lambda: {"1m": 30, "5m": 36, "15m": 32})
    n_returns: int = 5
    n_classes: int = 3
    direction_weight: float = 1.0
    return_weight: float = 0.5
    vol_weight: float = 0.3
    max_weight: float = 0.3

    @property
    def context_len(self) -> int:
        """Minimum 1m history needed for every branch (aligned resampling)."""
        need = self.seq_len  # 1m branch
        for name, n in self.branches.items():
            k = BRANCH_TIMEFRAMES.get(name, 1)
            need = max(need, (n - 1) * k + 1)
        return need


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class BranchEncoder(nn.Module):
    def __init__(self, n_features: int, d_model: int, layers: int,
                 heads: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, n_features)
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        return h.mean(dim=1)  # (B, d_model)


class MultiTimeframeTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.branches = nn.ModuleDict({
            name: BranchEncoder(cfg.n_features, cfg.d_model, cfg.layers,
                                cfg.heads, cfg.dropout)
            for name in cfg.branches
        })
        fusion_layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.heads, dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout, activation="gelu", batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=1)

        self.direction_head = nn.Linear(cfg.d_model, cfg.n_classes)
        self.returns_head = nn.Linear(cfg.d_model, cfg.n_returns)
        self.vol_head = nn.Linear(cfg.d_model, 1)
        self.max_up_head = nn.Linear(cfg.d_model, 1)
        self.max_down_head = nn.Linear(cfg.d_model, 1)

    @staticmethod
    def _resample(x: torch.Tensor, k: int, n: int) -> torch.Tensor:
        ctx = x.size(1)
        idx = torch.arange(ctx - 1, ctx - 1 - k * n, -k, device=x.device)
        idx = idx.flip(0)  # ascending (oldest first), aligned to the latest bar
        return x[:, idx, :]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, context_len, n_features)
        reps: list[torch.Tensor] = []
        for name, encoder in self.branches.items():
            k = BRANCH_TIMEFRAMES.get(name, 1)
            n = int(self.cfg.branches[name])
            seq = self._resample(x, k, n)
            reps.append(encoder(seq))
        reps = torch.stack(reps, dim=1)          # (B, n_branches, d_model)
        fused = self.fusion(reps)                # (B, n_branches, d_model)
        pooled = fused.mean(dim=1)               # (B, d_model)

        return {
            "direction": self.direction_head(pooled),       # (B, 3)
            "returns": self.returns_head(pooled),           # (B, n_returns)
            "vol": self.vol_head(pooled).squeeze(-1),       # (B,)
            "max_up": self.max_up_head(pooled).squeeze(-1), # (B,)
            "max_down": self.max_down_head(pooled).squeeze(-1),  # (B,)
        }


def multihead_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor],
                   cfg: ModelConfig) -> tuple[torch.Tensor, dict[str, float]]:
    loss_dir = F.cross_entropy(outputs["direction"], targets["direction"])
    loss_ret = F.huber_loss(outputs["returns"], targets["returns"])
    loss_vol = F.huber_loss(outputs["vol"], targets["vol"])
    loss_max_up = F.huber_loss(outputs["max_up"], targets["max_up"])
    loss_max_down = F.huber_loss(outputs["max_down"], targets["max_down"])

    total = (cfg.direction_weight * loss_dir
             + cfg.return_weight * loss_ret
             + cfg.vol_weight * loss_vol
             + cfg.max_weight * (loss_max_up + loss_max_down))
    parts = {
        "dir": float(loss_dir.detach()),
        "ret": float(loss_ret.detach()),
        "vol": float(loss_vol.detach()),
        "max_up": float(loss_max_up.detach()),
        "max_down": float(loss_max_down.detach()),
    }
    return total, parts


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def config_to_dict(cfg: ModelConfig) -> dict:
    return asdict(cfg)

