"""Fully connected residual network with a policy head and an optional value head.

    input  (N, 288)   one_hot(states): 48 non-centre stickers x 6 colours
      -> stem:  Linear(288, stem_dim) -> BN -> ReLU -> Linear(stem_dim, width) -> BN -> ReLU
      -> n_blocks x ResidualBlock(width)
      |-> policy head: Linear(width, 18)    logits over moves (softmax = P(move | state))
      |-> value head:  Linear(width, 1)     estimated moves remaining to solved (HTM)

Both heads share the trunk. The policy head ranks moves from one state; the value head
compares different states with each other (for A*-style search and value-iteration training).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cube import cube as C

IN_DIM = len(C.NONCENTER_IDX) * C.N_FACES  # 288


@dataclass
class CubeNetConfig:
    in_dim: int = IN_DIM
    stem_dim: int = 4096
    width: int = 1024
    n_blocks: int = 4
    n_moves: int = C.N_MOVES
    value_head: bool = True


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, width, bias=False)
        self.bn2 = nn.BatchNorm1d(width)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.bn2(self.fc2(h))
        return F.relu(x + h)


class CubeNet(nn.Module):
    def __init__(self, cfg: CubeNetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or CubeNetConfig()
        self.stem = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.stem_dim, bias=False),
            nn.BatchNorm1d(cfg.stem_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.stem_dim, cfg.width, bias=False),
            nn.BatchNorm1d(cfg.width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(cfg.width) for _ in range(cfg.n_blocks)])
        self.policy = nn.Linear(cfg.width, cfg.n_moves)
        self.value = nn.Linear(cfg.width, 1) if cfg.value_head else None

        # start each residual block as (close to) the identity: helps deep stacks train
        for b in self.blocks:
            nn.init.zeros_(b.bn2.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x: (N, 288) float. Returns (policy logits (N, 18), value (N,) or None)."""
        h = self.blocks(self.stem(x))
        logits = self.policy(h)
        value = self.value(h).squeeze(-1) if self.value is not None else None
        return logits, value

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def checkpoint(self) -> dict:
        return {"config": asdict(self.cfg), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "CubeNet":
        net = cls(CubeNetConfig(**ckpt["config"]))
        net.load_state_dict(ckpt["state_dict"])
        return net


# ------------------------------------------------------------------ helpers for search/training
_ALLOWED_T: dict[torch.device, torch.Tensor] = {}


def allowed_mask(prev_moves: torch.Tensor) -> torch.Tensor:
    """(N,) previous move indices (C.NO_PREV = none) -> (N, 18) bool mask of non-redundant moves."""
    dev = prev_moves.device
    if dev not in _ALLOWED_T:
        _ALLOWED_T[dev] = torch.tensor(C.ALLOWED, device=dev)
    return _ALLOWED_T[dev][prev_moves]


def masked_log_policy(logits: torch.Tensor, prev_moves: torch.Tensor | None = None) -> torch.Tensor:
    """log-softmax over moves, with redundant moves (same face as the previous move,
    or non-canonical opposite-face order) set to -inf."""
    if prev_moves is not None:
        logits = logits.masked_fill(~allowed_mask(prev_moves), float("-inf"))
    return F.log_softmax(logits, dim=-1)
