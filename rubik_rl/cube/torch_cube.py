"""GPU (torch) versions of the batched cube ops in cube.py. Same state layout and move indices."""
from __future__ import annotations

import torch

from . import cube as C


class TorchCube:
    """Holds the move tables on a device. All states are int64 tensors (N, 54)."""

    def __init__(self, device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.perms = torch.tensor(C.PERMS, device=self.device)                 # (18,54)
        self.inverse = torch.tensor(C.INVERSE_MOVE, device=self.device)        # (18,)
        self.solved_state = torch.tensor(C.SOLVED, device=self.device).long()  # (54,)
        self.noncenter = torch.tensor(C.NONCENTER_IDX, device=self.device)
        self._allowed_count = torch.tensor(C._ALLOWED_COUNT, device=self.device)
        self._allowed_table = torch.tensor(C._ALLOWED_TABLE, device=self.device)

    def solved(self, n: int) -> torch.Tensor:
        return self.solved_state.expand(n, -1).clone()

    def apply_move(self, states: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        """states (N,54), moves (N,) -> (N,54)"""
        return torch.gather(states, 1, self.perms[moves])

    def expand(self, states: torch.Tensor) -> torch.Tensor:
        """All 18 children: (N,54) -> (N,18,54)"""
        return states[:, self.perms]

    def is_solved(self, states: torch.Tensor) -> torch.Tensor:
        return (states == self.solved_state).all(-1)

    def random_moves(self, n: int, length: int, generator: torch.Generator | None = None) -> torch.Tensor:
        out = torch.empty((n, length), dtype=torch.long, device=self.device)
        prev = torch.full((n,), C.NO_PREV, dtype=torch.long, device=self.device)
        for t in range(length):
            u = torch.rand(n, device=self.device, generator=generator)
            r = (u * self._allowed_count[prev]).long()
            prev = self._allowed_table[prev, r]
            out[:, t] = prev
        return out

    def random_walks(self, n: int, length: int, generator: torch.Generator | None = None):
        """Like cube.random_walks: returns states (n,length,54) and moves (n,length)."""
        moves = self.random_moves(n, length, generator)
        states = torch.empty((n, length, C.N_STICKERS), dtype=torch.long, device=self.device)
        cur = self.solved(n)
        for t in range(length):
            cur = self.apply_move(cur, moves[:, t])
            states[:, t] = cur
        return states, moves

    def one_hot(self, states: torch.Tensor, drop_centers: bool = True) -> torch.Tensor:
        s = states[..., self.noncenter] if drop_centers else states
        return torch.nn.functional.one_hot(s, C.N_FACES).float().flatten(-2)
