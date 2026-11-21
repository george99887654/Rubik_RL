"""Policy-guided beam search (the value head is not used).

Keeps the `width` best partial solutions. Each step:
  1. run the network on every cube in the beam -> log p(move), redundant moves masked out
  2. score every child = parent score + log p(move)   (= log of the path's probability)
  3. merge duplicate cubes (keep the best-scoring path to each), keep the top `width`
  4. stop as soon as any cube in the beam is solved
width = 1 is greedy: always take the network's top move.

    BeamSearchSolver(net, width=1024).solve(state) -> list of moves or None
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..cube import cube as C
from ..eval.optimal import _zobrist
from ..models import CubeNet


def load_net(ckpt: str | Path, device=None) -> CubeNet:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = CubeNet.from_checkpoint(torch.load(ckpt, map_location=device)).to(device).eval()
    return net


class BeamSearchSolver:
    def __init__(self, net: CubeNet, width: int = 1024, max_steps: int = 40, name: str | None = None):
        self.net = net.eval()
        self.dev = next(net.parameters()).device
        self.width, self.max_steps = width, max_steps
        self.name = name or f"beam(w={width})"
        self.perms = torch.tensor(C.PERMS, device=self.dev)
        self.allowed = torch.tensor(C.ALLOWED, device=self.dev)
        self.noncenter = torch.tensor(C.NONCENTER_IDX, device=self.dev)
        self.solved = torch.tensor(C.SOLVED, dtype=torch.uint8, device=self.dev)
        self._z = _zobrist(self.dev)
        self.nodes = 0

    def _hash(self, s):
        h = torch.zeros(len(s), dtype=torch.int64, device=self.dev)
        for i in range(C.N_STICKERS):
            h += self._z[i * C.N_FACES + s[:, i].long()]
        return h

    @torch.no_grad()
    def log_policy(self, states, prev):
        x = F.one_hot(states[:, self.noncenter].long(), C.N_FACES).flatten(1).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            logits, _ = self.net(x)
        logits = logits.float().masked_fill(~self.allowed[prev], float("-inf"))
        return F.log_softmax(logits, -1)

    @torch.no_grad()
    def solve(self, state: np.ndarray) -> list[int] | None:
        beam = torch.tensor(np.asarray(state), dtype=torch.uint8, device=self.dev)[None]
        score = torch.zeros(1, device=self.dev)
        prev = torch.full((1,), C.NO_PREV, dtype=torch.long, device=self.dev)
        history = []  # per step: (parent index into previous beam, move)
        self.nodes = 0
        for _ in range(self.max_steps + 1):
            solved = (beam == self.solved).all(-1)
            if solved.any():
                i = int(solved.nonzero()[0])
                path = []
                for parent, move in reversed(history):
                    path.append(int(move[i]))
                    i = int(parent[i])
                return path[::-1]
            if len(history) == self.max_steps:
                return None
            self.nodes += len(beam)
            logp = self.log_policy(beam, prev)                         # (B,18)
            cand = (score[:, None] + logp).flatten()                   # (B*18,)
            k = min(len(cand), 2 * self.width)
            top_score, top_idx = torch.topk(cand, k)
            ok = torch.isfinite(top_score)
            top_score, top_idx = top_score[ok], top_idx[ok]
            parent, move = top_idx // C.N_MOVES, top_idx % C.N_MOVES
            kids = torch.gather(beam[parent], 1, self.perms[move])
            # merge duplicates: candidates are sorted by score, keep each cube's first (best) entry
            h = self._hash(kids)
            uniq, inv = torch.unique(h, return_inverse=True)
            first = torch.full((len(uniq),), len(h), dtype=torch.long, device=self.dev)
            first.scatter_reduce_(0, inv, torch.arange(len(h), device=self.dev), reduce="amin")
            keep = torch.sort(first).values[: self.width]
            beam, score, prev = kids[keep], top_score[keep], move[keep]
            history.append((parent[keep], move[keep]))
        return None
