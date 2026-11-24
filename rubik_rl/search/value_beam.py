"""Value-ranked beam search, batched over many cubes (the user's search design).

For each cube, every step:
  1. candidates = all allowed children of every cube in its beam (redundant moves masked)
  2. if any candidate is solved -> that cube is done (goal test happens BEFORE pruning)
  3. duplicates within the cube's candidates are merged
  4. keep the `width` candidates with the LOWEST value V (estimated moves to solved)
All cubes in a beam have made the same number of moves, so ranking by V equals ranking by
(moves made + V). The policy head is not used here.

    vb = ValueBeam(net, width=5)
    paths = vb.solve_batch(states, max_steps=n)     # list: move list, or None if not solved
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..cube import cube as C
from ..eval.optimal import _zobrist

_INF = float("inf")


class ValueBeam:
    def __init__(self, net, width: int = 5, eval_batch: int = 65536, noise: float = 0.0,
                 generator: torch.Generator | None = None, noise_abs: float = 0.0):
        """Exploration noise on the ranking score (the goal test and values() are unaffected):
        noise > 0:     rank by V * (1 + u), u uniform in [-noise, +noise]   (proportional)
        noise_abs > 0: rank by V + u,       u uniform in [-noise_abs, +noise_abs] (fixed size)"""
        self.net = net
        self.noise = noise
        self.noise_abs = noise_abs
        self.generator = generator
        self.dev = next(net.parameters()).device
        self.width = width
        self.eval_batch = eval_batch
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
    def values(self, states: torch.Tensor) -> torch.Tensor:
        was_training = self.net.training
        self.net.eval()
        out = []
        for i in range(0, len(states), self.eval_batch):
            x = F.one_hot(states[i:i + self.eval_batch][:, self.noncenter].long(), C.N_FACES).flatten(1).float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                _, v = self.net(x)
            out.append(v.float())
        if was_training:
            self.net.train()
        return torch.cat(out) if out else torch.zeros(0, device=self.dev)

    @torch.no_grad()
    def solve_batch(self, states, max_steps: int, collect: int = 0) -> list[list[int] | None]:
        """Solve all cubes. With collect > 0, also stores up to `collect` randomly chosen
        candidate cubes that the search evaluated in self.collected (uint8 tensor)."""
        s0 = torch.as_tensor(np.asarray(states), device=self.dev).to(torch.uint8)
        N, W, M = len(s0), self.width, C.N_MOVES
        beam = torch.zeros((N, W, C.N_STICKERS), dtype=torch.uint8, device=self.dev)
        beam[:, 0] = s0
        valid = torch.zeros((N, W), dtype=torch.bool, device=self.dev)
        valid[:, 0] = True
        prev = torch.full((N, W), C.NO_PREV, dtype=torch.long, device=self.dev)
        h_parent, h_move = [], []               # per step: (N, W) slot of parent, move taken
        found = {}                              # cube -> (step, slot, move)
        active = ~(s0 == self.solved).all(-1)
        result: list[list[int] | None] = [None] * N
        for i in (~active).nonzero().flatten().tolist():
            result[i] = []
        self.nodes = 0
        picked = []
        per_step = (collect + max_steps - 1) // max(1, max_steps)

        for t in range(max_steps):
            a = active.nonzero().flatten()
            if len(a) == 0:
                break
            A = len(a)
            B, vm, pv = beam[a], valid[a], prev[a]
            allow = self.allowed[pv] & vm[..., None]                                    # (A,W,18)
            kids = torch.gather(B[:, :, None, :].expand(A, W, M, C.N_STICKERS), 3,
                                self.perms[None, None].expand(A, W, M, C.N_STICKERS))   # (A,W,18,54)
            # 2. goal test before pruning
            is_goal = (kids == self.solved).all(-1) & allow
            done = is_goal.flatten(1).any(1)
            if done.any():
                first = is_goal.flatten(1).float().argmax(1)
                for r in done.nonzero().flatten().tolist():
                    f = int(first[r])
                    found[int(a[r])] = (t, f // M, f % M)
            # 3. values of allowed candidates, duplicates merged per cube
            flat_kids = kids.reshape(-1, C.N_STICKERS)
            flat_allow = allow.flatten()
            idx = flat_allow.nonzero().flatten()
            vals = torch.full((A * W * M,), _INF, device=self.dev)
            if len(idx):
                ck = flat_kids[idx]
                self.nodes += len(idx)
                cube_of = idx // (W * M)
                # mix the cube index into the hash so equal states of *different* cubes stay apart
                key = self._hash(ck) ^ (cube_of * 0x5851F42D4C957F2D)
                uniq, inv = torch.unique(key, return_inverse=True)
                firstocc = torch.full((len(uniq),), len(key), dtype=torch.long, device=self.dev)
                firstocc.scatter_reduce_(0, inv, torch.arange(len(key), device=self.dev), reduce="amin")
                keep = idx[firstocc]
                vals[keep] = self.values(flat_kids[keep])
                if per_step:
                    sel = keep[torch.randperm(len(keep), device=self.dev)[:per_step]]
                    picked.append(flat_kids[sel])
            vals = vals.view(A, W * M)
            if self.noise > 0:  # proportional +-noise fluctuation of the ranking score (inf stays inf)
                u = torch.rand(vals.shape, device=self.dev, generator=self.generator) * 2 - 1
                vals = vals * (1 + self.noise * u)
            if self.noise_abs > 0:  # fixed-size +-noise_abs fluctuation (inf stays inf)
                u = torch.rand(vals.shape, device=self.dev, generator=self.generator) * 2 - 1
                vals = vals + self.noise_abs * u
            # 4. keep the W lowest values
            k = min(W, W * M)
            best_v, best_i = torch.topk(vals, k, dim=1, largest=False)
            new_valid = torch.isfinite(best_v)
            parent_slot, move = best_i // M, best_i % M
            new_beam = torch.gather(kids.reshape(A, W * M, C.N_STICKERS), 1,
                                    best_i[..., None].expand(A, k, C.N_STICKERS))
            hp = torch.zeros((N, W), dtype=torch.long, device=self.dev)
            hm = torch.full((N, W), -1, dtype=torch.long, device=self.dev)
            hp[a], hm[a] = parent_slot, move
            h_parent.append(hp); h_move.append(hm)
            beam[a], valid[a], prev[a] = new_beam, new_valid, move
            # finished or dead (no valid candidates) cubes stop
            dead = ~new_valid.any(1)
            stop = torch.zeros(N, dtype=torch.bool, device=self.dev)
            stop[a] = done | dead
            active &= ~stop

        self.collected = torch.cat(picked) if picked else torch.zeros((0, C.N_STICKERS), dtype=torch.uint8, device=self.dev)
        hp_np = [x.cpu().numpy() for x in h_parent]
        hm_np = [x.cpu().numpy() for x in h_move]
        for i, (t, slot, m) in found.items():
            path = [m]
            for s in range(t - 1, -1, -1):
                path.append(int(hm_np[s][i, slot]))
                slot = int(hp_np[s][i, slot])
            result[i] = path[::-1]
        return result


class ValueBeamSolver:
    """Single-cube wrapper for the evaluation harness."""

    def __init__(self, net, width: int = 5, max_steps: int = 40, name: str | None = None):
        self.vb = ValueBeam(net, width)
        self.max_steps = max_steps
        self.name = name or f"vbeam(w={width})"

    @property
    def nodes(self):
        return self.vb.nodes

    def solve(self, state):
        return self.vb.solve_batch(np.asarray(state)[None], self.max_steps)[0]
