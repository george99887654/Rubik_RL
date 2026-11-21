"""Exact optimal solver for cubes up to 2*D moves from solved (default D=6, so up to 12 HTM).

Meet in the middle:
  1. Backward table: breadth-first search outward from solved to depth D. Every state
     within D moves is stored as a 64-bit hash, with its depth and a parent pointer.
     For D=6 that is about 8.2 M states (~110 MB), built in seconds on the GPU.
  2. Forward search: from the query state, expand all non-redundant move sequences of
     length k = 0, 1, 2, ... and look every frontier state up in the table. The first k
     with any hit gives the optimal length k + (smallest table depth among the hits).

Why the first k is optimal: if the true distance is L, no state k moves from the query can
be within D of solved while k < L - D, and at k = max(0, L - D) the state on an optimal path
is in the table at depth L - k, while no frontier state can be closer than L - k.

States are hashed by summing random 64-bit numbers per (sticker, colour). A hash collision is
astronomically unlikely, and every returned solution is replayed and checked anyway.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..cube import cube as C

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "data" / "optimal_table_d{depth}.pt"

_Z_SEED = 0x5EED_CAFE


def _zobrist(device) -> torch.Tensor:
    rng = np.random.default_rng(_Z_SEED)
    z = rng.integers(-(2**63), 2**63 - 1, size=(C.N_STICKERS, C.N_FACES), dtype=np.int64)
    return torch.tensor(z.reshape(-1), device=device)


class OptimalSolver:
    name = "optimal"

    def __init__(self, depth: int = 6, device: str | None = None, cache: bool = True,
                 chunk: int = 200_000):
        self.depth = depth
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.chunk = chunk
        self.perms = torch.tensor(C.PERMS, device=self.device)
        self.allowed = torch.tensor(C.ALLOWED, device=self.device)
        self.inverse = C.INVERSE_MOVE
        self._z = _zobrist(self.device)
        self._offsets = torch.arange(C.N_STICKERS, device=self.device) * C.N_FACES

        path = Path(str(DEFAULT_CACHE).format(depth=depth))
        if cache and path.exists():
            t = torch.load(path, map_location=self.device)
        else:
            t = self._build()
            if cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.cpu() for k, v in t.items()}, path)
                t = {k: v.to(self.device) for k, v in t.items()}
        self.hash, self.tdepth, self.parent, self.move = t["hash"], t["depth"], t["parent"], t["move"]
        order = torch.argsort(self.hash)
        self.sorted_hash = self.hash[order]
        self.sorted_idx = order
        self.layer_sizes = torch.bincount(self.tdepth.long(), minlength=depth + 1).tolist()

    # ------------------------------------------------------------------ helpers
    def hash_states(self, s: torch.Tensor) -> torch.Tensor:
        """(N,54) uint8 -> (N,) int64. Sum wraps around on overflow, which is fine for hashing."""
        return self._z[s.long() + self._offsets].sum(-1)

    def _expand(self, states: torch.Tensor, prev: torch.Tensor):
        """Children of states (uint8 (N,54)) that are allowed after prev moves.
        Returns child states, parent row index, move."""
        mask = self.allowed[prev]                              # (N,18)
        rows, moves = mask.nonzero(as_tuple=True)
        kids = torch.gather(states[rows], 1, self.perms[moves])
        return kids, rows, moves

    def _lookup(self, h: torch.Tensor):
        """Returns (found mask, table index) for hashes h."""
        pos = torch.searchsorted(self.sorted_hash, h).clamp_(max=len(self.sorted_hash) - 1)
        found = self.sorted_hash[pos] == h
        return found, self.sorted_idx[pos]

    # ------------------------------------------------------------------ build
    def _build(self) -> dict[str, torch.Tensor]:
        dev = self.device
        solved = torch.tensor(C.SOLVED, dtype=torch.uint8, device=dev)[None]
        hashes = [self.hash_states(solved)]
        depths = [torch.zeros(1, dtype=torch.int8, device=dev)]
        parents = [torch.full((1,), -1, dtype=torch.int32, device=dev)]
        moves = [torch.full((1,), -1, dtype=torch.int8, device=dev)]
        seen = hashes[0].clone()                      # sorted hashes of all earlier layers
        frontier, fprev, base = solved, torch.full((1,), C.NO_PREV, device=dev), 0

        for d in range(1, self.depth + 1):
            hs, ps, ms, ks = [], [], [], []
            for i in range(0, len(frontier), self.chunk):
                kids, rows, mv = self._expand(frontier[i:i + self.chunk], fprev[i:i + self.chunk])
                hs.append(self.hash_states(kids))
                ps.append(rows + i + base)
                ms.append(mv)
                if d < self.depth:
                    ks.append(kids)
            h, p, m = torch.cat(hs), torch.cat(ps), torch.cat(ms)
            # keep the first occurrence of each new hash, drop states already seen
            uniq, inv = torch.unique(h, return_inverse=True)
            first = torch.full((len(uniq),), len(h), dtype=torch.long, device=dev)
            first.scatter_reduce_(0, inv, torch.arange(len(h), device=dev), reduce="amin")
            pos = torch.searchsorted(seen, uniq).clamp_(max=len(seen) - 1)
            new = seen[pos] != uniq
            keep = first[new]
            base += len(frontier)
            hashes.append(h[keep])
            depths.append(torch.full((len(keep),), d, dtype=torch.int8, device=dev))
            parents.append(p[keep].int())
            moves.append(m[keep].to(torch.int8))
            seen = torch.sort(torch.cat([seen, h[keep]])).values
            if d < self.depth:
                frontier, fprev = torch.cat(ks)[keep], m[keep]
        return {
            "hash": torch.cat(hashes), "depth": torch.cat(depths),
            "parent": torch.cat(parents), "move": torch.cat(moves),
        }

    def _path_to_solved(self, idx: int) -> list[int]:
        """Moves that take table entry idx back to solved."""
        if not hasattr(self, "_p_np"):
            self._p_np = self.parent.cpu().numpy()
            self._m_np = self.move.cpu().numpy()
        out = []
        while self._p_np[idx] >= 0:
            out.append(int(self.inverse[self._m_np[idx]]))
            idx = int(self._p_np[idx])
        return out

    # ------------------------------------------------------------------ solve
    @torch.no_grad()
    def solve(self, state: np.ndarray, max_forward: int | None = None) -> list[int] | None:
        """Optimal move list for one state, or None if it is more than
        depth + max_forward moves from solved (default max_forward = depth)."""
        max_forward = self.depth if max_forward is None else max_forward
        dev = self.device
        front = torch.tensor(np.asarray(state), dtype=torch.uint8, device=dev)[None]
        prev = torch.full((1,), C.NO_PREV, device=dev)
        back_ptrs: list[tuple[torch.Tensor, torch.Tensor]] = []  # per layer: (parent row, move)
        self.nodes = 0

        for k in range(max_forward + 1):
            if k > 0:
                ks, rows, mvs = [], [], []
                for i in range(0, len(front), self.chunk):
                    kids, r, m = self._expand(front[i:i + self.chunk], prev[i:i + self.chunk])
                    ks.append(kids); rows.append(r + i); mvs.append(m)
                front, prow, prev = torch.cat(ks), torch.cat(rows), torch.cat(mvs)
                back_ptrs.append((prow, prev))
            self.nodes += len(front)
            found, tidx = self._lookup(self.hash_states(front))
            if found.any():
                cand = found.nonzero(as_tuple=True)[0]
                best = cand[torch.argmin(self.tdepth[tidx[cand]])]
                row = int(best)
                forward = []
                for prow, pm in reversed(back_ptrs):
                    forward.append(int(pm[row]))
                    row = int(prow[row])
                forward.reverse()
                sol = forward + self._path_to_solved(int(tidx[best]))
                if C.is_solved(C.apply_moves(np.asarray(state), sol)):
                    return sol
                raise RuntimeError("hash collision: replayed solution does not solve the cube")
        return None

    def distance(self, state: np.ndarray) -> int:
        """Exact distance, or -1 if beyond 2*depth."""
        sol = self.solve(state)
        return -1 if sol is None else len(sol)
