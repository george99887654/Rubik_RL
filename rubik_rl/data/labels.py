"""Exact training labels: distance to solved and the set of all optimal next moves.

For a cube s at distance d, move m is optimal  <=>  d(s after m) = d - 1.
We find the distances of all 18 children of many cubes at once:

  DistanceTable   every state within D moves of solved (default D = 7: 106 M states),
                  stored as sorted 64-bit hashes plus their distance (~1 GB, GPU).
  ExactLabeler    for each cube, searches forward from all 18 children together
                  (k = 0, 1, 2, ...) and looks every frontier state up in the table.
                  At the first k where any child of a cube hits the table, the children
                  with the smallest k + table distance are exactly the optimal moves, and
                  d(s) = 1 + that distance. Children's first forward move is not pruned,
                  so no child distance can be missed.

With D = 7 and max_forward = 5 this labels every cube up to 13 moves from solved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..cube import cube as C
from ..eval.optimal import _zobrist

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "distance_table_d{depth}.pt"
_BIG = 127


class DistanceTable:
    def __init__(self, depth: int = 7, device: str | None = None, cache: bool = True,
                 chunk: int = 1_000_000, verbose: bool = False):
        self.depth = depth
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.chunk = chunk
        self.verbose = verbose
        self.perms = torch.tensor(C.PERMS, device=self.device)
        self.allowed = torch.tensor(C.ALLOWED, device=self.device)
        self._z = _zobrist(self.device)
        path = Path(str(CACHE).format(depth=depth))
        if cache and path.exists():
            t = torch.load(path, map_location=self.device)
            self.hash, self.dist = t["hash"], t["dist"]
        else:
            self.hash, self.dist = self._build()
            if cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"hash": self.hash.cpu(), "dist": self.dist.cpu()}, path)
        self.layer_sizes = torch.bincount(self.dist.long(), minlength=depth + 1).tolist()

    # ------------------------------------------------------------------ shared helpers
    def hash_states(self, s: torch.Tensor) -> torch.Tensor:
        """(N,54) uint8 -> (N,) int64; same hash as OptimalSolver. Loops over stickers so
        memory stays O(N)."""
        h = torch.zeros(len(s), dtype=torch.int64, device=self.device)
        for i in range(C.N_STICKERS):
            h += self._z[i * C.N_FACES + s[:, i].long()]
        return h

    def expand(self, states: torch.Tensor, prev: torch.Tensor | None):
        """Children of uint8 states, pruned by prev (None = no pruning).
        Returns (children, parent row, move)."""
        kids, rows, moves = [], [], []
        for m in range(C.N_MOVES):
            if prev is None:
                r = torch.arange(len(states), device=self.device)
            else:
                r = self.allowed[prev, m].nonzero(as_tuple=True)[0]
            if len(r):
                kids.append(states[r][:, self.perms[m]])
                rows.append(r)
                moves.append(torch.full_like(r, m))
        return torch.cat(kids), torch.cat(rows), torch.cat(moves)

    def lookup(self, h: torch.Tensor) -> torch.Tensor:
        """Distance for each hash, or -1 if the state is more than `depth` moves away."""
        pos = torch.searchsorted(self.hash, h).clamp_(max=len(self.hash) - 1)
        d = self.dist[pos].long()
        return torch.where(self.hash[pos] == h, d, torch.full_like(d, -1))

    # ------------------------------------------------------------------ build
    def _build(self):
        dev = self.device
        solved = torch.tensor(C.SOLVED, dtype=torch.uint8, device=dev)[None]
        seen = self.hash_states(solved)
        dists = [torch.zeros(1, dtype=torch.int8, device=dev)]
        hashes = [seen.clone()]
        frontier, fprev = solved, torch.full((1,), C.NO_PREV, device=dev)
        for d in range(1, self.depth + 1):
            hs, ks, ms = [], [], []
            for i in range(0, len(frontier), self.chunk):
                kids, rows, mv = self.expand(frontier[i:i + self.chunk], fprev[i:i + self.chunk])
                h = self.hash_states(kids)
                pos = torch.searchsorted(seen, h).clamp_(max=len(seen) - 1)
                new = seen[pos] != h
                hs.append(h[new])
                if d < self.depth:
                    ks.append(kids[new]); ms.append(mv[new])
            h = torch.cat(hs)
            if d < self.depth:
                uniq, inv = torch.unique(h, return_inverse=True)
                first = torch.full((len(uniq),), len(h), dtype=torch.long, device=dev)
                first.scatter_reduce_(0, inv, torch.arange(len(h), device=dev), reduce="amin")
                frontier, fprev = torch.cat(ks)[first], torch.cat(ms)[first]
                del ks, ms
            else:
                uniq = torch.unique(h)
            del hs, h
            hashes.append(uniq)
            dists.append(torch.full((len(uniq),), d, dtype=torch.int8, device=dev))
            seen = torch.sort(torch.cat([seen, uniq])).values
            if self.verbose:
                print(f"  depth {d}: {len(uniq):,} states", flush=True)
        h, dist = torch.cat(hashes), torch.cat(dists)
        order = torch.argsort(h)
        return h[order], dist[order]


class ExactLabeler:
    def __init__(self, table: DistanceTable | None = None, max_forward: int = 5,
                 max_frontier: int = 30_000_000):
        self.table = table or DistanceTable()
        self.dev = self.table.device
        self.max_forward = max_forward
        self.max_frontier = max_frontier
        self.solved_hash = self.table.hash_states(
            torch.tensor(C.SOLVED, dtype=torch.uint8, device=self.dev)[None])[0]
        self.nodes = 0

    @property
    def horizon(self) -> int:
        """Largest cube distance that can be labelled."""
        return self.table.depth + self.max_forward + 1

    @torch.no_grad()
    def label(self, states, max_dist: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """states (N,54) -> (distance (N,) int16, optimal-move mask (N,18) bool).

        max_dist: a known upper bound on every cube's distance (e.g. its scramble length);
        it limits the forward search to what is needed. Cubes beyond the searched horizon
        get distance -1 and an empty mask."""
        mf = self.max_forward if max_dist is None else             int(min(self.max_forward, max(0, max_dist - 1 - self.table.depth)))
        s = torch.as_tensor(np.asarray(states), device=self.dev).to(torch.uint8)
        dist = np.empty(len(s), dtype=np.int16)
        mask = np.empty((len(s), C.N_MOVES), dtype=bool)
        # size groups so the worst-case frontier (18 children x 18 x 13.4^(mf-1)) stays bounded
        per_cube = C.N_MOVES * (C.N_MOVES * 13.4 ** (mf - 1) if mf > 0 else 1)
        group = int(max(1, min(200_000, self.max_frontier // per_cube)))
        for i in range(0, len(s), group):
            d, m = self._label_group(s[i:i + group], mf)
            dist[i:i + group], mask[i:i + group] = d, m
        return dist, mask

    def _label_group(self, s: torch.Tensor, max_forward: int):
        T, dev = self.table, self.dev
        n = len(s)
        kids, parent, move = T.expand(s, None)              # all 18 children of each cube
        child_id = parent * C.N_MOVES + move                # index = cube*18 + move
        child_dist = torch.full((n * C.N_MOVES,), -1, dtype=torch.long, device=dev)
        done = torch.zeros(n, dtype=torch.bool, device=dev)

        front, fid, prev = kids, child_id, None
        for k in range(max_forward + 1):
            if k > 0:
                front, rows, mv = T.expand(front, prev)
                fid, prev = fid[rows], mv
            self.nodes += len(front)
            td = T.lookup(T.hash_states(front))
            hit = td >= 0
            cand = torch.full((n * C.N_MOVES,), _BIG, dtype=torch.long, device=dev)
            cand.scatter_reduce_(0, fid[hit], k + td[hit], reduce="amin")
            newly = (child_dist < 0) & (cand < _BIG)
            child_dist[newly] = cand[newly]
            done |= (child_dist.view(n, C.N_MOVES) >= 0).any(-1)
            if done.all() or k == max_forward:
                break
            keep = ~done[fid // C.N_MOVES]                  # stop expanding labelled cubes
            front, fid = front[keep], fid[keep]
            prev = None if prev is None else prev[keep]   # k=0 -> next expansion is unpruned

        cd = child_dist.view(n, C.N_MOVES)
        big = torch.where(cd >= 0, cd, torch.full_like(cd, _BIG))
        mn = big.min(-1).values
        m = (big == mn[:, None]) & done[:, None]
        d = torch.where(done, mn + 1, torch.full_like(mn, -1))
        # the solved cube: distance 0, no optimal move
        is_solved = T.hash_states(s) == self.solved_hash
        d = torch.where(is_solved, torch.zeros_like(d), d)
        m = m & ~is_solved[:, None]
        return d.cpu().numpy().astype(np.int16), m.cpu().numpy()
