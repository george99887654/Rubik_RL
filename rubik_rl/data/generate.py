"""Generate an exactly labelled dataset of cubes within N moves of solved.

    uv run python -m rubik_rl.data.generate --n 2000000 --max-len 12 --out data/exact_L12

Each cube comes from a non-redundant random scramble whose length is drawn uniformly
from 1..max_len. Labels come only from the cube itself (ExactLabeler): its exact distance
and the set of all optimal next moves. Solved cubes (a scramble that cancelled out) are dropped.

Output: shards  <out>/shard_000.npz, ...  each holding
    states   (n, 54) int8    cube
    mask     (n, 18) bool    optimal next moves
    dist     (n,)    int8    exact distance
    slen     (n,)    int8    scramble length used (for analysis only)
Existing shards are skipped, so an interrupted run can be resumed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..cube import cube as C
from .labels import ExactLabeler

ROOT = Path(__file__).resolve().parents[2]


def make_shard(labeler: ExactLabeler, n: int, max_len: int, rng: np.random.Generator):
    lengths = rng.integers(1, max_len + 1, size=n)
    out = {k: [] for k in ("states", "mask", "dist", "slen")}
    times = {}
    for L in range(1, max_len + 1):
        k = int((lengths == L).sum())
        if k == 0:
            continue
        t = time.perf_counter()
        states, _ = C.scramble(k, L, rng)
        dist, mask = labeler.label(states, max_dist=L)
        keep = dist > 0
        out["states"].append(states[keep]); out["mask"].append(mask[keep])
        out["dist"].append(dist[keep].astype(np.int8)); out["slen"].append(np.full(keep.sum(), L, np.int8))
        times[L] = time.perf_counter() - t
    shard = {k: np.concatenate(v) for k, v in out.items()}
    perm = rng.permutation(len(shard["dist"]))
    return {k: v[perm] for k, v in shard.items()}, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--shard-size", type=int, default=250_000)
    ap.add_argument("--out", default="data/exact_L12")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    labeler = ExactLabeler()
    n_shards = (a.n + a.shard_size - 1) // a.shard_size
    for i in range(n_shards):
        path = out / f"shard_{i:03d}.npz"
        if path.exists():
            print(f"{path.name} exists, skipping", flush=True)
            continue
        t = time.perf_counter()
        rng = np.random.default_rng([a.seed, i])          # independent, reproducible per shard
        shard, times = make_shard(labeler, a.shard_size, a.max_len, rng)
        np.savez_compressed(path, **shard)
        dt = time.perf_counter() - t
        hist = np.bincount(shard["dist"], minlength=a.max_len + 1)
        print(f"{path.name}: {len(shard['dist']):,} cubes in {dt:.0f}s "
              f"(slowest: L={max(times, key=times.get)} {max(times.values()):.0f}s)  "
              f"dist hist {hist.tolist()}  mean #optimal {shard['mask'].sum(1).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
