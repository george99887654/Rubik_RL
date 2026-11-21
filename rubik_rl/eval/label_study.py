"""How good are random-walk labels? Measured with exact optimal distances.

For random walks from solved (non-redundant moves), at each step t we compute the true
distance D(s_t) and ask:
  * scramble depth vs true distance: how often is D(s_t) < t (the walk doubled back)?
  * "undo the last move" label: it lies on a shortest path iff D(s_{t-1}) = D(s_t) - 1.
  * how many of the 18 moves are optimal from s_t (several equally good answers)?

    uv run python -m rubik_rl.eval.label_study --walks 200 --length 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..cube import cube as C
from .optimal import OptimalSolver

OUT = Path(__file__).resolve().parents[2] / "runs" / "label_study.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", type=int, default=200)
    ap.add_argument("--length", type=int, default=12)
    ap.add_argument("--branch-depth", type=int, default=9,
                    help="count optimal moves (18 extra searches each) only up to this depth")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    opt = OptimalSolver(depth=6)
    states, moves = C.random_walks(a.walks, a.length, np.random.default_rng(a.seed))
    D = np.zeros((a.walks, a.length + 1), dtype=int)  # D[:, 0] = 0 (solved)
    for t in range(1, a.length + 1):
        D[:, t] = [opt.distance(s) for s in states[:, t - 1]]
        print(f"depth {t} done", flush=True)

    rows = []
    print(f"\n{'t':>3s} {'mean D':>7s} {'D==t':>6s} {'D<t':>6s} {'undo optimal':>13s} {'# optimal moves':>16s}")
    for t in range(1, a.length + 1):
        d_t, d_prev = D[:, t], D[:, t - 1]
        known = (d_t >= 0) & (d_prev >= 0)
        undo_ok = float((d_prev[known] == d_t[known] - 1).mean()) if known.any() else None
        n_opt = None
        if t <= a.branch_depth:
            counts = []
            for s, d in zip(states[:, t - 1], d_t):
                if d <= 0:
                    continue
                kids = C.expand(s[None])[0]
                counts.append(sum(opt.distance(k) == d - 1 for k in kids))
            n_opt = float(np.mean(counts)) if counts else None
        row = {
            "t": t, "mean_D": float(d_t[d_t >= 0].mean()), "frac_D_eq_t": float((d_t == t).mean()),
            "frac_D_lt_t": float(((d_t >= 0) & (d_t < t)).mean()), "frac_unknown": float((d_t < 0).mean()),
            "undo_is_optimal": undo_ok, "mean_optimal_moves": n_opt,
        }
        rows.append(row)
        f = lambda v, s: "-" if v is None else format(v, s)  # noqa: E731
        print(f"{t:3d} {row['mean_D']:7.2f} {row['frac_D_eq_t']:6.1%} {row['frac_D_lt_t']:6.1%} "
              f"{f(undo_ok, '13.1%')} {f(n_opt, '16.2f')}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
