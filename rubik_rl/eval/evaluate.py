"""Evaluate solvers on the fixed test sets.

    uv run python -m rubik_rl.eval.evaluate --solver twophase --sets depth_1-15,random_1000
    uv run python -m rubik_rl.eval.evaluate --solver twophase:1:5 --sets random_1000 --limit 100
    uv run python -m rubik_rl.eval.evaluate --compare          # table of all saved results

Every returned solution is replayed; a "solution" that doesn't solve the cube counts as
a failure and is reported as invalid (it means the solver has a bug).
Results go to runs/eval/<solver>__<set>.json (one file per solver/set; reruns overwrite).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from ..cube import cube as C
from . import solvers as S
from . import testsets as T

RESULTS_DIR = Path(__file__).resolve().parents[2] / "runs" / "eval"


def evaluate(solver, ts: T.TestSet, limit: int | None = None, progress: bool = True) -> dict:
    n = len(ts) if limit is None else min(limit, len(ts))
    records = []
    t_start = time.perf_counter()
    for i in range(n):
        st = ts.states[i]
        t = time.perf_counter()
        sol = solver.solve(st)
        dt = time.perf_counter() - t
        valid = sol is not None and C.is_solved(C.apply_moves(st, sol))
        records.append({
            "i": i,
            "solved": bool(valid),
            "invalid": sol is not None and not valid,
            "length": len(sol) if valid else None,
            "optimal": int(ts.optimal[i]),
            "time": dt,
            "nodes": getattr(solver, "nodes", None),
            "solution": C.format_moves(sol) if sol is not None else None,
        })
        if progress and (i + 1) % max(1, n // 10) == 0:
            print(f"  {ts.name}: {i + 1}/{n}", flush=True)
    return {
        "solver": solver.name, "set": ts.name, "n": n,
        "summary": summarize(records), "wall_time": time.perf_counter() - t_start,
        "records": records,
    }


def summarize(records: list[dict]) -> dict:
    n = len(records)
    solved = [r for r in records if r["solved"]]
    lens = np.array([r["length"] for r in solved])
    with_opt = [r for r in solved if r["optimal"] >= 0]
    gaps = np.array([r["length"] - r["optimal"] for r in with_opt])
    nodes = [r["nodes"] for r in records if r["nodes"] is not None]
    out = {
        "solve_rate": len(solved) / n if n else 0.0,
        "invalid": sum(r["invalid"] for r in records),
        "mean_length": float(lens.mean()) if len(lens) else None,
        "median_length": float(np.median(lens)) if len(lens) else None,
        "length_hist": {int(k): int(v) for k, v in zip(*np.unique(lens, return_counts=True))} if len(lens) else {},
        "n_known_optimal": len(with_opt),
        "pct_optimal": float((gaps == 0).mean()) if len(gaps) else None,
        "mean_gap": float(gaps.mean()) if len(gaps) else None,
        "mean_time": float(np.mean([r["time"] for r in records])) if n else None,
        "mean_nodes": float(np.mean(nodes)) if nodes else None,
    }
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.=-]+", "_", s).strip("_")


def save(result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{_slug(result['solver'])}__{result['set']}.json"
    path.write_text(json.dumps(result, indent=1))
    return path


def fmt_row(r: dict) -> str:
    s = r["summary"]
    f = lambda v, spec: "-" if v is None else format(v, spec)  # noqa: E731
    return (f"{r['solver']:28s} {r['set']:12s} {r['n']:5d} {s['solve_rate']:7.1%} "
            f"{f(s['mean_length'], '8.2f')} {f(s['pct_optimal'], '9.1%')} {f(s['mean_gap'], '7.2f')} "
            f"{f(s['mean_time'], '9.3f')}s {s['invalid']:4d}")


HEADER = (f"{'solver':28s} {'set':12s} {'n':>5s} {'solved':>7s} {'mean len':>8s} "
          f"{'% optimal':>9s} {'gap':>7s} {'time/cube':>10s} {'bad':>4s}")


def compare(pattern: str = "*"):
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob(f"{pattern}.json"))]

    def key(r):
        m = re.match(r"depth_(\d+)", r["set"])
        return (r["solver"], 0 if m else 1, int(m.group(1)) if m else 0, r["set"])

    print(HEADER)
    for r in sorted(rows, key=key):
        print(fmt_row(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="twophase")
    ap.add_argument("--sets", default="depth_1-15,random_1000,hard")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare()
        return
    solver = S.make(a.solver)
    print(HEADER)
    for name in T.expand_names(a.sets):
        res = evaluate(solver, T.load(name), a.limit, progress=False)
        save(res)
        print(fmt_row(res), flush=True)


if __name__ == "__main__":
    main()
