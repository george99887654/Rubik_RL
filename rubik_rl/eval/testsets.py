"""Fixed, seeded test sets, stored in testsets/*.npz and committed to git so every
experiment is measured on exactly the same cubes.

    uv run python -m rubik_rl.eval.testsets          # (re)build all sets
    uv run python -m rubik_rl.eval.testsets --list   # show what exists

Sets
    depth_1 ... depth_15  200 cubes each, scrambled with exactly d non-redundant random moves
    random_1000           1000 cubes scrambled with 100 moves (effectively random positions)
    hard                  superflip (known to need exactly 20 moves)

Each file holds
    states    (N, 54) int8     the cubes
    scramble  (N, L)  int16    moves used to make them (-1 padding)
    optimal   (N,)    int16    exact optimal HTM distance, or -1 if unknown (> 12 moves)
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..cube import cube as C

TESTSET_DIR = Path(__file__).resolve().parents[2] / "testsets"
SUPERFLIP = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"

DEPTHS = range(1, 16)
PER_DEPTH = 200
N_RANDOM = 1000
RANDOM_LEN = 100


@dataclass
class TestSet:
    name: str
    states: np.ndarray
    scramble: np.ndarray
    optimal: np.ndarray

    def __len__(self):
        return len(self.states)

    @property
    def known_optimal(self) -> np.ndarray:
        return self.optimal >= 0


def all_names() -> list[str]:
    return [f"depth_{d}" for d in DEPTHS] + ["random_1000", "hard"]


def expand_names(spec: str) -> list[str]:
    """'depth_1-5,random_1000' -> ['depth_1', ..., 'depth_5', 'random_1000']; 'all' -> every set."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part == "all":
            out += all_names()
        elif part.startswith("depth_") and "-" in part:
            a, b = part[len("depth_"):].split("-")
            out += [f"depth_{d}" for d in range(int(a), int(b) + 1)]
        elif part:
            out.append(part)
    return out


def load(name: str) -> TestSet:
    path = TESTSET_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: uv run python -m rubik_rl.eval.testsets")
    z = np.load(path)
    return TestSet(name, z["states"], z["scramble"], z["optimal"])


def _generate(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name.startswith("depth_"):
        d = int(name.split("_")[1])
        states, moves = C.scramble(PER_DEPTH, d, np.random.default_rng(1000 + d))
        return states, moves
    if name == "random_1000":
        return C.scramble(N_RANDOM, RANDOM_LEN, np.random.default_rng(2026))
    if name == "hard":
        mv = np.array([C.parse_moves(SUPERFLIP)])
        return C.apply_moves(C.solved(1), mv[0]), mv
    raise ValueError(name)


def build(names=None, solver=None, verbose=True):
    from .optimal import OptimalSolver

    solver = solver or OptimalSolver(depth=6)
    TESTSET_DIR.mkdir(exist_ok=True)
    for name in names or all_names():
        t = time.perf_counter()
        states, moves = _generate(name)
        if name == "hard":
            optimal = np.array([20], dtype=np.int16)  # superflip: proven distance 20
        elif name == "random_1000":
            # ~18 moves deep, beyond the 12-move horizon: searching would find nothing
            optimal = np.full(len(states), -1, dtype=np.int16)
        else:
            optimal = np.array([solver.distance(s) for s in states], dtype=np.int16)
        np.savez_compressed(
            TESTSET_DIR / f"{name}.npz",
            states=states.astype(np.int8), scramble=moves.astype(np.int16), optimal=optimal,
        )
        if verbose:
            describe(TestSet(name, states, moves, optimal), time.perf_counter() - t)


def describe(ts: TestSet, secs: float | None = None):
    known = ts.optimal[ts.known_optimal]
    hist = np.bincount(known, minlength=13) if len(known) else []
    msg = f"{ts.name:12s} n={len(ts):5d}  optimal known {len(known):5d}"
    if len(known):
        msg += f"  mean {known.mean():5.2f}  dist " + " ".join(
            f"{d}:{c}" for d, c in enumerate(hist) if c)
    if secs is not None:
        msg += f"  ({secs:.1f}s)"
    print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="all")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for n in expand_names(a.sets):
            describe(load(n))
    else:
        build(expand_names(a.sets))


if __name__ == "__main__":
    main()
