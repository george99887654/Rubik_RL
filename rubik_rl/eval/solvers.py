"""Solver interface and baseline solvers.

A solver is any object with
    name: str
    solve(state: np.ndarray) -> list[int] | None      # move indices, or None if it gave up
and, optionally, an attribute `nodes` (states examined during the last solve).
Future network-based solvers plug into the same interface.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np

from ..cube import cube as C

ROOT = Path(__file__).resolve().parents[2]


class Solver(Protocol):
    name: str

    def solve(self, state: np.ndarray) -> list[int] | None: ...


_twophase = None


def _load_twophase():
    """Import twophase.solver from the repo root, where its ~70 MB tables are cached
    (it reads and writes them relative to the current directory)."""
    global _twophase
    if _twophase is None:
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            import twophase.solver as sv
        finally:
            os.chdir(cwd)
        _twophase = sv
    return _twophase


class TwoPhaseSolver:
    """Kociemba's two-phase algorithm (RubikTwoPhase package).

    target: return as soon as a solution of at most this length is found
    timeout: otherwise keep searching for up to this many seconds and return the shortest
    Use target=20 for a fast baseline, target=1 with a timeout for "best within T seconds".
    """

    def __init__(self, target: int = 20, timeout: float = 3.0):
        self.target, self.timeout = target, timeout
        self.name = f"twophase(target={target},t={timeout:g}s)"
        self.sv = _load_twophase()
        self.nodes = None

    def solve(self, state):
        out = self.sv.solve(C.to_string(state), self.target, self.timeout)
        if out.startswith("Error"):
            raise ValueError(out)
        return C.parse_moves(out)


class OptimalBFSSolver:
    """Exact optimal solutions for cubes up to 12 moves away (gives up beyond that)."""

    def __init__(self, depth: int = 6):
        from .optimal import OptimalSolver

        self.inner = OptimalSolver(depth=depth)
        self.name = f"optimal(<= {2 * depth})"

    @property
    def nodes(self):
        return self.inner.nodes

    def solve(self, state):
        return self.inner.solve(state)


class RandomSolver:
    """Makes random non-redundant moves until solved or max_moves is reached.
    Expected to solve almost nothing beyond depth 2-3: a floor, and a check that the
    harness itself works."""

    def __init__(self, max_moves: int = 30, tries: int = 1000, seed: int = 0):
        self.max_moves, self.tries = max_moves, tries
        self.rng = np.random.default_rng(seed)
        self.name = f"random(tries={tries},len={max_moves})"
        self.nodes = 0

    def solve(self, state):
        moves = C.random_moves(self.tries, self.max_moves, self.rng)
        cur = np.tile(state, (self.tries, 1))
        self.nodes = 0
        for t in range(self.max_moves):
            cur = C.apply_move(cur, moves[:, t])
            self.nodes += self.tries
            hit = np.flatnonzero(C.is_solved(cur))
            if len(hit):
                return moves[hit[0], : t + 1].tolist()
        return None


def make(spec: str) -> Solver:
    """Build a solver from a short spec, e.g. 'twophase', 'twophase:1:5', 'optimal', 'random',
    'beam:<width>:<checkpoint path>' (width 1 = greedy)."""
    name, *args = spec.split(":")
    if name == "beam":
        from ..search.beam import BeamSearchSolver, load_net

        width, ckpt = int(args[0]), ":".join(args[1:])
        label = Path(ckpt).parent.name or Path(ckpt).stem
        return BeamSearchSolver(load_net(ckpt), width, name=f"beam(w={width},{label})")
    if name == "twophase":
        target = int(args[0]) if args else 20
        timeout = float(args[1]) if len(args) > 1 else 3.0
        return TwoPhaseSolver(target, timeout)
    if name == "optimal":
        return OptimalBFSSolver()
    if name == "random":
        return RandomSolver(tries=int(args[0]) if args else 1000)
    raise ValueError(f"unknown solver {spec!r}")
