"""Gym-style single-cube environment (no gym dependency).

Most training code will use the batched functions in cube.py / torch_cube.py directly;
this wrapper is for step-by-step RL agents, debugging, and demos.

Reward: -1 per move (so return = -solution length), episode ends when solved
(terminated) or after max_steps (truncated).
"""
from __future__ import annotations

import numpy as np

from . import cube as C


class CubeEnv:
    n_actions = C.N_MOVES

    def __init__(self, scramble_depth: int = 20, max_steps: int = 100, seed: int | None = None):
        self.scramble_depth = scramble_depth
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.state = C.solved()
        self.steps = 0
        self.history: list[int] = []
        self.scramble_moves: list[int] = []

    def reset(self, scramble_depth: int | None = None, state: np.ndarray | None = None):
        if state is not None:
            self.state = np.array(state, dtype=np.int8)
            self.scramble_moves = []
        else:
            depth = self.scramble_depth if scramble_depth is None else scramble_depth
            moves = C.random_moves(1, depth, self.rng)[0]
            self.state = C.apply_moves(C.solved(), moves)
            self.scramble_moves = moves.tolist()
        self.steps = 0
        self.history = []
        return self.state.copy(), {"scramble": C.format_moves(self.scramble_moves)}

    def step(self, action: int):
        self.state = C.apply_move(self.state, int(action))
        self.steps += 1
        self.history.append(int(action))
        terminated = C.is_solved(self.state)
        truncated = (not terminated) and self.steps >= self.max_steps
        return self.state.copy(), -1.0, terminated, truncated, {}

    def valid_actions(self) -> np.ndarray:
        """Boolean mask of non-redundant next moves (see cube.ALLOWED)."""
        prev = self.history[-1] if self.history else C.NO_PREV
        return C.ALLOWED[prev].copy()

    def render(self) -> str:
        return net_string(self.state)


def net_string(state: np.ndarray) -> str:
    """ASCII net of a state:
            U
          L F R B
            D
    """
    s = C.to_string(state)
    face = {f: s[i * 9 : i * 9 + 9] for i, f in enumerate(C.FACES)}
    rows = lambda f: [face[f][r * 3 : r * 3 + 3] for r in range(3)]  # noqa: E731
    pad = " " * 4
    lines = [pad + r for r in rows("U")]
    for r in range(3):
        lines.append(" ".join(rows(f)[r] for f in "LFRB"))
    lines += [pad + r for r in rows("D")]
    return "\n".join(lines)
