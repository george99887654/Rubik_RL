"""The 48 symmetries of the cube (24 rotations x optional mirror) acting on states and moves.

A symmetry g is a 3x3 signed permutation matrix (all 48 of them; det +1 = rotation,
det -1 = rotation + mirror). Applying g to a cube state s:
  1. move every sticker: the sticker at (position p, normal n) goes to (g p, g n)
  2. rename colours so the centres are standard again: colour c becomes the face that
     c's centre was carried to
The result s' = g(s) is a different cube with exactly the same distance to solved, and
g(solved) = solved.

Moves map too: face f goes to face g(f); a mirror reverses the turning direction
(R <-> R', R2 stays R2). For every state s and move m:  g(s after m) = g(s) after g(m).
So an optimal move m for s becomes the optimal move g(m) for g(s).

Arrays (numpy, built at import):
  SYM_PERM   (48, 54)  new_state[j] = SYM_COLOR[k][ state[SYM_PERM[k, j]] ]
  SYM_COLOR  (48, 6)   colour renaming
  SYM_MOVE   (48, 18)  move m -> SYM_MOVE[k, m]
  SYM_MOVE_INV (48,18) inverse map, for permuting move masks/policies:
                       new_mask[:, j] = mask[:, SYM_MOVE_INV[k, j]]
Symmetry 0 is the identity.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import cube as C

_FACE_NORMAL = {"U": (0, 1, 0), "D": (0, -1, 0), "R": (1, 0, 0), "L": (-1, 0, 0),
                "F": (0, 0, 1), "B": (0, 0, -1)}


def _face_of(n) -> str:
    for f, v in _FACE_NORMAL.items():
        if tuple(int(x) for x in n) == v:
            return f
    raise ValueError(n)


def _matrices() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), dtype=int)
            for row, (col, sg) in enumerate(zip(perm, signs)):
                M[row, col] = sg
            mats.append(M)
    mats.sort(key=lambda M: (not np.array_equal(M, np.eye(3, dtype=int)), round(np.linalg.det(M)) < 0))
    return mats  # identity first, then rotations, then mirrors


def _build():
    geo = {}
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                p = np.array([x, y, z])
                for f, n in _FACE_NORMAL.items():
                    n = np.array(n)
                    if (p * n).sum() == 1:  # sticker on face f
                        geo[C.facelet_index(f, x, y, z)] = (p, n)
    lookup = {(tuple(p), tuple(n)): i for i, (p, n) in geo.items()}

    perms, colors, moves, dets = [], [], [], []
    for M in _matrices():
        perm = np.zeros(C.N_STICKERS, dtype=np.int64)
        for i, (p, n) in geo.items():
            perm[lookup[(tuple(M @ p), tuple(M @ n))]] = i
        color = np.array([C.FACES.index(_face_of(M @ np.array(_FACE_NORMAL[f]))) for f in C.FACES])
        det = round(np.linalg.det(M))
        mv = np.zeros(C.N_MOVES, dtype=np.int64)
        for m in range(C.N_MOVES):
            f, k = C.FACES[m // 3], m % 3
            if det < 0 and k < 2:
                k = 1 - k  # mirror: clockwise <-> counter-clockwise
            mv[m] = 3 * color[C.FACES.index(f)] + k
        perms.append(perm); colors.append(color); moves.append(mv); dets.append(det)
    moves = np.array(moves)
    inv = np.argsort(moves, axis=1)
    return np.array(perms), np.array(colors, dtype=np.int8), moves, inv, np.array(dets)


SYM_PERM, SYM_COLOR, SYM_MOVE, SYM_MOVE_INV, SYM_DET = _build()
N_SYM = len(SYM_PERM)
for _a in (SYM_PERM, SYM_COLOR, SYM_MOVE, SYM_MOVE_INV, SYM_DET):
    _a.setflags(write=False)


def apply_sym(states: np.ndarray, k) -> np.ndarray:
    """Symmetry k (int, or one per state for a batch) applied to state(s)."""
    k = np.asarray(k)
    if states.ndim == 1:
        return SYM_COLOR[k][states[SYM_PERM[k]]]
    if k.ndim == 0:
        return SYM_COLOR[k][states[:, SYM_PERM[k]]]
    moved = np.take_along_axis(states, SYM_PERM[k], axis=1)
    return np.take_along_axis(SYM_COLOR[k], moved.astype(np.int64), axis=1)


def map_moves(moves, k):
    """Move index/indices under symmetry k."""
    return SYM_MOVE[k, moves]


def map_mask(mask: np.ndarray, k) -> np.ndarray:
    """Per-move array (N,18) (e.g. optimal-move mask or policy) under symmetry k."""
    k = np.asarray(k)
    if k.ndim == 0:
        return mask[..., SYM_MOVE_INV[k]]
    return np.take_along_axis(mask, SYM_MOVE_INV[k], axis=1)
