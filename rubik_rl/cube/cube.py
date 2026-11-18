"""Batched 3x3 Rubik's cube on the 54-sticker representation.

State
    An int8 array of shape (54,) or (N, 54). Entry i is the colour (0..5 = U R F D L B)
    of facelet i, using the Kociemba facelet order: faces U R F D L B, 9 stickers each,
    row-major as seen looking straight at that face. This is the same layout the
    twophase solver and viz/cube_viewer.html use.

Moves (half-turn metric, 18 moves)
    Index m = 3 * face + k, where face is 0..5 = U R F D L B and k is
    0 = clockwise (R), 1 = counter-clockwise (R'), 2 = half turn (R2).
    Applying a move is a gather: new_state = state[..., PERMS[m]].

The permutations are derived from a geometric sticker model (the same model the
viewer uses), not hand-typed, so the environment and the viewer cannot disagree.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np

FACES = "URFDLB"
N_FACES = 6
N_STICKERS = 54
N_MOVES = 18
SUFFIXES = ("", "'", "2")
MOVE_NAMES: tuple[str, ...] = tuple(f + s for f in FACES for s in SUFFIXES)
MOVE_INDEX: dict[str, int] = {name: i for i, name in enumerate(MOVE_NAMES)}

SOLVED: np.ndarray = np.repeat(np.arange(N_FACES, dtype=np.int8), 9)
SOLVED.setflags(write=False)
SOLVED_STRING = "".join(f * 9 for f in FACES)

CENTER_IDX = np.array([f * 9 + 4 for f in range(N_FACES)])
NONCENTER_IDX = np.array([i for i in range(N_STICKERS) if i % 9 != 4])

# face -> (axis, sign) of its outward normal; axes x->R, y->U, z->F
_FACE_AXIS = {"U": (1, 1), "D": (1, -1), "R": (0, 1), "L": (0, -1), "F": (2, 1), "B": (2, -1)}
OPPOSITE_FACE = np.array([3, 4, 5, 0, 1, 2])  # U<->D, R<->L, F<->B


# --------------------------------------------------------------------------- geometry
def facelet_index(face: str, x: int, y: int, z: int) -> int:
    row, col = {
        "U": (z + 1, x + 1),
        "R": (1 - y, 1 - z),
        "F": (1 - y, x + 1),
        "D": (1 - z, x + 1),
        "L": (1 - y, z + 1),
        "B": (1 - y, 1 - x),
    }[face]
    return FACES.index(face) * 9 + row * 3 + col


def _rotation(axis: int, quarter_turns: int) -> np.ndarray:
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][quarter_turns % 4]
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _build_perms() -> np.ndarray:
    geo: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                p = np.array([x, y, z])
                for face, (axis, sign) in _FACE_AXIS.items():
                    if p[axis] == sign:
                        n = np.zeros(3, dtype=int)
                        n[axis] = sign
                        geo[facelet_index(face, x, y, z)] = (p, n)
    lookup = {(tuple(p), tuple(n)): i for i, (p, n) in geo.items()}

    def face_of(n):
        axis = int(np.argmax(np.abs(n)))
        return [("R", "L"), ("U", "D"), ("F", "B")][axis][0 if n[axis] > 0 else 1]

    assert all(face_of(n) == FACES[i // 9] for i, (_, n) in geo.items())

    perms = np.zeros((N_MOVES, N_STICKERS), dtype=np.int64)
    for m, name in enumerate(MOVE_NAMES):
        axis, sign = _FACE_AXIS[name[0]]
        quarter = {"": 1, "'": -1, "2": 2}[name[1:]]
        # clockwise as seen from the face = -90deg about its outward normal
        R = _rotation(axis, -sign * quarter)
        perm = np.arange(N_STICKERS)
        for i, (p, n) in geo.items():
            if p[axis] == sign:
                j = lookup[(tuple(R @ p), tuple(R @ n))]
                perm[j] = i  # sticker at i moves to j  =>  new[j] = old[i]
        perms[m] = perm
    return perms


PERMS: np.ndarray = _build_perms()
PERMS.setflags(write=False)

INVERSE_MOVE = np.array([3 * (m // 3) + (1, 0, 2)[m % 3] for m in range(N_MOVES)])
MOVE_FACE = np.arange(N_MOVES) // 3


def _build_allowed() -> np.ndarray:
    """ALLOWED[prev, m]: may move m follow move prev? Row N_MOVES means 'no previous move'.

    Removes redundant sequences: the same face twice in a row (R R' etc.), and
    opposite faces in non-canonical order (D U is the same as U D, so only U D is kept).
    """
    allowed = np.ones((N_MOVES + 1, N_MOVES), dtype=bool)
    for prev in range(N_MOVES):
        pf = prev // 3
        for m in range(N_MOVES):
            f = m // 3
            if f == pf or (f == OPPOSITE_FACE[pf] and f < pf):
                allowed[prev, m] = False
    return allowed


ALLOWED: np.ndarray = _build_allowed()
ALLOWED.setflags(write=False)
NO_PREV = N_MOVES  # row index into ALLOWED for "no previous move"

# padded lists of allowed moves per prev, for fast vectorised sampling
_ALLOWED_COUNT = ALLOWED.sum(1)
_ALLOWED_TABLE = np.zeros((N_MOVES + 1, N_MOVES), dtype=np.int64)
for _p in range(N_MOVES + 1):
    _idx = np.flatnonzero(ALLOWED[_p])
    _ALLOWED_TABLE[_p, : len(_idx)] = _idx


# --------------------------------------------------------------------------- core ops
def solved(n: int | None = None) -> np.ndarray:
    """A solved state, or a batch of n solved states."""
    return SOLVED.copy() if n is None else np.tile(SOLVED, (n, 1))


def apply_move(states: np.ndarray, moves) -> np.ndarray:
    """Apply one move per state. states: (54,) or (N,54); moves: int or (N,) ints."""
    moves = np.asarray(moves)
    if states.ndim == 1:
        return states[PERMS[moves]]
    if moves.ndim == 0:
        return states[:, PERMS[moves]]
    return np.take_along_axis(states, PERMS[moves], axis=1)


def apply_moves(states: np.ndarray, moves: Iterable) -> np.ndarray:
    """Apply a sequence of moves (indices or names) to a state or a batch of states."""
    for m in moves:
        if isinstance(m, str):
            m = MOVE_INDEX[m]
        states = apply_move(states, m)
    return states


def expand(states: np.ndarray) -> np.ndarray:
    """All 18 children of each state. (N,54) -> (N,18,54)."""
    return states[:, PERMS]


def is_solved(states: np.ndarray) -> np.ndarray | bool:
    if states.ndim == 1:
        return bool(np.array_equal(states, SOLVED))
    return np.all(states == SOLVED, axis=-1)


# --------------------------------------------------------------------------- scrambles
def random_moves(
    n: int, length: int, rng: np.random.Generator | None = None, prune: bool = True
) -> np.ndarray:
    """(n, length) random move indices. With prune=True, consecutive moves never
    turn the same face and opposite faces always appear in canonical order."""
    rng = np.random.default_rng() if rng is None else rng
    out = np.empty((n, length), dtype=np.int64)
    if not prune:
        out[:] = rng.integers(0, N_MOVES, size=(n, length))
        return out
    prev = np.full(n, NO_PREV)
    for t in range(length):
        r = (rng.random(n) * _ALLOWED_COUNT[prev]).astype(np.int64)
        prev = _ALLOWED_TABLE[prev, r]
        out[:, t] = prev
    return out


def scramble(
    n: int, length: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """n states, each scrambled from solved with `length` random (pruned) moves.
    Returns (states (n,54), moves (n,length))."""
    moves = random_moves(n, length, rng)
    states = solved(n)
    for t in range(length):
        states = apply_move(states, moves[:, t])
    return states, moves


def random_walks(
    n: int, length: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """States along n random walks from solved, for generating training data.

    Returns
        states: (n, length, 54); states[:, t] is the state after t+1 moves
        moves:  (n, length);     moves[:, t] is the move that produced states[:, t]
    The supervised label for states[:, t] is INVERSE_MOVE[moves[:, t]], and its
    scramble depth (an upper bound on its distance to solved) is t+1.
    """
    moves = random_moves(n, length, rng)
    states = np.empty((n, length, N_STICKERS), dtype=np.int8)
    cur = solved(n)
    for t in range(length):
        cur = apply_move(cur, moves[:, t])
        states[:, t] = cur
    return states, moves


# --------------------------------------------------------------------------- encoding
_EYE = np.eye(N_FACES, dtype=np.float32)


def one_hot(states: np.ndarray, drop_centers: bool = True) -> np.ndarray:
    """Network input. (N,54) -> (N, 48*6=288) float32 (centers never move, so they
    are dropped by default), or (N, 324) with drop_centers=False."""
    s = states[..., NONCENTER_IDX] if drop_centers else states
    return _EYE[s].reshape(*s.shape[:-1], -1)


# --------------------------------------------------------------------------- strings
_MOVE_RE = re.compile(r"^([URFDLB])(2'|2|'|’|1|3)?$")


def parse_moves(text: str | Sequence[str]) -> list[int]:
    """Parse "R U' F2" (also accepts twophase-style "R1 U3 F2 (3f)") into move indices."""
    tokens = text.replace("(", " ( ").split() if isinstance(text, str) else list(text)
    out = []
    for tok in tokens:
        if tok.startswith("("):
            break  # twophase appends "(Nf)"
        m = _MOVE_RE.match(tok)
        if not m:
            raise ValueError(f"unrecognised move {tok!r}")
        suf = {None: "", "1": "", "'": "'", "’": "'", "3": "'", "2": "2", "2'": "2"}[m.group(2)]
        out.append(MOVE_INDEX[m.group(1) + suf])
    return out


def format_moves(moves: Iterable[int]) -> str:
    return " ".join(MOVE_NAMES[int(m)] for m in moves)


def invert_moves(moves: Sequence[int]) -> list[int]:
    return [int(INVERSE_MOVE[m]) for m in reversed(list(moves))]


def to_string(state: np.ndarray) -> str:
    """State -> 54-char facelet string (e.g. for the twophase solver or the viewer)."""
    return "".join(FACES[c] for c in state)


def from_string(s: str, check: bool = True) -> np.ndarray:
    """54-char facelet string -> state. With check=True, raises ValueError unless the
    string describes a cube reachable from solved (uses twophase's cubie checks)."""
    s = s.strip().upper()
    if len(s) != N_STICKERS or any(c not in FACES for c in s):
        raise ValueError("state must be 54 characters from U R F D L B")
    state = np.array([FACES.index(c) for c in s], dtype=np.int8)
    if check:
        if not np.array_equal(state[CENTER_IDX], np.arange(N_FACES)):
            raise ValueError("centers must be in standard orientation (U R F D L B)")
        err = solvability_error(s)
        if err:
            raise ValueError(err)
    return state


def solvability_error(s: str) -> str | None:
    """None if the facelet string is a legal cube state, otherwise an error message."""
    from twophase import cubie, face  # light modules: no solver tables are loaded

    fc = face.FaceCube()
    r = fc.from_string(s)
    if r is not True:
        return r
    r = fc.to_cubie_cube().verify()
    return None if r is cubie.CUBE_OK else r
