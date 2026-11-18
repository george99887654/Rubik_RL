"""Checks that the conventions used by viz/cube_viewer.html match Kociemba's facelet format.

This mirrors the viewer's JS math exactly: stickers are (cubie position, outward normal)
pairs; a move rotates every cubie in the layer by -90deg * sign about the face axis
(clockwise as seen from that face), and stickers map to facelet indices with
`facelet_index`. If the viewer's mapping or turn directions were wrong, applying the
two-phase solver's solution would not return the cube to solved.
"""
import random

import numpy as np
import pytest

FACES = "URFDLB"
FACE_AXIS = {"U": (1, 1), "D": (1, -1), "R": (0, 1), "L": (0, -1), "F": (2, 1), "B": (2, -1)}
SOLVED = "".join(f * 9 for f in FACES)


def facelet_index(face, x, y, z):
    row, col = {
        "U": (z + 1, x + 1),
        "R": (1 - y, 1 - z),
        "F": (1 - y, x + 1),
        "D": (1 - z, x + 1),
        "L": (1 - y, z + 1),
        "B": (1 - y, 1 - x),
    }[face]
    return FACES.index(face) * 9 + row * 3 + col


def face_from_normal(n):
    axis = int(np.argmax(np.abs(n)))
    return [("R", "L"), ("U", "D"), ("F", "B")][axis][0 if n[axis] > 0 else 1]


def rot(axis, angle):
    c, s = round(np.cos(angle)), round(np.sin(angle))
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_stickers(state=SOLVED):
    out = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                p = np.array([x, y, z])
                for face, (axis, sign) in FACE_AXIS.items():
                    if p[axis] == sign:
                        n = np.zeros(3, int)
                        n[axis] = sign
                        out.append([p.copy(), n, state[facelet_index(face, x, y, z)]])
    return out


def apply_move(stickers, move):
    axis, sign = FACE_AXIS[move[0]]
    quarter = 2 if move.endswith("2") else (-1 if move.endswith("'") else 1)
    R = rot(axis, -sign * quarter * np.pi / 2)
    for st in stickers:
        if st[0][axis] == sign:
            st[0] = R @ st[0]
            st[1] = R @ st[1]


def read_state(stickers):
    out = [""] * 54
    for p, n, color in stickers:
        out[facelet_index(face_from_normal(n), *p)] = color
    return "".join(out)


def run(moves, state=SOLVED):
    st = make_stickers(state)
    for m in moves:
        apply_move(st, m)
    return read_state(st)


def parse(sol):
    """twophase output like 'R1 U3 F2 (3f)' -> ['R', "U'", 'F2']"""
    out = []
    for tok in sol.split():
        if tok.startswith("("):
            break
        face, n = tok[0], int(tok[1])
        out.append(face + {1: "", 2: "2", 3: "'"}[n])
    return out


def test_basic_group_identities():
    for f in FACES:
        assert run([f] * 4) == SOLVED
        assert run([f, f + "'"]) == SOLVED
        assert run([f + "2", f + "2"]) == SOLVED
    assert run(["R", "U", "R'", "U'"] * 6) == SOLVED
    assert run(["R", "U"] * 105) == SOLVED


def test_known_facelets_after_U():
    # After U (clockwise from above), the F face's top row shows what was R's top row.
    s = run(["U"])
    assert s[18:21] == "RRR"  # F top row
    assert s[9:12] == "BBB"   # R top row
    assert s[0:9] == "U" * 9


def test_known_facelets_after_R():
    # After R, the U face's right column shows what was F's right column.
    s = run(["R"])
    assert s[2] + s[5] + s[8] == "FFF"


def test_matches_two_phase_solver():
    sv = pytest.importorskip("twophase.solver")
    rng = random.Random(0)
    for _ in range(5):
        scramble = [rng.choice(FACES) + rng.choice(["", "'", "2"]) for _ in range(25)]
        state = run(scramble)
        sol = parse(sv.solve(state, 20, 2))
        assert run(sol, state) == SOLVED, (scramble, sol)
