import numpy as np
import pytest
import torch

from rubik_rl.cube import cube as C
from rubik_rl.cube.env import CubeEnv, net_string
from rubik_rl.cube.torch_cube import TorchCube
from rubik_rl.viz import viewer_url

from test_viewer_conventions import run as viewer_run

SUPERFLIP_MOVES = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"
SUPERFLIP_STRING = "UBULURUFURURFRBRDRFUFLFRFDFDFDLDRDBDLULBLFLDLBUBRBLBDB"


def S(seq):
    return C.apply_moves(C.solved(), C.parse_moves(seq))


# ------------------------------------------------------------------ move tables
def test_perms_are_permutations_fixing_centers():
    for m in range(C.N_MOVES):
        assert sorted(C.PERMS[m]) == list(range(54))
        assert np.array_equal(C.PERMS[m][C.CENTER_IDX], C.CENTER_IDX)
        # a face turn moves 20 stickers (8 on the face + 12 on the sides)
        assert (C.PERMS[m] != np.arange(54)).sum() == 20


def test_move_orders_and_inverses():
    for m in range(C.N_MOVES):
        order = 2 if m % 3 == 2 else 4
        assert C.is_solved(C.apply_moves(C.solved(), [m] * order))
        assert not C.is_solved(C.apply_moves(C.solved(), [m] * (order - 1)))
        assert C.is_solved(C.apply_moves(C.solved(), [m, C.INVERSE_MOVE[m]]))


def test_classic_identities():
    assert C.is_solved(S("R U R' U' " * 6))
    assert not C.is_solved(S("R U R' U' " * 3))
    assert C.is_solved(S("R U " * 105))
    assert C.is_solved(S("R2 U2 " * 6))


def test_matches_viewer_model():
    rng = np.random.default_rng(1)
    for _ in range(50):
        moves = rng.integers(0, 18, size=30)
        names = [C.MOVE_NAMES[m] for m in moves]
        assert C.to_string(C.apply_moves(C.solved(), moves)) == viewer_run(names)


def test_superflip():
    s = S(SUPERFLIP_MOVES)
    assert C.to_string(s) == SUPERFLIP_STRING
    assert C.solvability_error(SUPERFLIP_STRING) is None


def test_two_phase_solutions_solve_our_states():
    sv = pytest.importorskip("twophase.solver")
    states, _ = C.scramble(3, 30, np.random.default_rng(2))
    for st in states:
        sol = C.parse_moves(sv.solve(C.to_string(st), 20, 2))
        assert len(sol) <= 20
        assert C.is_solved(C.apply_moves(st, sol))


# ------------------------------------------------------------------ batching
def test_batched_matches_single():
    rng = np.random.default_rng(3)
    moves = rng.integers(0, 18, size=(64, 12))
    batch = C.solved(64)
    for t in range(12):
        batch = C.apply_move(batch, moves[:, t])
    for i in range(64):
        assert np.array_equal(batch[i], C.apply_moves(C.solved(), moves[i]))


def test_expand():
    states, _ = C.scramble(5, 10)
    kids = C.expand(states)
    assert kids.shape == (5, 18, 54)
    for m in range(18):
        assert np.array_equal(kids[:, m], C.apply_move(states, m))


def test_random_moves_respect_pruning():
    mv = C.random_moves(2000, 30, np.random.default_rng(4))
    assert C.ALLOWED[C.NO_PREV, mv[:, 0]].all()
    assert C.ALLOWED[mv[:, :-1], mv[:, 1:]].all()
    assert set(np.unique(mv)) == set(range(18))
    # 18 first moves; 15 after U/R/F; 12 after D/L/B
    assert C.ALLOWED.sum(1).tolist() == [15] * 9 + [12] * 9 + [18]


def test_random_walks_labels():
    states, moves = C.random_walks(16, 8, np.random.default_rng(5))
    assert states.shape == (16, 8, 54) and moves.shape == (16, 8)
    # undoing the last move of step t gives step t-1
    for t in range(1, 8):
        back = C.apply_move(states[:, t], C.INVERSE_MOVE[moves[:, t]])
        assert np.array_equal(back, states[:, t - 1])
    assert C.is_solved(C.apply_move(states[:, 0], C.INVERSE_MOVE[moves[:, 0]])).all()


def test_one_hot():
    x = C.one_hot(C.solved(3))
    assert x.shape == (3, 288) and x.dtype == np.float32
    assert np.all(x.reshape(3, 48, 6).sum(-1) == 1)
    assert C.one_hot(C.solved(2), drop_centers=False).shape == (2, 324)


# ------------------------------------------------------------------ torch
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_torch_matches_numpy(device):
    tc = TorchCube(device)
    g = torch.Generator(device=device).manual_seed(0)
    t_states, t_moves = tc.random_walks(32, 15, g)
    mv = t_moves.cpu().numpy()
    assert C.ALLOWED[mv[:, :-1], mv[:, 1:]].all()
    np_states = C.solved(32)
    for t in range(15):
        np_states = C.apply_move(np_states, mv[:, t])
        assert np.array_equal(t_states[:, t].cpu().numpy(), np_states)
    assert torch.equal(
        tc.one_hot(t_states[:, -1]).cpu(), torch.from_numpy(C.one_hot(np_states))
    )
    assert torch.equal(tc.expand(t_states[:, -1]).cpu(), torch.from_numpy(C.expand(np_states).astype(np.int64)))
    assert tc.is_solved(tc.solved(4)).all()


# ------------------------------------------------------------------ strings / validity
def test_parse_and_format():
    assert C.format_moves(C.parse_moves("R U' F2 D2' L’ B")) == "R U' F2 D2 L' B"
    assert C.format_moves(C.parse_moves("R1 U3 F2 (3f)")) == "R U' F2"
    m = C.parse_moves("R U F' D2")
    assert C.format_moves(C.invert_moves(m)) == "D2 F U' R'"
    with pytest.raises(ValueError):
        C.parse_moves("R X")


def test_string_roundtrip():
    st, _ = C.scramble(1, 25)
    s = C.to_string(st[0])
    assert np.array_equal(C.from_string(s), st[0])
    assert C.to_string(C.solved()) == C.SOLVED_STRING


def _swap(s, *pairs):
    s = list(s)
    for a, b in pairs:
        s[a], s[b] = s[b], s[a]
    return "".join(s)


def test_illegal_states_rejected():
    good = C.SOLVED_STRING
    twisted = list(good)
    twisted[8], twisted[9], twisted[20] = good[9], good[20], good[8]  # twist URF corner
    flipped = _swap(good, (7, 19))                                   # flip UF edge
    swapped = _swap(good, (7, 5), (19, 10))                          # swap UF and UR edges
    for bad in ("".join(twisted), flipped, swapped):
        assert C.solvability_error(bad) is not None
        with pytest.raises(ValueError):
            C.from_string(bad)
    with pytest.raises(ValueError):
        C.from_string("U" * 54)


# ------------------------------------------------------------------ env / viz
def test_env_episode():
    env = CubeEnv(scramble_depth=10, seed=0)
    state, info = env.reset()
    total, done = 0.0, False
    for m in C.invert_moves(env.scramble_moves):
        state, r, done, trunc, _ = env.step(m)
        total += r
    assert done and total == -10
    assert env.valid_actions().sum() in (12, 15)
    assert net_string(C.solved()).splitlines()[0] == "    UUU"


def test_env_truncates():
    env = CubeEnv(scramble_depth=20, max_steps=3, seed=1)
    env.reset()
    out = [env.step(0) for _ in range(3)]
    assert out[-1][3] or out[-1][2]


def test_viewer_url():
    url = viewer_url(C.solved(), "R U R' U'")
    assert url.startswith("file:///") and "cube_viewer.html#state=" + C.SOLVED_STRING in url
    assert url.endswith("moves=R_U_R%27_U%27")
