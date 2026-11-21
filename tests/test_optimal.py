import numpy as np
import pytest
import torch

from rubik_rl.cube import cube as C
from rubik_rl.eval.optimal import OptimalSolver

# Known number of positions at each exact HTM distance from solved.
KNOWN_COUNTS = [1, 18, 243, 3240, 43239, 574908, 7618438]


@pytest.fixture(scope="module")
def small():
    return OptimalSolver(depth=4, cache=False)  # exact up to 8 moves; builds in < 1 s


def test_small_table_counts(small):
    assert small.layer_sizes == KNOWN_COUNTS[:5]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="depth-6 table is slow on CPU")
def test_full_table_counts():
    assert OptimalSolver(depth=6).layer_sizes == KNOWN_COUNTS


def test_solutions_are_optimal_and_valid(small):
    rng = np.random.default_rng(0)
    for d in range(0, 9):
        states, _ = C.scramble(5, d, rng)
        for st in states:
            sol = small.solve(st)
            assert sol is not None and len(sol) <= d
            assert C.is_solved(C.apply_moves(st, sol))


def test_distance_changes_by_one_per_move(small):
    # neighbours differ in distance by at most 1 (and the solution path decreases by 1 each step)
    st = C.scramble(1, 6, np.random.default_rng(1))[0][0]
    d = small.distance(st)
    for m in range(18):
        assert abs(small.distance(C.apply_move(st, m)) - d) <= 1
    sol = small.solve(st)
    for i, m in enumerate(sol):
        st = C.apply_move(st, m)
        assert small.distance(st) == d - i - 1


def test_known_short_algorithms(small):
    assert small.distance(C.solved()) == 0
    assert small.distance(C.apply_moves(C.solved(), C.parse_moves("R U R' U'"))) == 4
    # superflip is 20 moves away: far beyond the 8-move horizon
    sf = C.apply_moves(C.solved(), C.parse_moves("U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"))
    assert small.solve(sf) is None
