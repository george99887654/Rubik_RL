import numpy as np
import pytest
import torch

from rubik_rl.cube import cube as C
from rubik_rl.data.labels import DistanceTable, ExactLabeler
from rubik_rl.eval.optimal import OptimalSolver

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU tables")


@pytest.fixture(scope="module")
def labeler():
    return ExactLabeler(DistanceTable(depth=7))


@pytest.fixture(scope="module")
def slow():
    return OptimalSolver(depth=6)


def slow_label(slow, st):
    d = slow.distance(st)
    kids = [slow.distance(k) for k in C.expand(st[None])[0]]
    return d, np.array([k == d - 1 for k in kids])


def test_table_counts(labeler):
    assert labeler.table.layer_sizes == [1, 18, 243, 3240, 43239, 574908, 7618438, 100803036]


@pytest.mark.parametrize("length", [1, 3, 5, 7, 8, 9, 10, 11, 12])
def test_matches_slow_method(labeler, slow, length):
    states, _ = C.scramble(12, length, np.random.default_rng(100 + length))
    dist, mask = labeler.label(states, max_dist=length)
    for st, d, m in zip(states, dist, mask):
        sd, sm = slow_label(slow, st)
        assert d == sd
        assert np.array_equal(m, sm)
        assert m.any()  # every unsolved cube has at least one optimal move


def test_labels_without_bound_and_solved(labeler):
    states = np.stack([C.solved(), C.apply_moves(C.solved(), C.parse_moves("R U"))])
    dist, mask = labeler.label(states)
    assert dist.tolist() == [0, 2]
    assert not mask[0].any()
    assert np.flatnonzero(mask[1]).tolist() == [C.MOVE_INDEX["U'"]]


def test_following_optimal_moves_reaches_solved(labeler):
    st = C.scramble(1, 12, np.random.default_rng(5))[0][0]
    d, m = labeler.label(st[None], max_dist=12)
    d = int(d[0])
    for step in range(d):
        st = C.apply_move(st, int(np.flatnonzero(m[0])[0]))
        d2, m = labeler.label(st[None], max_dist=12)
        assert d2[0] == d - step - 1
    assert C.is_solved(st)
