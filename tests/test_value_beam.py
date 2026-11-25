import numpy as np
import pytest
import torch

from rubik_rl.cube import cube as C
from rubik_rl.models import CubeNet, CubeNetConfig
from rubik_rl.search.value_beam import ValueBeam

dev = "cuda" if torch.cuda.is_available() else "cpu"


def test_random_net_solves_all_depth_1_and_paths_are_valid():
    torch.manual_seed(0)
    net = CubeNet(CubeNetConfig(stem_dim=64, width=32, n_blocks=1)).to(dev)
    vb = ValueBeam(net, width=5)
    states = np.stack([C.apply_move(C.solved(), m) for m in range(18)] + [C.solved()])
    paths = vb.solve_batch(states, max_steps=1)
    assert paths[-1] == []
    for st, p in zip(states[:-1], paths[:-1]):
        assert p is not None and len(p) == 1 and C.is_solved(C.apply_moves(st, p))


def test_paths_valid_and_bounded_on_mixed_batch():
    torch.manual_seed(1)
    net = CubeNet(CubeNetConfig(stem_dim=64, width=32, n_blocks=1)).to(dev)
    vb = ValueBeam(net, width=5)
    states, _ = C.scramble(300, 3, np.random.default_rng(0))
    paths = vb.solve_batch(states, max_steps=3)
    n_ok = 0
    for st, p in zip(states, paths):
        if p is not None:
            assert len(p) <= 3 and C.is_solved(C.apply_moves(st, p))
            n_ok += 1
    assert n_ok > 0


def test_width_one_with_perfect_value_is_optimal():
    """With an oracle value (exact distance) a width-1 beam follows an optimal path."""
    from rubik_rl.eval.optimal import OptimalSolver

    opt = OptimalSolver(depth=4, cache=False)

    class Oracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1, device=dev))
            self.cur = None

    vb = ValueBeam(Oracle(), width=1)

    def oracle_values(states):
        s = states.cpu().numpy()
        return torch.tensor([float(opt.distance(x)) for x in s], device=dev)

    vb.values = oracle_values
    states, _ = C.scramble(10, 6, np.random.default_rng(2))
    for st, p in zip(states, vb.solve_batch(states, max_steps=6)):
        assert p is not None and len(p) == opt.distance(st)


def test_length_schedule_halves_down_from_n():
    from rubik_rl.train.self_train import length_schedule

    c = np.bincount(length_schedule(10, 4096), minlength=11)
    assert c[10] == 2048 and c[9] == 1024 and c[8] == 512 and c[2] == 8 and c[1] == 8
    assert c.sum() == 4096
    assert (length_schedule(1, 4096) == 1).all()
    for n in range(1, 16):
        s = length_schedule(n, 4096)
        assert len(s) == 4096 and s.min() >= 1 and s.max() == n


def test_value_group_scores():
    from rubik_rl.train.self_train import value_group_scores

    class FixedV:
        def values(self, s):
            return torch.full((len(s),), 4.0)

    states = np.zeros((6, 54), dtype=np.int8)
    paths = [[0] * 5, [0] * 4, [0] * 3, [0] * 2, None, [0] * 5]
    solved = np.array([p is not None for p in paths])
    lengths = np.array([5, 4, 3, 2, 5, 5])
    sc = value_group_scores(FixedV(), states, paths, solved, lengths, 5, "cpu")
    # group n=5: L=5,5 -> 1-1/5 = 0.8; n-1: L=4 -> 1.0; n-2: L=3 -> 1-1/3; <=n-3: L=2 -> 0.0
    assert np.allclose(sc, [0.8, 1.0, 2 / 3, 0.0])
    sc1 = value_group_scores(FixedV(), states[:1], [[0]], np.array([True]), np.array([1]), 1, "cpu")
    assert sc1[1:] == [None, None, None] and np.isclose(sc1[0], 1 - 3.0)


def test_value_scores_by_remaining():
    from rubik_rl.train.self_train import value_scores_by_remaining

    class FixedV:
        def values(self, s):
            return torch.full((len(s),), 4.0)

    S = np.zeros((5, 54), dtype=np.int8)
    R = np.array([5, 4, 3, 2, 1])
    sc = value_scores_by_remaining(FixedV(), (S, np.zeros(5, int), R), 5, "cpu", np.random.default_rng(0))
    # r=5: 0.8; r=4: 1.0; r=3: 1-1/3; r<=2: mean(1-2/2, 1-3/1) = mean(0, -2) = -1
    assert np.allclose(sc, [0.8, 1.0, 2 / 3, -1.0])


def test_noisy_search_paths_still_valid_and_first_step_unpruned():
    torch.manual_seed(3)
    net = CubeNet(CubeNetConfig(stem_dim=64, width=32, n_blocks=1)).to(dev)
    g = torch.Generator(device=dev).manual_seed(0)
    vb = ValueBeam(net, width=32, noise_abs=0.3, generator=g)
    states, _ = C.scramble(200, 2, np.random.default_rng(4))
    paths = vb.solve_batch(states, max_steps=2)
    # width 32 >= 18 children: step 1 keeps every child, so every 2-move cube is solved
    for st, p in zip(states, paths):
        assert p is not None and C.is_solved(C.apply_moves(st, p))


def test_halving_solve_rule():
    from rubik_rl.train.self_train import check_lower_depths, solve_need

    assert [solve_need(k) for k in range(4)] == [0.5, 0.75, 0.875, 0.9375]
    torch.manual_seed(5)
    net = CubeNet(CubeNetConfig(stem_dim=64, width=32, n_blocks=1)).to(dev)
    rates = check_lower_depths(ValueBeam(net, width=32), 3, 64, np.random.default_rng(0))
    # depths 2 and 1: width 32 >= 18 keeps every child at step 1, so both are always solved
    assert rates == [1.0, 1.0]
