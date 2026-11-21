import numpy as np
import pytest

from rubik_rl.cube import cube as C
from rubik_rl.eval import testsets as T
from rubik_rl.eval.evaluate import evaluate
from rubik_rl.eval.solvers import RandomSolver, TwoPhaseSolver


def test_testsets_are_reproducible_and_labelled():
    for name in ["depth_1", "depth_5", "depth_10"]:
        ts = T.load(name)
        states, moves = T._generate(name)
        assert np.array_equal(ts.states, states)
        d = int(name.split("_")[1])
        assert ts.known_optimal.all()                       # all within 12 moves
        assert (ts.optimal <= d).all() and (ts.optimal >= 0).all()
        for st, mv in zip(ts.states[:20], ts.scramble[:20]):
            assert np.array_equal(C.apply_moves(C.solved(), mv[mv >= 0]), st)
    assert (T.load("depth_1").optimal == 1).all()
    hard = T.load("hard")
    assert hard.optimal[0] == 20 and C.to_string(hard.states[0]).startswith("UBULURUFU")


def test_expand_names():
    assert T.expand_names("depth_2-4,hard") == ["depth_2", "depth_3", "depth_4", "hard"]
    assert len(T.expand_names("all")) == 17


def test_random_solver_on_depth_1():
    res = evaluate(RandomSolver(tries=500, max_moves=5), T.load("depth_1"), limit=20, progress=False)
    s = res["summary"]
    assert s["solve_rate"] == 1.0 and s["invalid"] == 0
    assert s["mean_length"] >= 1


class _BrokenSolver:
    name = "broken"

    def solve(self, state):
        return [0]  # just turns U: never a real solution for these cubes


def test_invalid_solutions_are_caught():
    res = evaluate(_BrokenSolver(), T.load("depth_3"), limit=10, progress=False)
    assert res["summary"]["solve_rate"] < 1.0
    assert res["summary"]["invalid"] >= 9


def test_twophase_solves_and_reports_gap():
    pytest.importorskip("twophase")
    res = evaluate(TwoPhaseSolver(20, 2), T.load("depth_8"), limit=5, progress=False)
    s = res["summary"]
    assert s["solve_rate"] == 1.0 and s["n_known_optimal"] == 5
    assert s["mean_gap"] >= 0
