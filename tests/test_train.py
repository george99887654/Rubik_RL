import numpy as np
import pytest
import torch

from rubik_rl.data import generate as G
from rubik_rl.data.labels import ExactLabeler
from rubik_rl.models import CubeNetConfig
from rubik_rl.train import train as T

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU tables")


def test_shard_and_short_training(tmp_path, monkeypatch):
    shard, _ = G.make_shard(ExactLabeler(), 20_000, 4, np.random.default_rng(0))
    assert (shard["dist"] >= 1).all() and (shard["dist"] <= 4).all()
    assert shard["mask"].any(1).all()
    np.savez_compressed(tmp_path / "shard_000.npz", **shard)

    monkeypatch.setattr(T, "ROOT", tmp_path)  # keep runs/ and checkpoints/ inside tmp
    cfg = T.TrainConfig(data=str(tmp_path), name="t", steps=300, batch=512, warmup=20,
                        val_size=2000, log_every=100, val_every=300, lr=2e-3,
                        model=CubeNetConfig(stem_dim=256, width=128, n_blocks=2))
    T.train(cfg)
    log = (tmp_path / "runs" / "train" / "t" / "log.jsonl").read_text().splitlines()
    import json
    last = json.loads(log[-1])
    assert last["val"]["acc"] > 0.9         # depth <= 4 is easy: should be nearly perfect
    assert last["val"]["mae"] < 0.5
    assert (tmp_path / "checkpoints" / "t" / "best.pt").exists()


def test_beam_search_with_tiny_trained_net(tmp_path, monkeypatch):
    """Train a tiny net on depth<=4 data, then check beam search solves depth-3 cubes
    and that every returned solution is valid."""
    from rubik_rl.cube import cube as C
    from rubik_rl.search.beam import BeamSearchSolver

    shard, _ = G.make_shard(ExactLabeler(), 20_000, 4, np.random.default_rng(1))
    np.savez_compressed(tmp_path / "shard_000.npz", **shard)
    monkeypatch.setattr(T, "ROOT", tmp_path)
    net = T.train(T.TrainConfig(data=str(tmp_path), name="b", steps=300, batch=512, warmup=20,
                                val_size=2000, log_every=300, val_every=300, lr=2e-3,
                                model=CubeNetConfig(stem_dim=256, width=128, n_blocks=2)))
    greedy, beam = BeamSearchSolver(net, 1, max_steps=10), BeamSearchSolver(net, 64, max_steps=10)
    states, _ = C.scramble(30, 3, np.random.default_rng(2))
    n_beam = 0
    for st in states:
        for solver in (greedy, beam):
            sol = solver.solve(st)
            if sol is not None:
                assert C.is_solved(C.apply_moves(st, sol))
        n_beam += beam.solve(st) is not None
    assert n_beam >= 28
    assert beam.solve(C.solved()) == []
