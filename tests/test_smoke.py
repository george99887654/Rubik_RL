import numpy as np
import torch


def test_imports():
    import rubik_rl  # noqa: F401
    import twophase.solver  # noqa: F401


def test_numpy():
    assert np.arange(4).sum() == 6


def test_torch_cuda():
    x = torch.ones(3)
    if torch.cuda.is_available():
        x = x.cuda()
    assert x.sum().item() == 3
