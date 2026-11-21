import numpy as np
import torch
import torch.nn.functional as F

from rubik_rl.cube import cube as C
from rubik_rl.cube.torch_cube import TorchCube
from rubik_rl.models import CubeNet, CubeNetConfig, masked_log_policy

SMALL = CubeNetConfig(stem_dim=256, width=128, n_blocks=2)


def test_shapes_and_default_size():
    net = CubeNet(SMALL).eval()
    x = torch.from_numpy(C.one_hot(C.scramble(7, 10)[0]))
    logits, value = net(x)
    assert logits.shape == (7, 18) and value.shape == (7,)
    assert 10e6 < CubeNet().n_params() < 20e6  # default ~14M parameters


def test_no_value_head():
    net = CubeNet(CubeNetConfig(stem_dim=64, width=32, n_blocks=1, value_head=False)).eval()
    logits, value = net(torch.zeros(3, 288))
    assert logits.shape == (3, 18) and value is None


def test_masked_policy_blocks_redundant_moves():
    logits = torch.zeros(3, 18)
    prev = torch.tensor([C.MOVE_INDEX["R"], C.MOVE_INDEX["D"], C.NO_PREV])
    p = masked_log_policy(logits, prev).exp()
    assert torch.allclose(p.sum(-1), torch.ones(3))
    assert p[0, 3:6].sum() == 0                     # no R/R'/R2 after R
    assert p[1, 9:12].sum() == 0 and p[1, 0:3].sum() == 0  # no D after D, no U after D
    assert (p[2] > 0).all()                         # no previous move: all 18 allowed


def test_checkpoint_roundtrip():
    net = CubeNet(SMALL).eval()
    x = torch.randn(4, 288)
    net2 = CubeNet.from_checkpoint(net.checkpoint()).eval()
    for a, b in zip(net(x), net2(x)):
        assert torch.allclose(a, b)


def test_learns_one_move_scrambles():
    """Sanity check: both heads can fit depth-1/2 data (policy = undo the last move,
    value = depth)."""
    torch.manual_seed(0)
    tc = TorchCube("cpu")
    net = CubeNet(SMALL)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(80):
        states, moves = tc.random_walks(128, 2)
        x = tc.one_hot(states.reshape(-1, 54))
        y = tc.inverse[moves.reshape(-1)]
        depth = torch.arange(1, 3).repeat(128).float()
        logits, value = net(x)
        loss = F.cross_entropy(logits, y) + F.mse_loss(value, depth)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    states = torch.from_numpy(np.stack([C.apply_move(C.solved(), m) for m in range(18)])).long()
    logits, value = net(tc.one_hot(states))
    assert (logits.argmax(-1) == tc.inverse).all()
    assert (value - 1).abs().max() < 0.3


def test_runs_on_gpu():
    if not torch.cuda.is_available():
        return
    net = CubeNet().cuda().eval()
    with torch.no_grad():
        logits, value = net(torch.zeros(1024, 288, device="cuda"))
    assert logits.shape == (1024, 18) and value.shape == (1024,)
