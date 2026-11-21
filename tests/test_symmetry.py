import numpy as np
import pytest
import torch

from rubik_rl.cube import cube as C
from rubik_rl.cube import symmetry as S


def test_48_distinct_symmetries_identity_first():
    assert S.N_SYM == 48
    assert np.array_equal(S.SYM_PERM[0], np.arange(54)) and (S.SYM_MOVE[0] == np.arange(18)).all()
    assert (S.SYM_DET == 1).sum() == 24 and (S.SYM_DET == -1).sum() == 24
    st = C.scramble(1, 20, np.random.default_rng(0))[0][0]
    images = {C.to_string(S.apply_sym(st, k)) for k in range(48)}
    assert len(images) == 48  # a random cube has 48 distinct symmetric versions


def test_solved_is_fixed_and_centres_stay_standard():
    for k in range(48):
        assert C.is_solved(S.apply_sym(C.solved(), k))
        st = S.apply_sym(C.scramble(1, 15)[0][0], k)
        assert np.array_equal(st[C.CENTER_IDX], np.arange(6))
        assert C.solvability_error(C.to_string(st)) is None  # still a legal cube


def test_moves_commute_with_symmetry():
    rng = np.random.default_rng(1)
    states, _ = C.scramble(20, 12, rng)
    for k in range(48):
        for m in range(18):
            lhs = S.apply_sym(C.apply_move(states, m), k)
            rhs = C.apply_move(S.apply_sym(states, k), S.SYM_MOVE[k, m])
            assert np.array_equal(lhs, rhs)


def test_move_map_is_a_permutation_and_mirror_flips_direction():
    for k in range(48):
        assert sorted(S.SYM_MOVE[k]) == list(range(18))
        assert (S.SYM_MOVE[k, S.SYM_MOVE_INV[k]] == np.arange(18)).all()
        half = S.SYM_MOVE[k, 2::3]
        assert (half % 3 == 2).all()  # half turns stay half turns
        cw = S.SYM_MOVE[k, 0::3] % 3
        assert (cw == (0 if S.SYM_DET[k] > 0 else 1)).all()


def test_batched_versions_match_single():
    states, _ = C.scramble(64, 10, np.random.default_rng(2))
    ks = np.random.default_rng(3).integers(0, 48, 64)
    batch = S.apply_sym(states, ks)
    masks = np.random.default_rng(4).random((64, 18)) < 0.2
    mm = S.map_mask(masks, ks)
    for i in range(64):
        assert np.array_equal(batch[i], S.apply_sym(states[i], ks[i]))
        assert np.array_equal(mm[i], S.map_mask(masks[i], ks[i]))
        # mask entry for move m moves to entry SYM_MOVE[k, m]
        assert np.array_equal(mm[i][S.SYM_MOVE[ks[i]]], masks[i])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU tables")
def test_exact_labels_are_symmetric():
    from rubik_rl.data.labels import ExactLabeler

    lab = ExactLabeler()
    states, _ = C.scramble(40, 10, np.random.default_rng(5))
    d, m = lab.label(states, max_dist=10)
    for k in (1, 7, 23, 24, 30, 47):
        d2, m2 = lab.label(S.apply_sym(states, k), max_dist=10)
        assert np.array_equal(d, d2)
        assert np.array_equal(S.map_mask(m, k), m2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU tables")
def test_training_augmentation_keeps_labels_exact():
    from rubik_rl.data.labels import ExactLabeler
    from rubik_rl.train.train import SymAugment

    lab = ExactLabeler()
    states, _ = C.scramble(200, 9, np.random.default_rng(6))
    d, m = lab.label(states, max_dist=9)
    dev = torch.device("cuda")
    s_aug, m_aug = SymAugment(dev)(torch.tensor(states.astype(np.uint8), device=dev),
                                   torch.tensor(m, device=dev))
    d2, m2 = lab.label(s_aug.cpu().numpy(), max_dist=9)
    assert np.array_equal(d, d2)                     # distance unchanged
    assert np.array_equal(m_aug.cpu().numpy(), m2)   # permuted mask = true optimal moves
    assert not np.array_equal(s_aug.cpu().numpy(), states)
