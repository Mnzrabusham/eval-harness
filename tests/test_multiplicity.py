"""Unit tests for §6 Holm and Benjamini-Hochberg against hand-computed values."""

import numpy as np
import pytest

from evalkit.stats import benjamini_hochberg, holm


def test_holm_hand_computed():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted: [0.005, 0.01, 0.03, 0.04], multipliers [4,3,2,1]
    # -> [0.02, 0.03, 0.06, 0.04] -> cummax -> [0.02, 0.03, 0.06, 0.06]
    adj = holm(p)
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_bh_hand_computed():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted * m/rank: [0.02, 0.02, 0.04, 0.04]; reverse cummin unchanged
    adj = benjamini_hochberg(p)
    assert adj == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_holm_dominates_bonferroni_and_clips():
    p = np.array([0.4, 0.5, 0.9])
    adj = holm(p)
    assert np.all(adj <= 1.0)
    assert np.all(adj <= np.minimum(1.0, p * p.size) + 1e-15)


def test_single_test_identity():
    assert holm([0.03]) == pytest.approx([0.03])
    assert benjamini_hochberg([0.03]) == pytest.approx([0.03])


def test_monotone_in_input():
    rng = np.random.Generator(np.random.PCG64(1))
    p = rng.random(10)
    for adj in (holm(p), benjamini_hochberg(p)):
        order = np.argsort(p)
        assert np.all(np.diff(adj[order]) >= -1e-15)  # adjusted preserves ranking


def test_ties_handled():
    p = np.array([0.02, 0.02, 0.02])
    assert holm(p) == pytest.approx([0.06, 0.06, 0.06])
    assert benjamini_hochberg(p) == pytest.approx([0.02, 0.02, 0.02])


def test_invalid_pvalues_rejected():
    with pytest.raises(ValueError):
        holm([0.5, 1.2])
    with pytest.raises(ValueError):
        benjamini_hochberg([-0.1])
    with pytest.raises(ValueError):
        holm([])
