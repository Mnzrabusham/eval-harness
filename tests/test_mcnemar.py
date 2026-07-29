"""Unit tests for the exact McNemar cross-check (§5)."""

import numpy as np
import pytest
from scipy import stats as sps

from evalkit.stats import discordant_counts, mcnemar_exact


def test_known_value():
    # n_pos=5, n_neg=1: p = 2 * P(X <= 1 | n=6, 1/2) = 2 * 7/64
    assert mcnemar_exact(5, 1) == pytest.approx(14 / 64)


def test_equal_counts_p_one():
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) == 1.0


def test_symmetry():
    assert mcnemar_exact(2, 9) == mcnemar_exact(9, 2)


def test_capped_at_one():
    assert mcnemar_exact(4, 3) <= 1.0


def test_matches_scipy_binom():
    for n_pos, n_neg in [(7, 2), (1, 6), (10, 3)]:
        expected = min(1.0, 2 * sps.binom.cdf(min(n_pos, n_neg), n_pos + n_neg, 0.5))
        assert mcnemar_exact(n_pos, n_neg) == pytest.approx(expected)


def test_discordant_counts():
    d = np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0])
    assert discordant_counts(d) == (2, 1, 3)
    with pytest.raises(ValueError):
        discordant_counts([0.5, 1.0])


def test_invalid_inputs():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)
    with pytest.raises(ValueError):
        mcnemar_exact(1.5, 2)
