"""Unit tests for §8 variance-component estimators (recovery of simulation
truth is validated in validation/test_variance_components.py)."""

import pytest

from evalkit.stats import sigma2_judge, sigma2_within_item, variance_components


def test_sigma2_judge_hand_computed():
    # responses: [1,3] (SS=2, df=1), [2,2] (SS=0, df=1), [5] (excluded)
    jv = sigma2_judge([[1.0, 3.0], [2.0, 2.0], [5.0]])
    assert jv.sigma2_j == pytest.approx(1.0)
    assert jv.n_responses_used == 2
    assert jv.df == 2


def test_sigma2_judge_unavailable_without_replicates():
    assert sigma2_judge([[1.0], [2.0]]) is None  # D4: report must say so


def test_sigma2_within_item():
    # scalar/binary: 2*(sigma_g^2/r + sigma_j^2/(r*m))
    assert sigma2_within_item(r=1, m=2, sigma2_j=0.16) == pytest.approx(0.16)
    assert sigma2_within_item(r=2, m=2, sigma2_j=0.16, sigma2_g=0.08) == pytest.approx(2 * (0.04 + 0.04))
    assert sigma2_within_item(r=1, m=2, sigma2_j=0.16, pairwise=True) == pytest.approx(0.08)
    with pytest.raises(ValueError):
        sigma2_within_item(r=0, m=1, sigma2_j=0.1)


def test_variance_components_and_truncation():
    d = [0.0, 1.0, 2.0, 3.0]  # s2 = 5/3
    vc = variance_components(d, 1.0)
    assert vc.sigma2_b == pytest.approx(5 / 3 - 1.0)
    assert not vc.truncated
    assert vc.icc_judge == pytest.approx(vc.sigma2_b / (vc.sigma2_b + 1.0))

    vc2 = variance_components(d, 10.0)  # within exceeds total -> truncate at 0
    assert vc2.sigma2_b == 0.0
    assert vc2.truncated
    assert vc2.icc_judge == 0.0
