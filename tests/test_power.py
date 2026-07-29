"""Unit tests for §9 analytic power/MDE: round trips, the §9.3 worked
example, and input validation. Agreement with simulated power is validated
in validation/test_power_agreement.py."""

import pytest

from evalkit.stats import (
    design_effect,
    mde_binary_paired,
    mde_paired_mean,
    n_binary_paired,
    n_paired_mean,
    power_binary_paired,
    power_paired_mean,
)


def test_paired_mean_round_trip():
    n = n_paired_mean(0.3, 1.0)
    assert power_paired_mean(n, 0.3, 1.0) >= 0.795
    assert power_paired_mean(n - 4, 0.3, 1.0) < power_paired_mean(n, 0.3, 1.0)


def test_mde_inverts_power():
    mde = mde_paired_mean(50, 1.0)
    # MDE formula is the t-refined approximation; nct power at the MDE ~ 0.80
    assert power_paired_mean(50, mde, 1.0) == pytest.approx(0.80, abs=0.01)


def test_worked_example_9_3():
    # §9.3: 50 items, binary, psi = 0.3 -> MDE ~ 0.21 pass-rate points.
    mde = mde_binary_paired(50, 0.3)
    assert 0.19 <= mde <= 0.23
    n = n_binary_paired(0.21, 0.3)
    assert 45 <= n <= 60


def test_connor_power_monotone():
    assert power_binary_paired(100, 0.15, 0.3) > power_binary_paired(50, 0.15, 0.3)
    assert power_binary_paired(100, 0.20, 0.3) > power_binary_paired(100, 0.10, 0.3)


def test_design_effect():
    assert design_effect(5, 0.5) == pytest.approx(3.0)
    assert design_effect(1, 0.9) == pytest.approx(1.0)
    n_flat = n_paired_mean(0.3, 1.0)
    assert n_paired_mean(0.3, 1.0, deff=2.0) >= 2 * n_flat - 1


def test_validation_errors():
    with pytest.raises(ValueError):
        n_paired_mean(0.0, 1.0)
    with pytest.raises(ValueError):
        n_binary_paired(0.4, 0.3)  # |delta| > psi
    with pytest.raises(ValueError):
        mde_binary_paired(2, 0.3)  # unreachable power
    with pytest.raises(ValueError):
        design_effect(0.5, 0.2)
