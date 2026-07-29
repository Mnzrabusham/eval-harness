"""Unit tests for the §2 engine: exact identities, determinism, API contract.

Correctness of coverage/level/power is established in validation/, not here
(CLAUDE.md rule 1); these tests pin exact algebraic identities and plumbing.
"""

import numpy as np
import pytest
from scipy import stats as sps

from evalkit.stats import mcnemar_exact, run_engine


def test_singleton_clusters_reduce_to_paired_t():
    # §2.3: with all-singleton clusters the cluster-robust SE is exactly the
    # classical paired-t SE — required unit test.
    rng = np.random.Generator(np.random.PCG64(7))
    d = rng.normal(0.2, 1.0, 25)
    res = run_engine(d, seed=1, n_boot=100, n_perm=100)
    se_expected = np.std(d, ddof=1) / np.sqrt(d.size)
    assert res.analytic.se == pytest.approx(se_expected, rel=1e-12)
    t_expected, p_expected = sps.ttest_rel(d, np.zeros_like(d))
    assert res.analytic.t_stat == pytest.approx(t_expected, rel=1e-10)
    assert res.analytic.p_value == pytest.approx(p_expected, rel=1e-10)
    assert res.analytic.df == d.size - 1


@pytest.mark.parametrize("n_pos,n_neg,n_zero", [
    (3, 1, 2), (5, 0, 3), (2, 2, 4), (1, 0, 0), (4, 2, 7), (0, 0, 5), (6, 1, 6),
])
def test_exact_signflip_equals_exact_mcnemar(n_pos, n_neg, n_zero):
    # §5 / §11.4: sign-flip on binary singletons reproduces McNemar's exact
    # conditional-binomial distribution; with full enumeration (C <= 13) the
    # p-values are identical, not just close.
    d = np.concatenate([np.ones(n_pos), -np.ones(n_neg), np.zeros(n_zero)])
    res = run_engine(d, seed=3, n_boot=10, n_perm=10)
    assert res.test_method == "sign-flip-exact"
    assert res.p_value == pytest.approx(mcnemar_exact(n_pos, n_neg), abs=1e-12)


def test_enumeration_threshold_boundary():
    rng = np.random.Generator(np.random.PCG64(11))
    assert run_engine(rng.normal(0, 1, 13), seed=1, n_boot=10, n_perm=50).test_method == "sign-flip-exact"
    assert run_engine(rng.normal(0, 1, 14), seed=1, n_boot=10, n_perm=50).test_method == "sign-flip-mc"


def test_reproducibility_same_seed():
    rng = np.random.Generator(np.random.PCG64(5))
    d = rng.normal(0.1, 1.0, 40)
    a = run_engine(d, seed=42, n_boot=500, n_perm=500)
    b = run_engine(d, seed=42, n_boot=500, n_perm=500)
    assert (a.ci_low, a.ci_high, a.p_value, a.estimate) == (b.ci_low, b.ci_high, b.p_value, b.estimate)


def test_input_order_invariance():
    # §0: pinned lexicographic sort means shuffled input gives identical output.
    rng = np.random.Generator(np.random.PCG64(9))
    d = rng.normal(0.1, 1.0, 30)
    items = [f"it{i:03d}" for i in range(30)]
    clusters = [f"c{i % 6}" for i in range(30)]
    perm = rng.permutation(30)
    a = run_engine(d, clusters, items, seed=17, n_boot=400, n_perm=400)
    b = run_engine(d[perm], [clusters[i] for i in perm], [items[i] for i in perm],
                   seed=17, n_boot=400, n_perm=400)
    assert (a.ci_low, a.ci_high, a.p_value) == (b.ci_low, b.ci_high, b.p_value)


def test_small_c_warning_and_degenerate():
    res = run_engine(np.ones(5), seed=1, n_boot=10, n_perm=10)
    assert res.degenerate
    assert any("C=5" in w for w in res.warnings)
    assert res.ci_low == res.ci_high == 1.0


def test_clustered_grouping():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    res = run_engine(d, clusters=["a", "a", "b", "b"], seed=1, n_boot=50, n_perm=10)
    assert res.n_clusters == 2
    assert res.cluster_sizes == {2: 2}
    assert res.estimate == pytest.approx(2.5)


def test_one_sided_alternatives():
    rng = np.random.Generator(np.random.PCG64(21))
    d = rng.normal(0.5, 1.0, 12)  # exact enumeration path
    two = run_engine(d, seed=2, n_boot=10, n_perm=10)
    hi = run_engine(d, seed=2, n_boot=10, n_perm=10, alternative="greater")
    lo = run_engine(d, seed=2, n_boot=10, n_perm=10, alternative="less")
    # Exact enumeration over a symmetric group: one-sided halves sum to 1 + P(T*=T).
    assert hi.p_value < two.p_value
    assert hi.p_value + lo.p_value >= 1.0  # both count the identity assignment
    with pytest.raises(ValueError):
        run_engine(d, seed=2, alternative="upper")


def test_validation_errors():
    with pytest.raises(ValueError):
        run_engine([], seed=1)
    with pytest.raises(ValueError):
        run_engine([1.0, np.nan], seed=1)
    with pytest.raises(ValueError):
        run_engine([1.0, 2.0], clusters=["a"], seed=1)
