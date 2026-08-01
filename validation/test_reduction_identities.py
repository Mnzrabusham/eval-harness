"""§11.4 — Stochastic identity studies.

1. McNemar identity on the Monte Carlo path: for binary singletons with
   C > enumerate_threshold, the sign-flip MC p-value must agree with the
   exact McNemar p-value within Monte Carlo error on every dataset (the
   exact-enumeration identity is pinned bit-for-bit in
   tests/test_engine.py).
2. Counterbalanced reduction removes an injected ADDITIVE position effect
   (bias < MC error) even with unequal call counts per order, while naive
   call-level pooling is demonstrably biased — proving the study has teeth.
"""

import math

import numpy as np
import pytest

from evalkit.stats import discordant_counts, mcnemar_exact, reduce_pairwise, run_engine
from evalkit.stats.simulate import simulate_d_binary, simulate_pairwise_records
from validation._tolerances import mc_se, publish, rep_seeds

pytestmark = pytest.mark.validation

N_PERM = 4999
PSI = 0.4
N_ITEMS = 60
# Minimum Var(B) = n_perm*p_exact*(1-p_exact) for B ~ Binomial(n_perm, p_exact)
# to be treated as a non-degenerate MC estimator (rule-of-thumb analogous to
# the np >= 5 normal-approximation guideline). p_exact == 1.0 exactly (Var=0)
# whenever the discordant split ties (n_pos == n_neg), and occasionally for
# small, asymmetric discordant counts too (e.g. n_pos=1, n_neg=0 -> p=1.0) --
# both give se_mc = 0 and must be excluded from the standardized sample
# rather than papered over with a variance floor.
MIN_VAR = 5


def _theoretical_excluded_fraction(p_pos, p_neg, n_items, n_perm, min_var):
    """Exact probability, under the simulator's multinomial generative model
    for (n_pos, n_neg, n_zero), that a dataset falls in the MIN_VAR exclusion.
    Enumerates every (n_pos, n_neg) pair (n_items=60 -> ~1900 pairs) rather
    than approximating, so it is an exact prediction to check the measured
    exclusion rate against -- itself a check that the simulator's discordant
    counts land where the stated generative model says they should."""
    p_zero = 1.0 - p_pos - p_neg
    total = 0.0
    for n_pos in range(n_items + 1):
        for n_neg in range(n_items + 1 - n_pos):
            n_zero = n_items - n_pos - n_neg
            coef = math.comb(n_items, n_pos) * math.comb(n_items - n_pos, n_neg)
            prob = coef * p_pos ** n_pos * p_neg ** n_neg * p_zero ** n_zero
            p_exact = mcnemar_exact(n_pos, n_neg)
            var = p_exact * (1.0 - p_exact)
            if n_perm * var < min_var:
                total += prob
    return total


def test_mcnemar_identity_mc_path(reps):
    """Standardize each non-degenerate dataset's deviation as
    z_j = (p_perm - p_exact) / se_mc and check the resulting sample's
    distribution, not its per-replication worst case: a bound sized for a
    single 4-sigma draw is, across `reps` independent datasets, actually a
    bound on max_j |z_j| -- whose expectation grows with reps (~4.45 sigma at
    reps=10,000), so it fails on correct code roughly 40% of the time at full
    scale. p_perm = (1 + B) / (n_perm + 1) with B ~ Binomial(n_perm, p_exact)
    under the identity, so both its known discreteness bias and its variance
    are computed exactly (not approximated as p_exact*(1-p_exact)/n_perm)
    before standardizing, so z_j is centered at 0 by construction and only
    real drift shows up as a mean/variance/tail shift.

    Datasets with Var(B) below MIN_VAR (dominated by p_exact == 1.0, where
    se_mc = 0) are excluded from the z-sample; the identity still requires
    p_perm == 1.0 exactly there (asserted directly), and the excluded
    fraction is checked against its exact theoretical value."""
    rows, failures = [], []
    for ci, delta in enumerate((0.0, 0.2)):
        data_seeds, eng_seeds = rep_seeds(970_000 + ci, reps)
        p_pos, p_neg = (PSI + delta) / 2.0, (PSI - delta) / 2.0
        z_list, excluded, tie_mismatches = [], 0, []
        for j in range(reps):
            d, _ = simulate_d_binary(delta=delta, psi=PSI, n_items=N_ITEMS,
                                     seed=int(data_seeds[j]))
            res = run_engine(d, seed=int(eng_seeds[j]), n_boot=50, n_perm=N_PERM)
            assert res.test_method == "sign-flip-mc"
            n_pos, n_neg, _ = discordant_counts(d)
            p_exact = mcnemar_exact(n_pos, n_neg)
            if p_exact == 1.0:
                # Deterministic under the identity: every sign-flip draw ties
                # or beats t_obs, so p_perm = (1+n_perm)/(n_perm+1) == 1.0 bit-for-bit.
                if res.p_value != 1.0:
                    tie_mismatches.append(
                        f"rep={j}: n_pos={n_pos} n_neg={n_neg} p_exact=1.0 but "
                        f"p_perm={res.p_value!r} != 1.0")
                excluded += 1
                continue
            var = p_exact * (1.0 - p_exact)
            if N_PERM * var < MIN_VAR:
                excluded += 1
                continue
            se_mc = np.sqrt(N_PERM * var) / (N_PERM + 1)
            bias = (1.0 - p_exact) / (N_PERM + 1)
            z_list.append((res.p_value - p_exact - bias) / se_mc)

        n = len(z_list)
        z = np.asarray(z_list)
        mean_z, rms_z = float(z.mean()), float(np.sqrt(np.mean(z ** 2)))
        frac2 = float(np.mean(np.abs(z) > 2))
        frac3 = float(np.mean(np.abs(z) > 3))
        max_abs_z = float(np.max(np.abs(z)))

        se_mean = 1.0 / np.sqrt(n)         # Var(z) ~= 1 under the identity
        se_rms = 1.0 / np.sqrt(2.0 * n)    # delta method on mean(chi-sq_1)
        se_frac2 = np.sqrt(0.0455 * 0.9545 / n)
        se_frac3 = np.sqrt(0.0027 * 0.9973 / n)
        # Gross-error tripwire only: expected max|z| over n std-normal draws,
        # two-sided ~ sqrt(2*ln(2n)); +1.5 margin so it never fires by chance.
        max_tol = np.sqrt(2.0 * np.log(2 * n)) + 1.5

        mean_bound = 4 * se_mean                    # two-sided: bias in either direction is a bug
        rms_upper_bound = 1.0 + 4 * se_rms          # one-sided: undershoot can't signal divergence
        frac2_bound, frac3_bound = 0.0455 + 4 * se_frac2, 0.0027 + 4 * se_frac3

        excluded_frac = excluded / reps
        theo_frac = _theoretical_excluded_fraction(p_pos, p_neg, N_ITEMS, N_PERM, MIN_VAR)
        excl_tol = 4 * mc_se(theo_frac, reps) + 0.01

        rows.append(dict(
            delta=delta, datasets=reps, included=n, excluded=excluded,
            excluded_frac=f"{excluded_frac:.5f}", excluded_theoretical=f"{theo_frac:.5f}",
            mean_z=f"{mean_z:.5f}", mean_bound=f"{mean_bound:.5f}",
            rms_z=f"{rms_z:.5f}", rms_upper_bound=f"{rms_upper_bound:.5f}",
            frac_gt2=f"{frac2:.5f}", frac_gt2_bound=f"{frac2_bound:.5f}",
            frac_gt3=f"{frac3:.5f}", frac_gt3_bound=f"{frac3_bound:.5f}",
            max_abs_z=f"{max_abs_z:.5f}", max_tripwire=f"{max_tol:.5f}"))

        if tie_mismatches:
            failures.append(f"delta={delta}: p_exact=1.0 bit-exact identity broken:\n  " +
                            "\n  ".join(tie_mismatches[:10]))
        if abs(excluded_frac - theo_frac) > excl_tol:
            failures.append(f"delta={delta}: excluded fraction {excluded_frac:.4f} vs "
                            f"theoretical {theo_frac:.4f} (tol {excl_tol:.4f}) -- "
                            f"simulator/exact-path mismatch")
        if abs(mean_z) > mean_bound:
            failures.append(f"delta={delta}: mean(z)={mean_z:.4f} exceeds "
                            f"4*SE={mean_bound:.4f} -- systematic bias")
        if rms_z > rms_upper_bound:
            failures.append(f"delta={delta}: RMS(z)={rms_z:.4f} exceeds "
                            f"1 + 4*SE={rms_upper_bound:.4f}")
        if frac2 > frac2_bound:
            failures.append(f"delta={delta}: P(|z|>2)={frac2:.4f} exceeds {frac2_bound:.4f}")
        if frac3 > frac3_bound:
            failures.append(f"delta={delta}: P(|z|>3)={frac3:.4f} exceeds {frac3_bound:.4f}")
        if max_abs_z > max_tol:
            failures.append(f"delta={delta}: max|z|={max_abs_z:.4f} exceeds "
                            f"gross-error tripwire {max_tol:.4f}")
    publish("mcnemar_identity_mc", reps, rows)
    assert not failures, "Sign-flip MC diverged from exact McNemar:\n" + "\n".join(failures)


def test_counterbalancing_removes_additive_position_bias(reps):
    theta = 0.6
    bias = 0.15
    # Unequal order counts (3 A-first : 1 B-first): naive call-level pooling
    # inherits bias * (3-1)/4 = 0.075; the order-balanced reduction does not.
    seeds, = rep_seeds(971_000, reps, streams=1)
    cb, naive = [], []
    for i in range(reps):
        records, _ = simulate_pairwise_records(
            n_items=100, theta=theta, tie_rate=0.1, position_bias=bias,
            calls_a_first=3, calls_b_first=1, seed=int(seeds[i]))
        red = reduce_pairwise(records, "A", "B")
        cb.append(float(np.mean(red.s)))
        ys = []
        for rec in records:
            a_first = rec["variant_first"] == "A"
            if rec["judgment"] == "tie":
                ys.append(0.5)
            elif rec["judgment"] == "first":
                ys.append(1.0 if a_first else 0.0)
            else:
                ys.append(0.0 if a_first else 1.0)
        naive.append(float(np.mean(ys)))
    cb_mean, naive_mean = float(np.mean(cb)), float(np.mean(naive))
    cb_se = float(np.std(cb, ddof=1) / np.sqrt(reps))
    publish("counterbalancing", reps, [dict(
        true_theta=theta, counterbalanced_mean=f"{cb_mean:.5f}",
        counterbalanced_bias=f"{cb_mean - theta:.5f}", mc_se=f"{cb_se:.5f}",
        naive_pooled_mean=f"{naive_mean:.5f}",
        naive_bias=f"{naive_mean - theta:.5f}", injected_bias=bias)])
    assert abs(cb_mean - theta) <= 4.0 * cb_se + 1e-9, (
        f"counterbalanced estimator biased: {cb_mean - theta:.5f} "
        f"(> 4*MC-SE {4*cb_se:.5f}); §1.2 order-balancing broken")
    # Teeth: the naive estimator must show the injected bias (~0.075).
    assert naive_mean - theta > 0.05, (
        f"naive pooling shows bias {naive_mean - theta:.5f} < 0.05 — the "
        f"injected position effect is not reaching the data; study invalid")
