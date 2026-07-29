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

import numpy as np
import pytest

from evalkit.stats import discordant_counts, mcnemar_exact, reduce_pairwise, run_engine
from evalkit.stats.simulate import simulate_d_binary, simulate_pairwise_records
from validation._tolerances import publish, rep_seeds

pytestmark = pytest.mark.validation

N_PERM = 4999


def test_mcnemar_identity_mc_path(reps):
    rows, failures = [], []
    for ci, delta in enumerate((0.0, 0.2)):
        data_seeds, eng_seeds = rep_seeds(970_000 + ci, reps)
        max_diff = 0.0
        for j in range(reps):
            d, _ = simulate_d_binary(delta=delta, psi=0.4, n_items=60,
                                     seed=int(data_seeds[j]))
            res = run_engine(d, seed=int(eng_seeds[j]), n_boot=50, n_perm=N_PERM)
            assert res.test_method == "sign-flip-mc"
            n_pos, n_neg, _ = discordant_counts(d)
            p_exact = mcnemar_exact(n_pos, n_neg)
            diff = abs(res.p_value - p_exact)
            max_diff = max(max_diff, diff)
            # MC tolerance: 4 sigma of the permutation estimate + the +1 correction.
            tol = 4.0 * np.sqrt(max(p_exact * (1 - p_exact), 1e-4) / N_PERM) + 2.0 / (N_PERM + 1)
            if diff > tol:
                failures.append(f"delta={delta} rep={j}: |p_perm - p_mcnemar| = "
                                f"{diff:.5f} > {tol:.5f} (p_exact={p_exact:.5f})")
        rows.append(dict(delta=delta, datasets=reps, max_abs_diff=f"{max_diff:.5f}"))
    publish("mcnemar_identity_mc", reps, rows)
    assert not failures, "Sign-flip MC diverged from exact McNemar:\n" + "\n".join(failures[:10])


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
