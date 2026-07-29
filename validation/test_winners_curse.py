"""§11.7 — Winner's curse demonstration (doubles as documentation).

k candidate variants with IDENTICAL true effect Delta vs a shared control:
the empirical best's estimate is biased upward conditional on selection,
while a pre-designated variant's estimate is unbiased. The measured
selection bias is published; the assert requires the phenomenon to be
clearly detected (it is why §6 forbids testing 'best vs rest' on the data
that selected it).
"""

import numpy as np
import pytest

from validation._tolerances import publish, rep_seeds

pytestmark = pytest.mark.validation

K = 5          # 4 candidates + control
TRUE_DELTA = 0.1
N_ITEMS = 50


def test_selection_bias_of_empirical_best(reps):
    seeds, = rep_seeds(980_000, reps, streams=1)
    best_est, fixed_est = [], []
    for i in range(reps):
        rng = np.random.Generator(np.random.PCG64(int(seeds[i])))
        x = rng.normal(0.0, 1.0, (K, N_ITEMS))
        x[1:] += TRUE_DELTA
        deltas = (x[1:] - x[0]).mean(axis=1)  # contrasts share the control arm
        best_est.append(float(deltas.max()))
        fixed_est.append(float(deltas[0]))    # pre-designated candidate
    best_bias = float(np.mean(best_est) - TRUE_DELTA)
    fixed_bias = float(np.mean(fixed_est) - TRUE_DELTA)
    se_best = float(np.std(best_est, ddof=1) / np.sqrt(reps))
    se_fixed = float(np.std(fixed_est, ddof=1) / np.sqrt(reps))
    publish("winners_curse", reps, [dict(
        k_candidates=K - 1, true_delta=TRUE_DELTA, n_items=N_ITEMS,
        mean_best_estimate=f"{np.mean(best_est):.5f}",
        selection_bias=f"{best_bias:.5f}", mc_se=f"{se_best:.5f}",
        fixed_candidate_bias=f"{fixed_bias:.5f}", fixed_mc_se=f"{se_fixed:.5f}")])
    # The pre-designated candidate is unbiased...
    assert abs(fixed_bias) <= 4.0 * se_fixed + 1e-9
    # ...while the empirical best must show clear upward selection bias
    # (theory: ~ sd(contrast diff) * E[max of 4 correlated normals] ~ 0.14 here).
    assert best_bias > max(0.03, 4.0 * se_best), (
        f"winner's-curse bias {best_bias:.5f} not detected — simulation or "
        f"estimator drift")
