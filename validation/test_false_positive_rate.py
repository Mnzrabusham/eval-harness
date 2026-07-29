"""§11.2 — False positive rate under TRUE (exchangeability) nulls.

Sign-flip test and McNemar cross-check under exchangeable-variant nulls:
rejection rate at alpha = 0.05 must not exceed alpha + 3*MC-SE in any cell.
Anti-conservatism fails the build; conservatism (common for the discrete
cells) is recorded, not failed.

Every generator here produces a SYMMETRIC null: "skew-null" is the
difference of two iid gammas — what judging identically-distributed skewed
scores actually yields under H0. The asymmetric mean-zero cells live in
test_null_robustness_probes.py (§11.8), which probes the weaker null and is
not build-failing.
"""

import numpy as np
import pytest

from evalkit.stats import discordant_counts, mcnemar_exact, run_engine
from evalkit.stats.simulate import (
    ClusterDesign,
    simulate_d_binary,
    simulate_d_preference,
    simulate_d_scalar,
)
from validation._tolerances import fpr_bound, mc_se, publish, rep_seeds

pytestmark = pytest.mark.validation

ALPHA = 0.05
N_PERM = 2000
N_BOOT = 100  # CI not under test here


def _cells():
    return [
        dict(name="scalar-normal-n50",
             make=lambda s: simulate_d_scalar(delta=0.0, sd=1.0, dist="normal", n_items=50, seed=s)),
        dict(name="scalar-skewnull-n50",
             make=lambda s: simulate_d_scalar(delta=0.0, sd=1.0, dist="skew-null", n_items=50, seed=s)),
        dict(name="scalar-skewnull-n20",
             make=lambda s: simulate_d_scalar(delta=0.0, sd=1.0, dist="skew-null", n_items=20, seed=s)),
        dict(name="binary-psi0.3-n50", binary=True,
             make=lambda s: simulate_d_binary(delta=0.0, psi=0.3, n_items=50, seed=s)),
        dict(name="binary-psi0.1-n100", binary=True,
             make=lambda s: simulate_d_binary(delta=0.0, psi=0.1, n_items=100, seed=s)),
        dict(name="pref-theta0.5-tie0.2-n50",
             make=lambda s: simulate_d_preference(theta=0.5, tie_rate=0.2, n_items=50, seed=s)),
        dict(name="cluster-normal-C20-rho0.5",
             make=lambda s: simulate_d_scalar(delta=0.0, sd=1.0, dist="normal",
                                              cluster=ClusterDesign(20, 5, 0.5), seed=s)),
        dict(name="cluster-normal-C30-rho0.2",
             make=lambda s: simulate_d_scalar(delta=0.0, sd=1.0, dist="normal",
                                              cluster=ClusterDesign(30, 5, 0.2), seed=s)),
        dict(name="cluster-skewnull-C20-rho0.5",
             make=lambda s: simulate_d_scalar(delta=0.0, sd=1.0, dist="skew-null",
                                              cluster=ClusterDesign(20, 5, 0.5), seed=s)),
    ]


def test_false_positive_rate_under_true_null(reps):
    bound = fpr_bound(reps, ALPHA)
    rows, failures = [], []
    for ci, cell in enumerate(_cells()):
        data_seeds, eng_seeds = rep_seeds(920_000 + ci, reps)
        rej_perm = rej_t = rej_mcn = 0
        n_mcn = 0
        for j in range(reps):
            d, clusters = cell["make"](int(data_seeds[j]))
            res = run_engine(d, clusters, seed=int(eng_seeds[j]),
                             n_boot=N_BOOT, n_perm=N_PERM)
            rej_perm += res.p_value <= ALPHA
            if np.isfinite(res.analytic.p_value):
                rej_t += res.analytic.p_value <= ALPHA
            if cell.get("binary"):
                n_pos, n_neg, _ = discordant_counts(d)
                n_mcn += 1
                rej_mcn += mcnemar_exact(n_pos, n_neg) <= ALPHA
        fpr_perm = rej_perm / reps
        fpr_t = rej_t / reps
        fpr_mcn = rej_mcn / n_mcn if n_mcn else float("nan")
        rows.append(dict(cell=cell["name"], signflip_fpr=f"{fpr_perm:.4f}",
                         analytic_fpr=f"{fpr_t:.4f}", mcnemar_fpr=f"{fpr_mcn:.4f}",
                         bound=f"{bound:.4f}",
                         conservative=("yes" if fpr_perm < ALPHA - 3 * mc_se(ALPHA, reps) else "no")))
        # Build-failing: the primary sign-flip test and the McNemar cross-check.
        if fpr_perm > bound:
            failures.append(f"{cell['name']}: sign-flip FPR {fpr_perm:.4f} > {bound:.4f}")
        if n_mcn and fpr_mcn > bound:
            failures.append(f"{cell['name']}: McNemar FPR {fpr_mcn:.4f} > {bound:.4f}")
        # Analytic t is the secondary cross-check; small-C liberalism is a
        # known property, so it gets a small documented allowance.
        if fpr_t > fpr_bound(reps, ALPHA, extra=0.02):
            failures.append(f"{cell['name']}: analytic FPR {fpr_t:.4f} > {fpr_bound(reps, ALPHA, extra=0.02):.4f}")
    publish("false_positive_rate", reps, rows)
    assert not failures, "FPR exceeded the anti-conservatism bound:\n" + "\n".join(failures)
