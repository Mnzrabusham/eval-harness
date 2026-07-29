"""§11.1 — Empirical coverage of the 95% CIs (bootstrap §2.1, analytic §2.3).

Synthetic data with known true Delta; coverage = fraction of replications
whose CI contains the truth. Cells at C in {20, 30, 50} clusters measure
FLAGGED F2 (percentile bootstrap undercoverage at small C) — F2 predicts
~1-3 points below nominal at C ~ 20, so those cells carry a documented
extra_low allowance rather than a silent pass; they still fail loudly if
coverage collapses beyond what F2 predicts.

Build-failing edge: under-coverage below the band. Over-coverage
(conservatism) is recorded in the published table per §11's asymmetry rule.
"""

import numpy as np
import pytest

from evalkit.stats import run_engine
from evalkit.stats.simulate import (
    ClusterDesign,
    simulate_d_binary,
    simulate_d_preference,
    simulate_d_scalar,
)
from validation._tolerances import (
    coverage_lower_bound,
    coverage_upper_edge,
    publish,
    rep_seeds,
)

pytestmark = pytest.mark.validation

N_BOOT = 2000
N_PERM = 199  # permutation p not under test here; keep cheap

DELTA = 0.3


def _cells():
    cells = []

    def scalar(name, n, dist, extra_boot, extra_t):
        cells.append(dict(
            name=name, truth=DELTA, extra_boot=extra_boot, extra_t=extra_t,
            make=lambda seed, n=n, dist=dist: simulate_d_scalar(
                delta=DELTA, sd=1.0, dist=dist, n_items=n, seed=seed),
        ))

    # Unclustered grid (spec §11.1: n in {20, 50, 200}).
    scalar("scalar-normal-n20", 20, "normal", 0.03, 0.02)   # small-C: F2/F4
    scalar("scalar-normal-n50", 50, "normal", 0.0, 0.0)
    scalar("scalar-normal-n200", 200, "normal", 0.0, 0.0)
    scalar("scalar-skewed-n50", 50, "skewed", 0.02, 0.02)   # F2: skew undercoverage

    # Binary and preference (discrete d: bootstrap is coarse at small n_d).
    cells.append(dict(name="binary-psi0.3-n50", truth=0.1, extra_boot=0.02, extra_t=0.02,
                      make=lambda seed: simulate_d_binary(delta=0.1, psi=0.3, n_items=50, seed=seed)))
    cells.append(dict(name="binary-psi0.1-n200", truth=0.04, extra_boot=0.02, extra_t=0.02,
                      make=lambda seed: simulate_d_binary(delta=0.04, psi=0.1, n_items=200, seed=seed)))
    cells.append(dict(name="pref-theta0.6-tie0.2-n50", truth=0.2, extra_boot=0.02, extra_t=0.02,
                      make=lambda seed: simulate_d_preference(theta=0.6, tie_rate=0.2, n_items=50, seed=seed)))

    # Clustered cells at C = 20, 30, 50 (F2 measurement grid; decision-log
    # F2 predicts ~1-3 points undercoverage at C ~ 20).
    for C, extra_boot, extra_t in ((20, 0.035, 0.02), (30, 0.025, 0.015), (50, 0.015, 0.01)):
        for rho in (0.2, 0.5):
            cells.append(dict(
                name=f"cluster-normal-C{C}-rho{rho}", truth=DELTA,
                extra_boot=extra_boot, extra_t=extra_t,
                make=lambda seed, C=C, rho=rho: simulate_d_scalar(
                    delta=DELTA, sd=1.0, dist="normal",
                    cluster=ClusterDesign(C, 5, rho), seed=seed),
            ))
    # Skew + few clusters: the F2 worst case.
    cells.append(dict(
        name="cluster-skewed-C20-rho0.5", truth=DELTA, extra_boot=0.05, extra_t=0.03,
        make=lambda seed: simulate_d_scalar(delta=DELTA, sd=1.0, dist="skewed",
                                            cluster=ClusterDesign(20, 5, 0.5), seed=seed),
    ))
    return cells


def test_ci_coverage(reps):
    rows, failures = [], []
    for ci, cell in enumerate(_cells()):
        data_seeds, eng_seeds = rep_seeds(910_000 + ci, reps)
        truth = cell["truth"]
        hit_boot = hit_t = n_t = 0
        for j in range(reps):
            d, clusters = cell["make"](int(data_seeds[j]))
            res = run_engine(d, clusters, seed=int(eng_seeds[j]),
                             n_boot=N_BOOT, n_perm=N_PERM)
            hit_boot += res.ci_low <= truth <= res.ci_high
            a = res.analytic
            if np.isfinite(a.ci_low):
                n_t += 1
                hit_t += a.ci_low <= truth <= a.ci_high
        cov_b = hit_boot / reps
        cov_t = hit_t / n_t if n_t else float("nan")
        lo_b = coverage_lower_bound(reps, extra_low=cell["extra_boot"])
        lo_t = coverage_lower_bound(reps, extra_low=cell["extra_t"])
        hi = coverage_upper_edge(reps)
        rows.append(dict(cell=cell["name"], boot_coverage=f"{cov_b:.4f}",
                         boot_lower_bound=f"{lo_b:.4f}",
                         analytic_coverage=f"{cov_t:.4f}",
                         analytic_lower_bound=f"{lo_t:.4f}",
                         upper_edge_recorded=f"{hi:.4f}",
                         conservative=("yes" if max(cov_b, cov_t) > hi else "no")))
        if cov_b < lo_b:
            failures.append(f"{cell['name']}: bootstrap coverage {cov_b:.4f} < {lo_b:.4f}")
        if np.isfinite(cov_t) and cov_t < lo_t:
            failures.append(f"{cell['name']}: analytic coverage {cov_t:.4f} < {lo_t:.4f}")
    publish("coverage", reps, rows)
    assert not failures, "CI coverage drifted below the acceptance band:\n" + "\n".join(failures)
