"""§11.1, F2 — measured coverage evidence backing DECISION D9
(docs/statistics-spec.md §2.1, §2.3, §13): below ``MIN_CLUSTERS_ANALYTIC_PRIMARY``
clusters the analytic cluster-robust interval is primary and the percentile
bootstrap is reported alongside it; at or above it, the roles are as
originally specified (bootstrap primary, analytic cross-check).

Single source for these figures: docs/statistics-spec.md quotes this table
rather than repeating the numbers, and scenarios.py's report builder reads
the threshold and lookup from here rather than each call site hardcoding its
own copy.

Measured at reps=10,000 (validation/test_coverage.py, §11.1); nominal 95%
coverage, MC-SE per cell = sqrt(0.95*0.05/10000) ~= 0.0022 (~0.22 points).
Worst cell per C is reported where a C has multiple grid cells (rho, skew).
"""

from __future__ import annotations

NOMINAL_COVERAGE = 0.95
MC_SE_10K = 0.0022

# C -> (bootstrap coverage, analytic coverage), worst measured cell at that C,
# read from validation/results/coverage_reps10000.csv (validation/test_coverage.py
# _cells()) -- that CSV is the underlying evidence; this table is the single
# in-code copy of the figures relevant to DECISION D9.
#   C=20: worst of cluster-normal-C20-rho{0.2,0.5}, cluster-skewed-C20-rho0.5
#         -> cluster-skewed-C20-rho0.5
#   C=30: worst of cluster-normal-C30-rho{0.2,0.5} -> rho0.2
#   C=50: worst of cluster-normal-C50-rho{0.2,0.5} -> rho0.2
#   C=200 (unclustered): scalar-normal-n200
COVERAGE_BY_C: dict[int, dict[str, float]] = {
    20: dict(bootstrap=0.9253, analytic=0.9494),
    30: dict(bootstrap=0.9348, analytic=0.9502),
    50: dict(bootstrap=0.9415, analytic=0.9505),
    200: dict(bootstrap=0.9472, analytic=0.9500),
}

# DECISION D9 (docs/statistics-spec.md §13): acceptable coverage shortfall is
# 1 percentage point below nominal (94.0%) -- comfortably above the ~0.22-point
# MC-SE noise floor at 10k reps, so it is a real cutoff, not measurement noise.
# The smallest measured C at which the bootstrap shortfall first drops under
# that floor (C=50, 0.942) sets the threshold. See the spec for alternatives
# considered and why they were rejected.
ACCEPTABLE_SHORTFALL = 0.01
MIN_CLUSTERS_ANALYTIC_PRIMARY = 50

__all__ = [
    "ACCEPTABLE_SHORTFALL",
    "COVERAGE_BY_C",
    "MC_SE_10K",
    "MIN_CLUSTERS_ANALYTIC_PRIMARY",
    "NOMINAL_COVERAGE",
    "coverage_disclosure",
    "primary_ci_method",
]


def primary_ci_method(n_clusters: int) -> str:
    """'analytic' below the measured-safe cluster count, else 'bootstrap'
    (DECISION D9)."""
    return "analytic" if n_clusters < MIN_CLUSTERS_ANALYTIC_PRIMARY else "bootstrap"


def coverage_disclosure(n_clusters: int) -> str:
    """§10 disclosure line for the report block. States what validation
    measured under simulated conditions at a nearby C, not a claim about
    this run's own data -- this run's own coverage was not measured."""
    primary = primary_ci_method(n_clusters)
    if n_clusters < min(COVERAGE_BY_C):
        evidence = (f"C={n_clusters} is below the smallest cluster count in the "
                    f"measured grid (C={min(COVERAGE_BY_C)}); coverage this small is "
                    f"unmeasured, not merely degraded (§2.4 F4 applies)")
    else:
        c_ref = max(c for c in COVERAGE_BY_C if c <= n_clusters)
        row = COVERAGE_BY_C[c_ref]
        evidence = (f"in simulation at C={c_ref} (reps=10,000, §11.1), measured "
                    f"{NOMINAL_COVERAGE:.0%}-nominal coverage was "
                    f"{row['bootstrap']:.1%} for the percentile bootstrap and "
                    f"{row['analytic']:.1%} for the analytic interval; this run's "
                    f"own coverage was not measured and may differ")
    return (f"CI primary method at this run's C={n_clusters}: {primary} "
           f"(DECISION D9, docs/statistics-spec.md §2.1/§2.3). {evidence}.")
