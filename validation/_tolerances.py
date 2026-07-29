"""Shared acceptance bands and result publication for the validation suite.

Band philosophy (spec §11, DECISION D8): the spec's acceptance bands assume
10,000 replications. At smaller replication counts (smoke runs, default 200)
the bands widen by 3x the binomial Monte Carlo SE so that smoke runs fail
only on real drift, and at 10,000 reps they reduce exactly to the spec's
numbers: coverage half-width max(0.015, 3*MC-SE) -> [93.5%, 96.5%] at 10k;
FPR bound alpha + 3*MC-SE -> ~5.65% at 10k.

Asymmetry rule (§11.1-11.2): anti-conservatism (under-coverage, inflated
FPR) fails the build; conservatism is recorded in the published table, not
failed. Coverage asserts therefore bind on the lower edge; the upper edge
is recorded.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def mc_se(p: float, reps: int) -> float:
    return float(np.sqrt(p * (1.0 - p) / reps))


def coverage_lower_bound(reps: int, nominal: float = 0.95, extra_low: float = 0.0) -> float:
    """Lower acceptance edge for CI coverage. ``extra_low`` is the documented
    allowance for known cases (F2 small-C percentile undercoverage, discrete
    outcomes); it must be justified per cell where used."""
    half = max(0.015, 3.0 * mc_se(nominal, reps))
    return nominal - half - extra_low


def coverage_upper_edge(reps: int, nominal: float = 0.95) -> float:
    """Upper edge, recorded (conservatism) but not build-failing."""
    half = max(0.015, 3.0 * mc_se(nominal, reps))
    return min(1.0, nominal + half)


def fpr_bound(reps: int, alpha: float = 0.05, extra: float = 0.0) -> float:
    """§11.2: rejection rate must not exceed alpha + 3*MC-SE (+ documented
    extra for cells outside the exact null)."""
    return alpha + 3.0 * mc_se(alpha, reps) + extra


def power_tolerance(reps: int, expected: float) -> float:
    """§11.3: analytic-vs-simulated agreement within 3 points, plus MC error."""
    return 0.03 + 3.0 * mc_se(expected, reps)


def rep_seeds(base_seed: int, reps: int, streams: int = 2):
    """Deterministic per-replication seed arrays (data stream, engine stream, ...)."""
    state = np.random.SeedSequence(base_seed).generate_state(streams * reps)
    return [state[k * reps:(k + 1) * reps] for k in range(streams)]


def publish(name: str, reps: int, rows: list[dict]) -> Path:
    """Print the measured table and persist it to validation/results/
    (§11.1 requires measured coverage to be published, not just asserted)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}_reps{reps}.csv"
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    print(f"\n[{name}, reps={reps}] -> {path}")
    print("  " + "  ".join(k.ljust(widths[k]) for k in keys))
    for r in rows:
        print("  " + "  ".join(str(r[k]).ljust(widths[k]) for k in keys))
    return path
