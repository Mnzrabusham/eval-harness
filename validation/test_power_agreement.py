"""§11.3 — Simulated power (running the actual §2 pipeline) vs the §9.1
analytic formulas on the formulas' home turf (no clustering, normal-ish).
Agreement within ±3 points (+ MC error at smoke scale), else the analytic
implementation is wrong.
"""

import pytest

from evalkit.stats import power_binary_paired, power_paired_mean, simulated_power
from evalkit.stats.simulate import simulate_d_binary, simulate_d_scalar
from validation._tolerances import power_tolerance, publish

pytestmark = pytest.mark.validation

N_PERM = 999
N_BOOT = 100


def _cells():
    return [
        dict(name="paired-mean-n50-delta0.31",
             analytic=power_paired_mean(50, 0.31, 1.0),
             make=lambda delta=0.31: (lambda s: simulate_d_scalar(
                 delta=delta, sd=1.0, dist="normal", n_items=50, seed=s))),
        dict(name="paired-mean-n30-delta0.52",
             analytic=power_paired_mean(30, 0.52, 1.0),
             make=lambda delta=0.52: (lambda s: simulate_d_scalar(
                 delta=delta, sd=1.0, dist="normal", n_items=30, seed=s))),
        dict(name="binary-connor-n200-psi0.3-delta0.10",
             analytic=power_binary_paired(200, 0.10, 0.3),
             make=lambda: (lambda s: simulate_d_binary(
                 delta=0.10, psi=0.3, n_items=200, seed=s))),
    ]


def test_simulated_power_matches_analytic(reps):
    rows, failures = [], []
    for ci, cell in enumerate(_cells()):
        sim = simulated_power(cell["make"](), reps=reps, seed=940_000 + ci,
                              n_boot=N_BOOT, n_perm=N_PERM)
        analytic = cell["analytic"]
        tol = power_tolerance(reps, analytic)
        diff = abs(sim.power - analytic)
        rows.append(dict(cell=cell["name"], analytic_power=f"{analytic:.4f}",
                         simulated_power=f"{sim.power:.4f}", diff=f"{diff:.4f}",
                         tolerance=f"{tol:.4f}"))
        if diff > tol:
            failures.append(f"{cell['name']}: |sim {sim.power:.4f} - analytic "
                            f"{analytic:.4f}| = {diff:.4f} > {tol:.4f}")
    publish("power_agreement", reps, rows)
    assert not failures, "Analytic power disagrees with the simulated pipeline:\n" + "\n".join(failures)
