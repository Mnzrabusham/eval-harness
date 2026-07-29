"""A priori power, sample size, and MDE (docs/statistics-spec.md §9).

Analytic formulas are normal approximations refined by iterating with t
quantiles to convergence (|n_new - n_old| < 1), per §9.1. The
simulation-based tool (§9.2) runs the actual §2 pipeline on synthetic data
and is normative for anything the formulas do not cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np
from scipy import optimize as _opt
from scipy import stats as _sps

from .engine import run_engine

__all__ = [
    "SimulatedPower",
    "design_effect",
    "mde_binary_paired",
    "mde_paired_mean",
    "mde_simulated",
    "n_binary_paired",
    "n_paired_mean",
    "power_binary_paired",
    "power_paired_mean",
    "simulated_power",
]

_MAX_ITER = 100


def design_effect(mean_cluster_size: float, rho: float) -> float:
    """DEFF = 1 + (mean cluster size - 1) * rho (§9.1)."""
    if mean_cluster_size < 1 or not 0 <= rho <= 1:
        raise ValueError("mean_cluster_size >= 1 and rho in [0, 1] required")
    return 1.0 + (mean_cluster_size - 1.0) * rho


def n_paired_mean(delta: float, sd_d: float, *, alpha: float = 0.05,
                  power: float = 0.8, deff: float = 1.0) -> int:
    """Items needed to detect a paired mean difference delta with SD(d) = sd_d
    (§9.1), t-refined, multiplied by the design effect."""
    if delta == 0 or sd_d <= 0:
        raise ValueError("delta != 0 and sd_d > 0 required")
    delta = abs(delta)
    za, zp = _sps.norm.ppf(1 - alpha / 2), _sps.norm.ppf(power)
    n = (za + zp) ** 2 * sd_d**2 / delta**2
    for _ in range(_MAX_ITER):
        df = max(2, ceil(n) - 1)
        ta, tp = _sps.t.ppf(1 - alpha / 2, df), _sps.t.ppf(power, df)
        n_new = (ta + tp) ** 2 * sd_d**2 / delta**2
        if abs(n_new - n) < 1:
            n = n_new
            break
        n = n_new
    return int(ceil(n * deff))


def mde_paired_mean(n: int, sd_d: float, *, alpha: float = 0.05,
                    power: float = 0.8, df: int | None = None) -> float:
    """MDE = (t_{1-a/2} + t_{power}) * sd_d / sqrt(n) (§9.1, §9.3). Pass
    df = C - 1 under clustering."""
    if n < 2 or sd_d <= 0:
        raise ValueError("n >= 2 and sd_d > 0 required")
    df = (n - 1) if df is None else df
    if df < 1:
        raise ValueError("df >= 1 required")
    return float((_sps.t.ppf(1 - alpha / 2, df) + _sps.t.ppf(power, df)) * sd_d / sqrt(n))


def power_paired_mean(n: int, delta: float, sd_d: float, *, alpha: float = 0.05,
                      df: int | None = None) -> float:
    """Exact two-sided paired-t power under normality (noncentral t); the
    reference curve for §11.3 power-agreement validation."""
    if n < 2 or sd_d <= 0:
        raise ValueError("n >= 2 and sd_d > 0 required")
    df = (n - 1) if df is None else df
    ncp = sqrt(n) * delta / sd_d
    tcrit = _sps.t.ppf(1 - alpha / 2, df)
    return float(_sps.nct.sf(tcrit, df, ncp) + _sps.nct.cdf(-tcrit, df, ncp))


def n_binary_paired(delta: float, psi: float, *, alpha: float = 0.05,
                    power: float = 0.8) -> int:
    """Connor (1987) paired binary sample size (§9.1):
    ``n = [z_{1-a/2} sqrt(psi) + z_power sqrt(psi - delta^2)]^2 / delta^2``,
    t-refined. psi is the discordance rate and is as important an input as
    delta."""
    if not 0 < psi <= 1 or delta == 0 or abs(delta) > psi:
        raise ValueError("need 0 < psi <= 1 and 0 < |delta| <= psi")
    delta = abs(delta)

    def solve(qa, qp):
        return (qa * sqrt(psi) + qp * sqrt(psi - delta**2)) ** 2 / delta**2

    za, zp = _sps.norm.ppf(1 - alpha / 2), _sps.norm.ppf(power)
    n = solve(za, zp)
    for _ in range(_MAX_ITER):
        df = max(2, ceil(n) - 1)
        n_new = solve(_sps.t.ppf(1 - alpha / 2, df), _sps.t.ppf(power, df))
        if abs(n_new - n) < 1:
            n = n_new
            break
        n = n_new
    return int(ceil(n))


def power_binary_paired(n: int, delta: float, psi: float, *, alpha: float = 0.05) -> float:
    """Normal-approximation power of the paired binary test (Connor 1987)."""
    if not 0 < psi <= 1 or abs(delta) > psi or n < 1:
        raise ValueError("need 0 < psi <= 1, |delta| <= psi, n >= 1")
    if psi - delta**2 <= 0:
        return 1.0
    za = _sps.norm.ppf(1 - alpha / 2)
    z = (sqrt(n) * abs(delta) - za * sqrt(psi)) / sqrt(psi - delta**2)
    return float(_sps.norm.cdf(z))


def mde_binary_paired(n: int, psi: float, *, alpha: float = 0.05,
                      power: float = 0.8) -> float:
    """Smallest |delta| reaching the target power at n items (§9.3's worked
    example: n=50, psi=0.3 gives MDE ~ 0.21)."""
    if not 0 < psi <= 1 or n < 1:
        raise ValueError("need 0 < psi <= 1 and n >= 1")
    hi = psi * (1 - 1e-9)
    if power_binary_paired(n, hi, psi, alpha=alpha) < power:
        raise ValueError("target power unreachable at this n; increase n")

    def f(x):
        return power_binary_paired(n, x, psi, alpha=alpha) - power

    return float(_opt.brentq(f, 1e-12, hi, xtol=1e-10))


@dataclass(frozen=True)
class SimulatedPower:
    power: float
    mc_se: float
    reps: int
    alpha: float


def simulated_power(simulate_fn, *, reps: int, seed: int, alpha: float = 0.05,
                    n_boot: int = 200, n_perm: int = 1000,
                    alternative: str = "two-sided") -> SimulatedPower:
    """§9.2: simulate datasets and run the ACTUAL §2 pipeline on each.

    ``simulate_fn(data_seed) -> (d, clusters)``. Rejection is
    ``p_value <= alpha`` from the sign-flip test. Per-rep data and engine
    seeds are derived deterministically from ``seed`` via SeedSequence.
    """
    if reps < 1:
        raise ValueError("reps >= 1 required")
    state = np.random.SeedSequence(seed).generate_state(2 * reps)
    rejections = 0
    for i in range(reps):
        d, clusters = simulate_fn(int(state[i]))
        res = run_engine(d, clusters, seed=int(state[reps + i]), alpha=alpha,
                         n_boot=n_boot, n_perm=n_perm, alternative=alternative)
        rejections += res.p_value <= alpha
    p = rejections / reps
    return SimulatedPower(p, sqrt(p * (1 - p) / reps), reps, alpha)


def mde_simulated(make_simulate_fn, lo: float, hi: float, *, target_power: float = 0.8,
                  reps: int = 1000, seed: int, alpha: float = 0.05, tol: float = 0.01,
                  max_iter: int = 20, n_boot: int = 200, n_perm: int = 1000):
    """§9.2 MDE by bisection on delta until simulated power is within ``tol``
    of the target. ``make_simulate_fn(delta) -> simulate_fn``. The same seed
    is reused at every bisection step (common random numbers).

    Returns (delta, SimulatedPower at delta).
    """
    if not lo < hi:
        raise ValueError("need lo < hi")
    mid, result = hi, None
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        result = simulated_power(make_simulate_fn(mid), reps=reps, seed=seed,
                                 alpha=alpha, n_boot=n_boot, n_perm=n_perm)
        if abs(result.power - target_power) <= tol:
            return mid, result
        if result.power < target_power:
            lo = mid
        else:
            hi = mid
    return mid, result
