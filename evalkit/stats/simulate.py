"""Synthetic eval data with known ground truth (docs/statistics-spec.md §9.2, §11).

One simulator, two consumers: the simulation-based power tool (§9.2) and the
validation studies in ``validation/``. No network calls; everything is
generated in-process from an explicit seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import make_rng

__all__ = [
    "ClusterDesign",
    "simulate_d_binary",
    "simulate_d_meanzero_skewed",
    "simulate_d_preference",
    "simulate_d_scalar",
    "simulate_pairwise_records",
    "simulate_score_records",
]


@dataclass(frozen=True)
class ClusterDesign:
    """C clusters of equal size with within-cluster correlation rho of d_i."""

    n_clusters: int
    cluster_size: int
    rho: float


def _noise(rng, dist: str, k: int, var: float) -> np.ndarray:
    """Mean-zero noise with the given variance.

    dist:
      "normal"    — Gaussian.
      "skewed"    — centered gamma (skewness ~1.4). Mean-zero but ASYMMETRIC:
                    legitimate for CI-coverage cells; OUTSIDE the §2.2
                    exchangeability null when used as a testing null.
      "skew-null" — difference of two iid gammas: symmetric, heavy-tailed;
                    the null-valid analogue of judging skewed scores.
    """
    s = float(np.sqrt(var))
    if dist == "normal":
        return rng.normal(0.0, s, k)
    if dist == "skewed":
        shape = 2.0
        theta = s / np.sqrt(shape)
        return rng.gamma(shape, theta, k) - shape * theta
    if dist == "skew-null":
        shape = 2.0
        theta = s / np.sqrt(2.0 * shape)
        return rng.gamma(shape, theta, k) - rng.gamma(shape, theta, k)
    raise ValueError(f"unknown dist {dist!r}")


def simulate_d_scalar(*, delta: float, sd: float = 1.0, dist: str = "normal",
                      n_items: int | None = None, cluster: ClusterDesign | None = None,
                      seed: int):
    """Item-level scalar differences with E[d_i] = delta and SD(d_i) = sd.

    Returns (d, clusters); clusters is None (singletons) without a
    ClusterDesign. Cluster effects are always normal; the chosen dist shapes
    the item-level noise.
    """
    rng = make_rng(seed)
    if cluster is None:
        if n_items is None:
            raise ValueError("n_items required without a ClusterDesign")
        return delta + _noise(rng, dist, n_items, sd**2), None
    C, size, rho = cluster.n_clusters, cluster.cluster_size, cluster.rho
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    u = rng.normal(0.0, sd * np.sqrt(rho), C)
    e = _noise(rng, dist, C * size, (1.0 - rho) * sd**2)
    d = delta + np.repeat(u, size) + e
    clusters = list(np.repeat([f"c{j:06d}" for j in range(C)], size))
    return d, clusters


def simulate_d_binary(*, delta: float, psi: float, n_items: int, seed: int):
    """Binary singletons (§5): d_i in {-1, 0, +1} with
    P(+1) = (psi + delta)/2, P(-1) = (psi - delta)/2, discordance rate psi."""
    p_pos = (psi + delta) / 2.0
    p_neg = (psi - delta) / 2.0
    if p_pos < 0 or p_neg < 0 or psi > 1:
        raise ValueError("need |delta| <= psi <= 1")
    rng = make_rng(seed)
    u = rng.random(n_items)
    d = np.where(u < p_pos, 1.0, np.where(u < psi, -1.0, 0.0))
    return d, None


def simulate_d_preference(*, theta: float, tie_rate: float, n_items: int, seed: int):
    """Preference singletons (§3): s_i in {0, 0.5, 1} with E[s_i] = theta and
    the given tie rate; returns d = 2s - 1."""
    w = theta - tie_rate / 2.0  # P(win)
    if w < 0 or w + tie_rate > 1:
        raise ValueError("incompatible theta and tie_rate")
    rng = make_rng(seed)
    u = rng.random(n_items)
    s = np.where(u < w, 1.0, np.where(u < w + tie_rate, 0.5, 0.0))
    return 2.0 * s - 1.0, None


def simulate_d_meanzero_skewed(*, kind: str, n_items: int, seed: int,
                               shape: float = 2.0, scale: float = 1.0,
                               r_a: int = 3, r_b: int = 1):
    """Mean-zero but ASYMMETRIC d — outside the §2.2 exchangeability null.

    The two realistic routes to E[d]=0 without symmetry (§11.8 probes):
      kind="shape-diff"      — variants differ in distribution with equal
                               means (d = centered gamma; skewness ~1.4).
      kind="side-unbalanced" — identical skewed score distribution but
                               unbalanced replicate structure (F11):
                               d = mean of r_a draws - mean of r_b draws.
    """
    rng = make_rng(seed)
    mu = shape * scale
    if kind == "shape-diff":
        return rng.gamma(shape, scale, n_items) - mu, None
    if kind == "side-unbalanced":
        a = rng.gamma(shape, scale, (n_items, r_a)).mean(axis=1)
        b = rng.gamma(shape, scale, (n_items, r_b)).mean(axis=1)
        return a - b, None
    raise ValueError(f"unknown kind {kind!r}")


def simulate_score_records(*, n_items: int, delta: float, sigma_b: float,
                           sigma_g: float, sigma_j: float, r: int, m: int,
                           seed: int, m_pattern=None, item_sd: float = 1.0,
                           variant_a: str = "A", variant_b: str = "B"):
    """Judgment-level records for the §1.2 reduction and §8 estimators.

    Model (§8.1): item base t_i ~ N(0, item_sd^2); true difference
    delta_i ~ N(delta, sigma_b^2); response value mu_iv + N(0, sigma_g^2);
    judge call adds N(0, sigma_j^2). ``m_pattern`` (list of ints, cycled
    across responses) overrides m to create unbalanced judging.

    Returns (records, truth) where truth holds the generating parameters.
    """
    rng = make_rng(seed)
    records = []
    for i in range(n_items):
        item = f"i{i:06d}"
        t_i = rng.normal(0.0, item_sd)
        delta_i = rng.normal(delta, sigma_b)
        for v, sign in ((variant_a, 0.5), (variant_b, -0.5)):
            mu = t_i + sign * delta_i
            for j in range(r):
                value = mu + rng.normal(0.0, sigma_g)
                mm = m_pattern[j % len(m_pattern)] if m_pattern else m
                for k in range(mm):
                    records.append({
                        "item_id": item,
                        "source_doc_id": None,
                        "variant_id": v,
                        "response_id": f"{item}-{v}-r{j}",
                        "judgment": value + rng.normal(0.0, sigma_j),
                    })
    truth = {"delta": delta, "sigma2_b": sigma_b**2, "sigma2_g": sigma_g**2,
             "sigma2_j": sigma_j**2}
    return records, truth


def simulate_pairwise_records(*, n_items: int, theta: float, tie_rate: float = 0.0,
                              position_bias: float = 0.0, calls_a_first: int = 1,
                              calls_b_first: int = 1, seed: int,
                              variant_a: str = "A", variant_b: str = "B"):
    """Pairwise judgment records with an injected ADDITIVE position effect.

    Per call, P(prefer A) = (theta - tie_rate/2) + position_bias when A is
    shown first, - position_bias when B is first; P(tie) = tie_rate. The
    §1.2 order-balanced reduction removes the effect exactly; call-level
    pooling does not when the order counts are unequal.

    Returns (records, truth) with truth["theta"] the true win rate.
    """
    w = theta - tie_rate / 2.0
    b = abs(position_bias)
    if w - b < 0 or w + b + tie_rate > 1:
        raise ValueError("theta/tie_rate/position_bias out of range")
    rng = make_rng(seed)
    records = []
    for i in range(n_items):
        item = f"i{i:06d}"
        pair = f"{item}-p0"
        for a_first, ncalls in ((True, calls_a_first), (False, calls_b_first)):
            pa = w + (position_bias if a_first else -position_bias)
            for _ in range(ncalls):
                u = rng.random()
                if u < pa:
                    j = "first" if a_first else "second"
                elif u < pa + tie_rate:
                    j = "tie"
                else:
                    j = "second" if a_first else "first"
                records.append({
                    "item_id": item,
                    "source_doc_id": None,
                    "pair_id": pair,
                    "variant_first": variant_a if a_first else variant_b,
                    "variant_second": variant_b if a_first else variant_a,
                    "judgment": j,
                })
    return records, {"theta": theta}
