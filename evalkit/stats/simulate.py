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
    "simulate_position_bias_records",
    "simulate_score_records",
    "simulate_self_preference_records",
    "simulate_verbosity_controlled_records",
    "simulate_verbosity_observational_records",
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


# ---------------------------------------------------------------------------
# Bias-injection simulators (spec §14.5, validation item 9). All random
# effects are uniform (bounded) so preference probabilities stay in range
# without clipping — a clip would silently change the injected truth.


def _check_prob_range(p_min: float, p_max: float, tie_rate: float) -> None:
    if p_min < 0.0 or p_max + tie_rate > 1.0:
        raise ValueError(
            f"preference probabilities out of range: [{p_min:.3f}, {p_max:.3f}] "
            f"with tie_rate {tie_rate}; shrink the effects or biases"
        )


def _draw_pref_first(rng, p_first: float, tie_rate: float) -> str:
    u = rng.random()
    if u < p_first:
        return "first"
    if u < p_first + tie_rate:
        return "tie"
    return "second"


def simulate_position_bias_records(*, n_clusters: int, cluster_size: int,
                                   beta: float, beta_effect_scale: float = 0.05,
                                   content_effect_scale: float = 0.12,
                                   tie_rate: float = 0.1, calls_per_order: int = 1,
                                   seed: int, variant_a: str = "A", variant_b: str = "B"):
    """Counterbalanced pairwise records with an injected additive position bias.

    Per item i in cluster c: content preference w_i = u_c + e_i and position
    effect beta_i = beta + v_c + f_i (uniform effects at cluster and item
    level, so the item-level position scores are cluster-correlated and the
    C-cluster coverage cells actually exercise clustering);
    P(first preferred | order) = (1 - tie_rate)/2 +- w_i + beta_i.
    Truth: beta_pos = E[beta_i] = beta.
    """
    base = (1.0 - tie_rate) / 2.0
    span = 2.0 * content_effect_scale + abs(beta) + 2.0 * beta_effect_scale
    _check_prob_range(base - span, base + span, tie_rate)
    rng = make_rng(seed)
    records = []
    for c in range(n_clusters):
        doc = f"doc{c:05d}"
        u = rng.uniform(-content_effect_scale, content_effect_scale)
        v = rng.uniform(-beta_effect_scale, beta_effect_scale)
        for k in range(cluster_size):
            w_i = u + rng.uniform(-content_effect_scale, content_effect_scale)
            beta_i = beta + v + rng.uniform(-beta_effect_scale, beta_effect_scale)
            item = f"i{c:05d}-{k:03d}"
            for first, second, w_signed in ((variant_a, variant_b, w_i),
                                            (variant_b, variant_a, -w_i)):
                p_first = base + w_signed + beta_i
                for _ in range(calls_per_order):
                    records.append({
                        "item_id": item, "source_doc_id": doc,
                        "pair_id": f"{item}-p0",
                        "variant_first": first, "variant_second": second,
                        "response_id_first": f"{item}-{first}",
                        "response_id_second": f"{item}-{second}",
                        "tokens_first": 100, "tokens_second": 100,
                        "judge_model": "judge-sim-v1",
                        "judgment": _draw_pref_first(rng, p_first, tie_rate),
                    })
    return records, {"beta": beta}


def _emit_longer_pair(rng, records, *, item, doc, pair, long_variant, short_variant,
                      tokens_long, tokens_short, p_long, position_bias, tie_rate,
                      judge_model):
    """Both counterbalanced orders of one length-discordant pair.

    ``p_long`` is the order-free probability the longer response is
    preferred; an additive position effect shifts it by +-position_bias
    depending on which side is shown first (the order-balanced reduction
    must cancel it)."""
    for longer_first in (True, False):
        p = p_long + (position_bias if longer_first else -position_bias)
        u = rng.random()
        if u < p:
            judgment = "first" if longer_first else "second"
        elif u < p + tie_rate:
            judgment = "tie"
        else:
            judgment = "second" if longer_first else "first"
        vf, vs = (long_variant, short_variant) if longer_first else (short_variant, long_variant)
        tf, ts = (tokens_long, tokens_short) if longer_first else (tokens_short, tokens_long)
        records.append({
            "item_id": item, "source_doc_id": doc, "pair_id": pair,
            "variant_first": vf, "variant_second": vs,
            "response_id_first": f"{item}-{vf}", "response_id_second": f"{item}-{vs}",
            "tokens_first": tf, "tokens_second": ts,
            "judge_model": judge_model,
            "judgment": judgment,
        })


def simulate_verbosity_observational_records(*, n_clusters: int, cluster_size: int,
                                             gamma: float, quality_confound: float,
                                             position_bias: float = 0.04,
                                             effect_scale: float = 0.1,
                                             tie_rate: float = 0.1, seed: int,
                                             tokens_long: int = 300, tokens_short: int = 150,
                                             variant_a: str = "A", variant_b: str = "B"):
    """Observational pairs where the judge's true length effect ``gamma`` and
    a genuine quality-length component ``quality_confound`` both push
    preference toward the longer response. Only their SUM (the association)
    is recoverable — that is the point (F14, §14.3).

    The longer response's variant is randomized per item. Truth:
    association = 1/2 + gamma + quality_confound; gamma itself is
    deliberately not recoverable from these records.
    """
    base = (1.0 - tie_rate) / 2.0
    lift = gamma + quality_confound
    span = 2.0 * effect_scale + abs(position_bias)
    _check_prob_range(base + min(lift, 0.0) - span, base + max(lift, 0.0) + span, tie_rate)
    rng = make_rng(seed)
    records = []
    for c in range(n_clusters):
        doc = f"doc{c:05d}"
        u = rng.uniform(-effect_scale, effect_scale)
        for k in range(cluster_size):
            w_i = u + rng.uniform(-effect_scale, effect_scale)
            item = f"i{c:05d}-{k:03d}"
            longer_is_a = rng.random() < 0.5
            long_v, short_v = (variant_a, variant_b) if longer_is_a else (variant_b, variant_a)
            _emit_longer_pair(rng, records, item=item, doc=doc, pair=f"{item}-p0",
                              long_variant=long_v, short_variant=short_v,
                              tokens_long=tokens_long, tokens_short=tokens_short,
                              p_long=base + lift + w_i, position_bias=position_bias,
                              tie_rate=tie_rate, judge_model="judge-sim-v1")
    return records, {"gamma": gamma, "quality_confound": quality_confound,
                     "association": 0.5 + gamma + quality_confound}


def simulate_verbosity_controlled_records(*, n_clusters: int, cluster_size: int,
                                          gamma: float, artifact_padded: float = 0.0,
                                          artifact_condensed: float = 0.0,
                                          position_bias: float = 0.04,
                                          effect_scale: float = 0.1,
                                          tie_rate: float = 0.1, seed: int,
                                          original_variant: str = "orig",
                                          padded_variant: str = "padded",
                                          condensed_variant: str = "condensed",
                                          tokens_original: int = 150,
                                          tokens_padded: int = 300,
                                          tokens_condensed: int = 80):
    """Content-matched manipulation arms (§14.3): original vs padded (longer
    = the manipulated rewrite) with preference-for-longer lift
    gamma - artifact_padded, and original vs condensed (longer = the
    original) with lift gamma + artifact_condensed. Artifact detection
    pushes the two arms apart in opposite directions — exactly what the
    manipulation check measures.

    Truth: gamma_padded = gamma - artifact_padded,
    gamma_condensed = gamma + artifact_condensed,
    check = gamma_padded - gamma_condensed = -(artifact_padded + artifact_condensed),
    gamma_pooled = gamma + (artifact_condensed - artifact_padded)/2.
    """
    if not (tokens_padded > tokens_original > tokens_condensed):
        raise ValueError("need tokens_padded > tokens_original > tokens_condensed")
    base = (1.0 - tie_rate) / 2.0
    lifts = {"pad": gamma - artifact_padded, "cond": gamma + artifact_condensed}
    span = 2.0 * effect_scale + abs(position_bias)
    lo = min(lifts.values())
    hi = max(lifts.values())
    _check_prob_range(base + min(lo, 0.0) - span, base + max(hi, 0.0) + span, tie_rate)
    rng = make_rng(seed)
    records = []
    for c in range(n_clusters):
        doc = f"doc{c:05d}"
        u = rng.uniform(-effect_scale, effect_scale)
        for k in range(cluster_size):
            item = f"i{c:05d}-{k:03d}"
            for arm, long_v, short_v, t_long, t_short in (
                ("pad", padded_variant, original_variant, tokens_padded, tokens_original),
                ("cond", original_variant, condensed_variant, tokens_original, tokens_condensed),
            ):
                w = u + rng.uniform(-effect_scale, effect_scale)
                _emit_longer_pair(rng, records, item=item, doc=doc,
                                  pair=f"{item}-{arm}", long_variant=long_v,
                                  short_variant=short_v, tokens_long=t_long,
                                  tokens_short=t_short,
                                  p_long=base + lifts[arm] + w,
                                  position_bias=position_bias,
                                  tie_rate=tie_rate, judge_model="judge-sim-v1")
    return records, {
        "gamma": gamma,
        "gamma_padded": lifts["pad"],
        "gamma_condensed": lifts["cond"],
        "check": lifts["pad"] - lifts["cond"],
        "gamma_pooled": (lifts["pad"] + lifts["cond"]) / 2.0,
    }


def simulate_self_preference_records(*, n_clusters: int, cluster_size: int,
                                     sigma_self: float, quality_delta: float = 0.05,
                                     sigma_effect_scale: float = 0.05,
                                     content_effect_scale: float = 0.08,
                                     position_bias: float = 0.04,
                                     tie_rate: float = 0.1, seed: int,
                                     self_judge: str = "judge-self",
                                     other_judges: tuple = ("judge-k1",),
                                     self_variant: str = "self",
                                     other_variant: str = "other"):
    """Multi-judge records on identical pairs (§14.4). Per item: genuine
    preference for the self variant w_i = quality_delta + u_c + e_i, shared
    by every judge; the self judge adds sigma_i = sigma_self + v_c + f_i
    (cluster-correlated, so the coverage cells exercise clustering).

    Truth: sigma_self. The naive theta_self^(J) - 1/2 recovers
    quality_delta + sigma_self instead — published, not asserted equal.
    """
    if not other_judges:
        raise ValueError("at least one other judge required")
    base = (1.0 - tie_rate) / 2.0
    wmax = abs(quality_delta) + 2.0 * content_effect_scale
    bmax = abs(sigma_self) + 2.0 * sigma_effect_scale
    span = wmax + bmax + abs(position_bias)
    _check_prob_range(base - span, base + span, tie_rate)
    rng = make_rng(seed)
    judges = (self_judge,) + tuple(other_judges)
    records = []
    for c in range(n_clusters):
        doc = f"doc{c:05d}"
        u = rng.uniform(-content_effect_scale, content_effect_scale)
        v = rng.uniform(-sigma_effect_scale, sigma_effect_scale)
        for k in range(cluster_size):
            w_i = quality_delta + u + rng.uniform(-content_effect_scale, content_effect_scale)
            sigma_i = sigma_self + v + rng.uniform(-sigma_effect_scale, sigma_effect_scale)
            item = f"i{c:05d}-{k:03d}"
            for judge in judges:
                bias = sigma_i if judge == self_judge else 0.0
                for first, second, signed in ((self_variant, other_variant, w_i + bias),
                                              (other_variant, self_variant, -(w_i + bias))):
                    p_first = base + signed + position_bias
                    records.append({
                        "item_id": item, "source_doc_id": doc,
                        "pair_id": f"{item}-p0",
                        "variant_first": first, "variant_second": second,
                        "response_id_first": f"{item}-{first}",
                        "response_id_second": f"{item}-{second}",
                        "tokens_first": 150, "tokens_second": 150,
                        "judge_model": judge,
                        "judgment": _draw_pref_first(rng, p_first, tie_rate),
                    })
    return records, {"sigma_self": sigma_self, "quality_delta": quality_delta,
                     "naive_theta_excess": quality_delta + sigma_self}
