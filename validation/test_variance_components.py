"""§11.5 — Variance-component estimators recover known simulation truth.

sigma^2_J (pooled within-response) and sigma^2_B (method of moments, §8.2)
across balanced and unbalanced replicate designs, plus the
truncation-at-zero regime. Bias and RMSE are published; the asserts bind
mean bias within Monte Carlo tolerance of zero.

Two distinct kinds of unbalance are covered. The "unbalanced" regime applies
the same call-count pattern to both variants — side-BALANCED unbalanced
judging, the benign case that leaves the §2.2 symmetry argument intact. The
"side-unbalanced" regime gives the two variants different call counts: the
F11 condition that DECISION D4's item-level mirrored replicate draw (spec
§8.2) is specified never to create, but which a run can still arrive with
(partial failures, historical data). The variance-component estimators must
recover truth there too — their unbiasedness does not depend on the balance
D4 guarantees, only the sign-flip test's exactness does.
"""

from collections import defaultdict

import numpy as np
import pytest

from evalkit.stats import (
    reduce_scores,
    sigma2_judge,
    sigma2_within_item,
    variance_components,
)
from evalkit.stats.simulate import simulate_score_records
from validation._tolerances import publish, rep_seeds

pytestmark = pytest.mark.validation

SIGMA2_J = 0.16
SIGMA2_B = 0.25


def _estimate_once(seed, *, n_items, r, m, m_pattern, sigma_b, w_exact,
                   m_pattern_b=None):
    records, _ = simulate_score_records(
        n_items=n_items, delta=0.2, sigma_b=sigma_b, sigma_g=0.0,
        sigma_j=float(np.sqrt(SIGMA2_J)), r=r, m=m, m_pattern=m_pattern,
        m_pattern_b=m_pattern_b, seed=seed)
    by_response = defaultdict(list)
    for rec in records:
        by_response[rec["response_id"]].append(rec["judgment"])
    jv = sigma2_judge(by_response.values())
    red = reduce_scores(records, "A", "B")
    # w_exact(sigma2_j_hat) computes the per-item within variance for the
    # design in force (§8.1; exact even when m varies across responses).
    vc = variance_components(red.d, w_exact(jv.sigma2_j))
    return jv.sigma2_j, vc.sigma2_b, vc.truncated


def _summarize(name, estimates, truth):
    est = np.asarray(estimates, dtype=float)
    bias = float(est.mean() - truth)
    se = float(est.std(ddof=1) / np.sqrt(est.size))
    rmse = float(np.sqrt(np.mean((est - truth) ** 2)))
    return dict(quantity=name, truth=truth, mean=f"{est.mean():.5f}",
                bias=f"{bias:.5f}", mc_se=f"{se:.5f}", rmse=f"{rmse:.5f}"), bias, se


def test_variance_components_recover_truth(reps):
    regimes = {
        # balanced: r=1, m=2 -> sigma2_W = 2*sigma2_J/2 = sigma2_J
        "balanced-r1-m2": dict(
            n_items=150, r=1, m=2, m_pattern=None, sigma_b=float(np.sqrt(SIGMA2_B)),
            w_exact=lambda s2j: sigma2_within_item(r=1, m=2, sigma2_j=s2j)),
        # unbalanced judging: r=2, calls (1, 3) per response ->
        # per side Var = (1/r^2) * sum_resp s2j/m_resp = s2j*(1 + 1/3)/4; d doubles it
        "unbalanced-r2-m1and3": dict(
            n_items=150, r=2, m=None, m_pattern=[1, 3], sigma_b=float(np.sqrt(SIGMA2_B)),
            w_exact=lambda s2j: 2.0 * s2j * (1.0 + 1.0 / 3.0) / 4.0),
        # SIDE-unbalanced judging (F11): r=1, variant A gets m=2 calls,
        # variant B gets m=1 — the condition D4's mirrored item-level draw
        # must never create (spec §8.2), probed here so recovery is shown
        # not to depend on it. Var_W(d) = s2j*(1/2 + 1/1); sigma2_J pools
        # over variant A's responses only (the only ones with m >= 2).
        "side-unbalanced-r1-mA2-mB1": dict(
            n_items=150, r=1, m=None, m_pattern=[2], m_pattern_b=[1],
            sigma_b=float(np.sqrt(SIGMA2_B)),
            w_exact=lambda s2j: s2j * (1.0 / 2.0 + 1.0)),
    }
    rows, failures = [], []
    for ri, (name, cfg) in enumerate(regimes.items()):
        seeds, = rep_seeds(960_000 + ri, reps, streams=1)
        s2j_est, s2b_est = [], []
        for i in range(reps):
            s2j, s2b, _ = _estimate_once(int(seeds[i]), **cfg)
            s2j_est.append(s2j)
            s2b_est.append(s2b)
        for qname, est, truth in ((f"{name}:sigma2_J", s2j_est, SIGMA2_J),
                                  (f"{name}:sigma2_B", s2b_est, SIGMA2_B)):
            row, bias, se = _summarize(qname, est, truth)
            rows.append(row)
            if abs(bias) > 4 * se + 0.005:
                failures.append(f"{qname}: bias {bias:.5f} exceeds 4*MC-SE {4*se:.5f} + 0.005")
    publish("variance_components", reps, rows)
    assert not failures, "Variance-component estimator drift:\n" + "\n".join(failures)


def test_truncation_regime(reps):
    # sigma2_B = 0: raw moment estimate is mean-zero noise; truncation at 0
    # must trigger often and the truncated mean must stay near the half-normal
    # bound sd(s2_d)/sqrt(2*pi) (§8.2 requires the truncation be reported).
    n_items = 100
    w = SIGMA2_J  # r=1, m=2
    seeds, = rep_seeds(961_000, reps, streams=1)
    est, trunc = [], 0
    for i in range(reps):
        s2j, s2b, truncated = _estimate_once(
            int(seeds[i]), n_items=n_items, r=1, m=2, m_pattern=None, sigma_b=0.0,
            w_exact=lambda s2j: sigma2_within_item(r=1, m=2, sigma2_j=s2j))
        est.append(s2b)
        trunc += truncated
    mean_est = float(np.mean(est))
    trunc_frac = trunc / reps
    sd_s2 = np.sqrt(2.0 / (n_items - 1)) * w  # Var(s^2) ~ 2 sigma^4/(n-1)
    half_normal_mean = sd_s2 / np.sqrt(2.0 * np.pi)
    publish("variance_truncation", reps, [dict(
        mean_sigma2_b=f"{mean_est:.5f}", truth=0.0,
        half_normal_bound=f"{half_normal_mean:.5f}",
        truncated_fraction=f"{trunc_frac:.4f}")])
    assert mean_est <= 3.0 * half_normal_mean, (
        f"truncated sigma2_B mean {mean_est:.5f} far above the half-normal "
        f"prediction {half_normal_mean:.5f}: estimator biased")
    assert trunc_frac >= 0.25, (
        f"truncation triggered in only {trunc_frac:.2%} of runs under "
        f"sigma2_B=0; expected ~50%")
