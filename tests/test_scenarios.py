"""Unit tests for the scenario front-ends and the §10 report block."""

import numpy as np
import pytest

from evalkit.stats import (
    JUDGE_RELATIVITY_CAVEAT,
    compare_binary,
    compare_pairwise,
    compare_scalar,
)
from evalkit.stats.simulate import simulate_pairwise_records, simulate_score_records


def _binary_records():
    recs = []
    rng = np.random.Generator(np.random.PCG64(4))
    for i in range(40):
        item = f"i{i:03d}"
        a = 1.0 if rng.random() < 0.7 else 0.0
        b = 1.0 if rng.random() < 0.5 else 0.0
        recs.append({"item_id": item, "source_doc_id": None, "variant_id": "A",
                     "response_id": f"{item}-a", "judgment": a})
        recs.append({"item_id": item, "source_doc_id": None, "variant_id": "B",
                     "response_id": f"{item}-b", "judgment": b})
    return recs


def test_scalar_report_block():
    records, _ = simulate_score_records(n_items=30, delta=0.5, sigma_b=0.3,
                                        sigma_g=0.0, sigma_j=0.4, r=1, m=2, seed=8)
    rep = compare_scalar(records, "A", "B", seed=9, n_boot=500, n_perm=500)
    assert rep.scenario == "B-scalar"
    assert rep.ci_low <= rep.estimate <= rep.ci_high
    assert rep.n_items == 30 and rep.n_clusters == 30
    assert np.isfinite(rep.mde_80) and rep.mde_80 > 0
    assert JUDGE_RELATIVITY_CAVEAT in rep.caveats
    assert "score_distribution" in rep.extras
    d = rep.to_dict()
    assert "engine" not in d and d["estimate"] == rep.estimate


def test_binary_concordance_and_mcnemar_crosscheck():
    rep = compare_binary(_binary_records(), "A", "B", seed=10, n_boot=500, n_perm=500)
    conc = rep.extras["concordance"]
    assert sum(conc.values()) == rep.n_items
    assert conc["n_pass_fail"] - conc["n_fail_pass"] == pytest.approx(rep.estimate * rep.n_items)
    assert 0.0 <= rep.extras["mcnemar_p"] <= 1.0


def test_binary_rejects_non_binary():
    recs = _binary_records()
    recs[0]["judgment"] = 0.5
    with pytest.raises(ValueError, match="binary"):
        compare_binary(recs, "A", "B", seed=1)


def test_pairwise_theta_mapping():
    records, truth = simulate_pairwise_records(n_items=40, theta=0.7, tie_rate=0.1,
                                               calls_a_first=1, calls_b_first=1, seed=12)
    rep = compare_pairwise(records, "A", "B", seed=13, n_boot=500, n_perm=500)
    assert rep.scenario == "A-pairwise"
    eng = rep.engine
    assert rep.estimate == pytest.approx((1 + eng.estimate) / 2)
    # C = 40 < 50: analytic is primary (DECISION D9, §2.1/§2.3).
    ci_report = rep.extras["ci_report"]
    assert ci_report["primary_method"] == "analytic"
    assert rep.ci_low == pytest.approx((1 + eng.analytic.ci_low) / 2)
    assert rep.ci_high == pytest.approx((1 + eng.analytic.ci_high) / 2)
    assert ci_report["bootstrap_ci"] == pytest.approx(((1 + eng.ci_low) / 2, (1 + eng.ci_high) / 2))
    assert 0.0 <= rep.estimate <= 1.0
    assert rep.extras["counterbalanced"]
    assert 0.0 <= rep.extras["tie_rate"] <= 1.0


def test_ci_primary_switches_with_cluster_count():
    # C = 200 >= 50: bootstrap is primary; the analytic interval is still
    # carried alongside it.
    records, _ = simulate_score_records(n_items=200, delta=0.3, sigma_b=0.3,
                                        sigma_g=0.0, sigma_j=0.4, r=1, m=2, seed=20)
    rep = compare_scalar(records, "A", "B", seed=21, n_boot=500, n_perm=500)
    eng = rep.engine
    ci_report = rep.extras["ci_report"]
    assert ci_report["primary_method"] == "bootstrap"
    assert rep.ci_low == pytest.approx(eng.ci_low)
    assert rep.ci_high == pytest.approx(eng.ci_high)
    assert ci_report["analytic_ci"] == pytest.approx((eng.analytic.ci_low, eng.analytic.ci_high))
    assert "coverage_disclosure" in ci_report

    # C = 30 < 50: analytic is primary.
    records2, _ = simulate_score_records(n_items=30, delta=0.3, sigma_b=0.3,
                                         sigma_g=0.0, sigma_j=0.4, r=1, m=2, seed=22)
    rep2 = compare_scalar(records2, "A", "B", seed=23, n_boot=500, n_perm=500)
    eng2 = rep2.engine
    ci_report2 = rep2.extras["ci_report"]
    assert ci_report2["primary_method"] == "analytic"
    assert rep2.ci_low == pytest.approx(eng2.analytic.ci_low)
    assert rep2.ci_high == pytest.approx(eng2.analytic.ci_high)


def test_pairwise_uncounterbalanced_flagged():
    records, _ = simulate_pairwise_records(n_items=25, theta=0.6, calls_a_first=1,
                                           calls_b_first=0, seed=14)
    rep = compare_pairwise(records, "A", "B", seed=15, n_boot=200, n_perm=200)
    assert not rep.extras["counterbalanced"]
    assert any("F1" in w for w in rep.warnings)


def test_one_sided_skew_caveat_f12():
    rng = np.random.Generator(np.random.PCG64(30))
    recs = []
    for i in range(40):
        item = f"i{i:03d}"
        recs.append({"item_id": item, "source_doc_id": None, "variant_id": "A",
                     "response_id": f"{item}-a", "judgment": float(rng.gamma(0.4, 2.0))})
        recs.append({"item_id": item, "source_doc_id": None, "variant_id": "B",
                     "response_id": f"{item}-b", "judgment": 0.5})
    rep = compare_scalar(recs, "A", "B", seed=16, n_boot=200, n_perm=200,
                         alternative="greater")
    assert any("F12" in w for w in rep.warnings)
