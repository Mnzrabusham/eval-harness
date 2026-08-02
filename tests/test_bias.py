"""Unit tests for bias/: deterministic identities, exclusion rules, and the
F14 naming rule. Statistical correctness (recovery, coverage) is
established by validation/test_bias_estimators.py, not here (CLAUDE.md
rule 1); these check the reductions' exact algebra on constructed records.
"""

import pytest

import evalkit.bias as bias_pkg
from evalkit.bias import (
    position_bias,
    self_preference_bias,
    verbosity_association,
    verbosity_bias_controlled,
)

FAST = dict(n_boot=200, n_perm=99)


def _pw(item, pair, vf, vs, judgment, doc=None, tf=100, ts=100, judge="judge-v1"):
    return {"item_id": item, "source_doc_id": doc, "pair_id": pair,
            "variant_first": vf, "variant_second": vs,
            "tokens_first": tf, "tokens_second": ts,
            "judge_model": judge, "judgment": judgment}


def _both_orders(item, pair, judgments, **kw):
    """One record per order; ``judgments`` = (j_when_A_first, j_when_B_first)."""
    return [_pw(item, pair, "A", "B", judgments[0], **kw),
            _pw(item, pair, "B", "A", judgments[1], **kw)]


def test_position_bias_always_first_is_half():
    recs = []
    for k in range(6):
        recs += _both_orders(f"i{k}", f"p{k}", ("first", "first"))
    rep = position_bias(recs, seed=1, **FAST)
    assert rep.estimate == pytest.approx(0.5)
    assert rep.quantity == "position-bias"
    assert rep.identified


def test_position_bias_consistent_content_preference_is_zero():
    # Judge always prefers variant A regardless of position: g = 1/2 exactly.
    recs = []
    for k in range(6):
        recs += _both_orders(f"i{k}", f"p{k}", ("first", "second"))
    rep = position_bias(recs, seed=1, **FAST)
    assert rep.estimate == pytest.approx(0.0)


def test_position_bias_requires_both_orders():
    recs = [_pw(f"i{k}", f"p{k}", "A", "B", "first") for k in range(6)]
    with pytest.raises(ValueError, match="both presentation orders"):
        position_bias(recs, seed=1, **FAST)


def test_verbosity_association_longer_always_wins():
    recs = []
    for k in range(6):
        # First order: longer shown first, preferred; second order: longer
        # shown second, still preferred.
        recs += _both_orders(f"i{k}", f"p{k}", ("first", "second"),
                             tf=300, ts=100)
        recs[-1]["tokens_first"], recs[-1]["tokens_second"] = 100, 300
    rep = verbosity_association(recs, seed=1, **FAST)
    assert rep.estimate == pytest.approx(1.0)
    assert not rep.identified
    assert "association" in rep.quantity
    assert any("ASSOCIATION" in d for d in rep.disclosures)


def test_verbosity_association_needs_length_discordant_pairs():
    recs = []
    for k in range(4):
        recs += _both_orders(f"i{k}", f"p{k}", ("first", "first"))  # equal tokens
    with pytest.raises(ValueError, match="length-discordant"):
        verbosity_association(recs, seed=1, **FAST)


def test_f14_naming_rule():
    # No public symbol containing "verbosity_bias" may return the
    # observational quantity; the only one allowed is the controlled-study
    # estimator (spec §14.3 naming rule).
    offenders = [n for n in bias_pkg.__all__
                 if "verbosity_bias" in n and n != "verbosity_bias_controlled"]
    assert offenders == []
    assert "verbosity_association" in bias_pkg.__all__


def _controlled_recs(n_items=6):
    """Padded arm: padded (longer) always preferred -> gamma_padded = +0.5.
    Condensed arm: condensed (shorter) always preferred -> gamma_condensed
    = -0.5. Check = +1.0: maximal manipulation detection."""
    recs = []
    for k in range(n_items):
        item = f"i{k}"
        recs += [
            _pw(item, f"{item}-pad", "padded", "orig", "first", tf=300, ts=150),
            _pw(item, f"{item}-pad", "orig", "padded", "second", tf=150, ts=300),
            _pw(item, f"{item}-cond", "orig", "condensed", "second", tf=150, ts=80),
            _pw(item, f"{item}-cond", "condensed", "orig", "first", tf=80, ts=150),
        ]
    return recs


def test_verbosity_controlled_arms_and_manipulation_check():
    rep = verbosity_bias_controlled(
        _controlled_recs(), original_variant="orig", padded_variant="padded",
        condensed_variant="condensed", seed=7, **FAST)
    assert rep.gamma_padded.estimate == pytest.approx(0.5)
    assert rep.gamma_condensed.estimate == pytest.approx(-0.5)
    assert rep.manipulation_check.estimate == pytest.approx(1.0)
    assert rep.gamma_pooled.estimate == pytest.approx(0.0)
    assert any("manipulation check rejects" in w for w in rep.warnings)


def test_verbosity_controlled_single_arm_is_unchecked():
    recs = [r for r in _controlled_recs() if "condensed" not in
            (r["variant_first"], r["variant_second"])]
    rep = verbosity_bias_controlled(recs, original_variant="orig",
                                    padded_variant="padded", seed=7, **FAST)
    assert rep.gamma_condensed is None
    assert rep.manipulation_check is None
    assert any("UNCHECKED" in w for w in rep.warnings)


def _self_pref_recs():
    """Self judge always prefers self (s = 1); panel judge always ties
    (s = 1/2): sigma_self = 1/2 exactly."""
    recs = []
    for k in range(5):
        item, pair = f"i{k}", f"i{k}-p"
        for judge, style in (("judge-self", "self"), ("judge-k", "tie")):
            for vf, vs in (("self", "other"), ("other", "self")):
                j = "tie" if style == "tie" else ("first" if vf == "self" else "second")
                recs.append(_pw(item, pair, vf, vs, j, judge=judge))
    return recs


def test_self_preference_cross_judge_contrast():
    rep = self_preference_bias(_self_pref_recs(), self_judge="judge-self",
                               self_variant="self", other_variant="other",
                               seed=3, **FAST)
    assert rep.estimate == pytest.approx(0.5)
    assert rep.diagnostics["naive_self_theta_minus_half"] == pytest.approx(0.5)
    assert "NOT a bias estimate" in rep.diagnostics["naive_label"]
    assert any("panel neutrality rests entirely" in w for w in rep.warnings)


def test_self_preference_requires_a_panel():
    recs = [r for r in _self_pref_recs() if r["judge_model"] == "judge-self"]
    with pytest.raises(ValueError, match="at least one judge other than"):
        self_preference_bias(recs, self_judge="judge-self", self_variant="self",
                             other_variant="other", seed=3, **FAST)
