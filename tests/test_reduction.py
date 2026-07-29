"""Unit tests for the §1.2 reduction: equal response weighting, order-balanced
pairwise scoring, exclusion accounting, and the F11 side-balance check."""

import numpy as np
import pytest

from evalkit.stats import reduce_pairwise, reduce_scores


def _score(item, variant, resp, y, doc=None):
    return {"item_id": item, "source_doc_id": doc, "variant_id": variant,
            "response_id": resp, "judgment": y}


def _pair(item, vf, vs, judgment, pair="p0", doc=None):
    return {"item_id": item, "source_doc_id": doc, "pair_id": pair,
            "variant_first": vf, "variant_second": vs, "judgment": judgment}


def test_scalar_reduction_equal_response_weighting():
    # Response a1 got 2 judge calls, a2 got 1; responses weigh equally (§1.2):
    # x_A = mean(mean(4,5), 3) = 3.75, NOT (4+5+3)/3 = 4.
    recs = [
        _score("i1", "A", "a1", 4.0), _score("i1", "A", "a1", 5.0),
        _score("i1", "A", "a2", 3.0),
        _score("i1", "B", "b1", 2.0),
    ]
    red = reduce_scores(recs, "A", "B")
    assert red.x_a[0] == pytest.approx(3.75)
    assert red.d[0] == pytest.approx(1.75)
    # unbalanced judging across sides triggers the F11 structure check
    assert red.side_unbalanced_items == ("i1",)
    assert any("F11" in w for w in red.warnings)


def test_scalar_reduction_side_balanced_no_warning():
    recs = [
        _score("i1", "A", "a1", 4.0), _score("i1", "A", "a1", 5.0),
        _score("i1", "B", "b1", 2.0), _score("i1", "B", "b1", 3.0),
    ]
    red = reduce_scores(recs, "A", "B")
    assert red.side_unbalanced_items == ()
    assert not any("F11" in w for w in red.warnings)


def test_missing_side_excluded_and_counted():
    recs = [
        _score("i1", "A", "a1", 4.0), _score("i1", "B", "b1", 2.0),
        _score("i2", "A", "a2", 5.0),  # no B side
        _score("i3", "C", "c1", 1.0),  # unknown variant -> ignored
    ]
    red = reduce_scores(recs, "A", "B")
    assert red.n_items == 1
    assert red.excluded_item_ids == ("i2",)
    assert red.n_records_ignored == 1
    assert any("excluded" in w for w in red.warnings)


def test_cluster_assignment():
    recs = [
        _score("i1", "A", "a1", 1.0, doc="doc7"), _score("i1", "B", "b1", 0.0, doc="doc7"),
        _score("i2", "A", "a2", 1.0), _score("i2", "B", "b2", 0.0),
    ]
    red = reduce_scores(recs, "A", "B")
    assert red.clusters[0] == "doc7"
    assert red.clusters[1].startswith("__singleton__:")


def test_conflicting_source_doc_raises():
    recs = [_score("i1", "A", "a1", 1.0, doc="d1"), _score("i1", "B", "b1", 0.0, doc="d2")]
    with pytest.raises(ValueError, match="conflicting"):
        reduce_scores(recs, "A", "B")


def test_pairwise_order_weighting_identity():
    # §1.2 step 3: average within order first, then across orders — the pair
    # score is (mean_AF + mean_BF)/2 regardless of unequal call counts.
    recs = [
        _pair("i1", "A", "B", "first"), _pair("i1", "A", "B", "first"),
        _pair("i1", "A", "B", "second"), _pair("i1", "A", "B", "first"),  # AF mean 0.75
        _pair("i1", "B", "A", "first"),                                    # BF mean: B preferred -> y=0
    ]
    red = reduce_pairwise(recs, "A", "B")
    assert red.s[0] == pytest.approx((0.75 + 0.0) / 2)
    assert red.counterbalanced


def test_pairwise_tie_and_recode():
    # B shown first, judge says "first" -> B preferred -> y = 0 for A.
    recs = [
        _pair("i1", "B", "A", "first"), _pair("i1", "A", "B", "tie"),
    ]
    red = reduce_pairwise(recs, "A", "B")
    # AF order mean = 0.5 (tie), BF order mean = 0.0 -> s = 0.25
    assert red.s[0] == pytest.approx(0.25)
    assert red.tie_rate == pytest.approx(0.5)


def test_pairwise_single_order_flagged():
    recs = [_pair("i1", "A", "B", "first"), _pair("i2", "A", "B", "first"),
            _pair("i2", "B", "A", "second")]
    red = reduce_pairwise(recs, "A", "B")
    assert not red.counterbalanced
    assert red.n_pairs_single_order == 1
    assert any("F1" in w for w in red.warnings)
    assert red.d[1] == pytest.approx(1.0)  # i2: both orders prefer A


def test_pairwise_d_mapping():
    recs = [_pair("i1", "A", "B", "first"), _pair("i1", "B", "A", "second")]
    red = reduce_pairwise(recs, "A", "B")
    assert red.s[0] == pytest.approx(1.0)
    assert red.d[0] == pytest.approx(1.0)


def test_prominent_f11_warning_when_skewed():
    # Unbalanced sides on every item plus strongly skewed d -> PROMINENT warning.
    rng = np.random.Generator(np.random.PCG64(3))
    recs = []
    for i in range(40):
        item = f"i{i:03d}"
        g = float(rng.gamma(0.3, 2.0))  # heavy right skew
        recs.append(_score(item, "A", f"{item}-a1", g))
        recs.append(_score(item, "A", f"{item}-a2", g))
        recs.append(_score(item, "B", f"{item}-b1", 0.5))
    red = reduce_scores(recs, "A", "B")
    assert len(red.side_unbalanced_items) == 40
    assert abs(red.d_skewness) > 1
    assert any(w.startswith("PROMINENT") for w in red.warnings)
