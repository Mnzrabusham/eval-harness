"""Unit tests for judge/: prompts, strict parsing, structural
counterbalancing, and data-model constraints. Statistical correctness is
not at stake here (nothing stochastic); these check the record contract.
"""

import pytest

from evalkit.judge import (
    PairwiseJudgment,
    ResponseJudgment,
    ResponseSide,
    RunConfig,
    VerdictParseError,
    build_binary_prompt,
    build_scalar_prompt,
    counterbalanced_pairwise_tasks,
    pairwise_judgment_from_task,
    parse_binary_verdict,
    parse_pairwise_verdict,
    parse_scalar_verdict,
)

SIDE_A = ResponseSide(variant_id="variant-alpha", response_id="resp-a1",
                      text="the alpha answer", tokens=12)
SIDE_B = ResponseSide(variant_id="variant-beta", response_id="resp-b1",
                      text="the beta answer", tokens=20)


def _tasks(**kw):
    return counterbalanced_pairwise_tasks(
        item_id="item-1", task_text="What is X?", side_a=SIDE_A, side_b=SIDE_B, **kw)


def test_counterbalancing_is_structural():
    t1, t2 = _tasks()
    # Both orders, same pair_id, presentation roles swapped.
    assert t1.pair_id == t2.pair_id
    assert (t1.first.response_id, t1.second.response_id) == ("resp-a1", "resp-b1")
    assert (t2.first.response_id, t2.second.response_id) == ("resp-b1", "resp-a1")
    # pair_id is order-invariant by construction (unordered response pair).
    ta, tb = counterbalanced_pairwise_tasks(
        item_id="item-1", task_text="What is X?", side_a=SIDE_B, side_b=SIDE_A)
    assert ta.pair_id == t1.pair_id


def test_pairwise_prompt_never_names_variants():
    for task in _tasks():
        assert "the alpha answer" in task.prompt
        assert "the beta answer" in task.prompt
        assert "variant-alpha" not in task.prompt
        assert "variant-beta" not in task.prompt
        assert "VERDICT: TIE" in task.prompt
    t1, _ = _tasks(ties_allowed=False)
    assert "VERDICT: TIE" not in t1.prompt


def test_parse_pairwise_verdict():
    assert parse_pairwise_verdict("VERDICT: FIRST") == "first"
    assert parse_pairwise_verdict("reasoning...\nverdict: second") == "second"
    assert parse_pairwise_verdict("VERDICT: TIE") == "tie"
    with pytest.raises(VerdictParseError):
        parse_pairwise_verdict("I prefer the first one.")  # no VERDICT line
    with pytest.raises(VerdictParseError):
        parse_pairwise_verdict("VERDICT: FIRST\nVERDICT: SECOND")  # conflict
    with pytest.raises(VerdictParseError):
        parse_pairwise_verdict("VERDICT: TIE", ties_allowed=False)


def test_parse_scalar_and_binary():
    assert parse_scalar_verdict("SCORE: 4", scale_min=1, scale_max=5) == 4.0
    assert parse_scalar_verdict("SCORE: 4.5", scale_min=1, scale_max=5) == 4.5
    assert parse_scalar_verdict("SCORE: 4\nSCORE: 4.0", scale_min=1, scale_max=5) == 4.0
    with pytest.raises(VerdictParseError):
        parse_scalar_verdict("SCORE: 6", scale_min=1, scale_max=5)  # not clamped
    with pytest.raises(VerdictParseError):
        parse_scalar_verdict("a fine answer", scale_min=1, scale_max=5)
    with pytest.raises(VerdictParseError):
        parse_scalar_verdict("SCORE: 3\nSCORE: 4", scale_min=1, scale_max=5)
    assert parse_binary_verdict("VERDICT: PASS") == 1
    assert parse_binary_verdict("verdict: fail") == 0
    with pytest.raises(VerdictParseError):
        parse_binary_verdict("VERDICT: PASS\nVERDICT: FAIL")


def test_record_stays_in_presentation_terms():
    _, t2 = _tasks()  # B shown first
    rec = pairwise_judgment_from_task(
        t2, "VERDICT: FIRST", run_id="run-1", judge_model="judge-2026-01-15",
        judge_config_id="jc-1", judge_call_id="call-1",
        created_at="2026-08-02T00:00:00Z")
    assert rec.judgment == "first"  # presentation term, even though B won
    assert rec.variant_first == "variant-beta"
    assert rec.variant_second == "variant-alpha"
    assert (rec.tokens_first, rec.tokens_second) == (20, 12)
    assert rec.pair_id == t2.pair_id


def test_floating_judge_model_alias_rejected():
    t1, _ = _tasks()
    with pytest.raises(ValueError, match="latest"):
        pairwise_judgment_from_task(
            t1, "VERDICT: FIRST", run_id="r", judge_model="judge-latest",
            judge_config_id="jc", judge_call_id="c", created_at="2026-08-02")
    with pytest.raises(ValueError, match="latest"):
        ResponseJudgment(run_id="r", item_id="i", source_doc_id=None,
                         variant_id="v", response_id="resp", gen_seed=0,
                         response_tokens=10, judge_model="judge-latest",
                         judge_config_id="jc", judge_call_id="c",
                         judgment=1.0, created_at="2026-08-02")


def test_pairwise_judgment_verdict_constraint():
    with pytest.raises(ValueError, match="presentation terms"):
        PairwiseJudgment(run_id="r", item_id="i", source_doc_id=None,
                         pair_id="p", variant_first="a", variant_second="b",
                         response_id_first="r1", response_id_second="r2",
                         gen_seed_first=0, gen_seed_second=0,
                         tokens_first=1, tokens_second=2,
                         judge_model="judge-2026-01-15", judge_config_id="jc",
                         judge_call_id="c", judgment="variant-a",
                         created_at="2026-08-02")


def test_run_config_constraints():
    with pytest.raises(ValueError):
        RunConfig(run_id="r", judgment_type="ranked", created_at="2026-08-02")
    with pytest.raises(ValueError):
        RunConfig(run_id="r", judgment_type="scalar", created_at="2026-08-02")
    cfg = RunConfig(run_id="r", judgment_type="scalar", created_at="2026-08-02",
                    scale_min=1, scale_max=5)
    assert cfg.ties_allowed


def test_scalar_and_binary_prompts_carry_scale_and_criteria():
    p = build_scalar_prompt("task?", "answer.", rubric="be right",
                            scale_min=1, scale_max=5)
    assert "1" in p and "5" in p and "be right" in p and "SCORE:" in p
    b = build_binary_prompt("task?", "answer.", criteria="must compile")
    assert "must compile" in b and "VERDICT: PASS" in b
