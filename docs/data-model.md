# Data model

Four structures: one run-level config, one run-level request plan, two
record types.

## RunConfig (one per run_id, immutable once written)

  run_id             identifies a full eval run
  judgment_type      pairwise | scalar | binary
  scale_min          scalar only; e.g. 1
  scale_max          scalar only; e.g. 5
  ties_allowed       whether the judge may return a tie
  baseline_variant_id  scenario E only; the frozen baseline
  created_at

## PlannedJudgeCall (one per intended pairwise judge call, written at request creation, immutable)

  run_id, item_id, source_doc_id
  pair_id
  variant_first, variant_second    intended presentation order
  judge_model, judge_config_id
  call_role          primary | replicate (spec §12 gap 9, D11)
  judge_call_id      pre-assigned; the realized PairwiseJudgment echoes it
  created_at

## ResponseJudgment (scalar and binary judgments)

  run_id, item_id, source_doc_id
  variant_id         which prompt/system variant produced the response
  response_id        the generation being judged
  gen_seed           generation sampling params + seed
  response_tokens    for verbosity analysis
  judge_model        pinned snapshot id
  judge_config_id    judge temperature + seed + params
  judge_call_id      unique per judge invocation; replicates are separate rows
  annotator_id       human labels only; null for model judges
  judgment           scalar score or binary pass/fail
  created_at

## PairwiseJudgment (preference judgments)

  run_id, item_id, source_doc_id
  pair_id            stable id for the unordered variant pair on this item
  variant_first, variant_second
  response_id_first, response_id_second
  gen_seed_first, gen_seed_second
  tokens_first, tokens_second
  judge_model        pinned snapshot id
  judge_config_id
  judge_call_id
  call_role          primary | replicate; assigned at request creation,
                     immutable (spec §12 gap 9, D11)
  annotator_id       human labels only; null for model judges
  judgment           first | second | tie
  created_at

## Constraints

- `judge_model` must be an immutable snapshot id. Floating aliases ending
  in "-latest" are rejected at write time.
- `judgment` in PairwiseJudgment is recorded in presentation terms
  (first/second), never in variant terms. Which variant won is derived at
  analysis time.
- `presentation_order` is removed. Under PairwiseJudgment the ordering is
  structural, so the field is redundant and unverifiable.
- Human labels use the same records with `annotator_id` populated.
  Per-annotator identity is required for kappa and Krippendorff's alpha.
- RunConfig is immutable per `run_id`.
- `call_role` is immutable once written. Bias estimands consume exactly one
  primary-labeled call per (pair, order); a replicate is consumed (a
  "promotion") only when the plan shows a primary was requested and no
  primary succeeded. Promotions are derived at analysis time by joining
  records to `PlannedJudgeCall` on `judge_call_id`, never written back.
- The planned request set (`PlannedJudgeCall`) is written before any judge
  call is made. A realized judgment whose `judge_call_id` has no plan row,
  or a (pair, order) with judgments but no planned primary, is a pipeline
  error: counted separately from promotions and run-blocking (spec §12
  gap 9, D11).