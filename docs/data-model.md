# Data model

The unit of data in this project is a single judgment record: one judge's
verdict on one response, one row. Every table, estimator, and bias check in
`stats/`, `bias/`, and `agreement/` operates on collections of these records.

## Judgment record

| Field | Description |
|---|---|
| `run_id` | Identifies a full eval run. Groups every record produced by a single invocation of the harness, so results from different runs (different dates, configs, or codebases) are never accidentally pooled together. |
| `item_id` | The input case being evaluated. Identifies the underlying task/prompt/question independent of which variant or generation answered it, so responses to the same item can be compared to each other. |
| `variant_id` | Which prompt or system variant produced the response. The primary grouping variable for the comparisons this library exists to make — estimators contrast outcomes across `variant_id` values. |
| `response_id` | The specific generation being judged. Distinguishes individual samples even when they share the same `run_id`, `item_id`, and `variant_id`, since a variant may be sampled multiple times per item. |
| `gen_seed` | The sampling parameters and random seed used to produce the response. Makes generation reproducible and lets analyses separate genuine variant effects from sampling variance. |
| `judge_model` | Which model produced the judgment. Required to measure and report judge-specific bias, and to compare or calibrate across judges. |
| `judge_call_id` | Unique identifier per judge invocation. Repeated judgments of the same response (for measuring judge noise or computing intra-judge agreement) are separate rows rather than overwriting each other. |
| `presentation_order` | Which variant appeared first in the pairwise prompt. Needed to detect and correct for position bias, which requires counterbalanced presentation. |
| `judgment` | The judge's output: a preference between two responses, a scalar score, or a binary pass/fail. The outcome variable that every downstream estimator and test is computed on. |
| `response_tokens` | Token count of the judged response. Used to test for and adjust verbosity bias, i.e. whether judgments correlate with response length independent of quality. |
| `source_doc_id` | Nullable. Groups items drawn from the same source document. Records sharing a `source_doc_id` are not independent, so this field is required for correct variance estimation (e.g. clustered standard errors) whenever it is non-null. |
