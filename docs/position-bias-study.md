# Position bias study — design

Status: design, not yet run. This document pre-registers the design; deviations
at run time must be recorded in the results write-up.

Normative references: `docs/statistics-spec.md` §14.1–§14.2 (estimand and
reduction, DECISION D10, FLAGGED F13), §2 (inference engine, DECISION D9),
§6 (multiplicity, DECISION D6, FLAGGED F8), §8 (judge variance, DECISION D4),
§9 (power), §11 items 4 and 9 (validation prerequisites), §14.6 (report
block). Where this document makes a choice the spec leaves open, the choice is
marked **DESIGN DECISION** with the alternative and tradeoff.

---

## 1. Question and estimand

Do Anthropic judge models, when shown two candidate responses side by side,
systematically prefer the response shown first (or second), and does the
magnitude vary with judge capability?

The estimand is `β_pos` per judge, exactly as defined in spec §14.2: under the
additive model, the average increment in probability points to a response's
chance of being preferred from being displayed first, over and above content
preference, averaged over the study's pairs. `β̂_pos = 0.05` means being shown
first adds 5 percentage points of preference probability.

The estimator is the §14.2 order-balanced recode: every judge call recoded to
preference-for-first (first → 1, second → 0, tie → ½), averaged within
(pair, order), then across the two orders, then across pairs within an item,
giving `g_i`; `β̂_pos = mean(g_i) − ½`; the §2 engine runs on `d_i = 2g_i − 1`
and CIs/SEs map back by halving. Content preference and tie propensity cancel
exactly for any pair judged in both orders — this is counterbalancing by
construction, not by randomization.

**What is not identified (F13, disclosed in every report):** the position ×
pair interaction and any non-additive position structure. Heterogeneous
position sensitivity widens the CI without moving the point estimate. The
estimand is also judge-configuration-relative (§14 preamble): it describes
these judge models under this judge prompt and decoding config, not "position
bias of LLM judges" in general.

Each judge is a separate estimand. There is one primary test per judge
(`H0: β_pos = 0`, two-sided, α = 0.05 before correction — see §7).

---

## 2. Judges

Three Anthropic models spanning capability tiers, per constraint:

| Tier | Model ID | Judge pricing (in/out per MTok) |
|---|---|---|
| High | `claude-opus-5` | $5 / $25 |
| Mid | `claude-sonnet-5` | $3 / $15 |
| Small | `claude-haiku-4-5` (snapshot `claude-haiku-4-5-20251001`) | $1 / $5 |

Configuration, held fixed per judge for the whole run:

- **Verdict via structured outputs** (`output_config.format`, JSON schema with
  `verdict ∈ {"first", "second", "tie"}`). This makes parse failures near-zero
  by construction and removes free-text-parsing asymmetries that could
  correlate with position.
- **Thinking disabled** where the API permits it (Opus 5 at effort ≤ high,
  Sonnet 5; Haiku 4.5 has no thinking by default). **DESIGN DECISION:**
  thinking-on judges are plausibly less position-biased and are a natural
  follow-up, but adaptive thinking makes per-call output cost unbounded and
  unpredictable, which the budget cannot absorb, and thinking-off matches the
  most common production LLM-judge deployment. The estimate is explicitly
  relative to this config.
- **No sampling parameters.** Opus 5 and Sonnet 5 reject
  `temperature`/`top_p`; Haiku 4.5 runs at its default for symmetry. Judge
  nondeterminism is measured, not suppressed (§5, replicate subset).
- **One fixed judge prompt template** shared by all three judges: rubric-free
  "which response better answers the prompt" instruction, responses in two
  fixed slots. β_pos is relative to this template; slot labels are part of
  what "position" means here.

**Model pinning caveat (spec §12 gap 6):** the spec requires immutable judge
snapshots. Haiku 4.5 has a dated snapshot ID; `claude-opus-5` and
`claude-sonnet-5` are fixed-but-undated IDs, which is the strongest pin the
API offers for them. Mitigations: all judging for a given judge runs in one
co-batched window (≤ 48 h), the response-level `model` field and run
timestamps are recorded, and the report discloses the residual risk of a
silent provider-side update. This is a disclosed limitation, not a fixable
one.

---

## 3. Items and response pairs

### 3.1 Prompts

300 single-turn prompts, six categories × 50: explanation, summarization
(source text embedded in the prompt), reasoning/word problems, coding,
creative writing, practical advice. Prompts are curated from permissively
licensed public sets plus author-written items, deduplicated, each prompt an
independent topic (no shared source documents across prompts).

Inclusion filters, applied before generation:

- Open-ended enough that response quality can genuinely vary; no single
  short verifiable answer (those make every comparison a tie or a fact check).
- No prompts that elicit model self-identification ("who are you", "which
  model wrote this") and no Anthropic- or model-specific content — required
  for cross-family reuse (§10).
- Target response length reachable in 150–350 words, stated in the prompt, so
  length is roughly controlled by instruction rather than post-editing.
- License-clean for publication.

### 3.2 Response pairs — two strata, both orders each

Each prompt contributes **two pairs**, one per stratum, 600 pairs total:

- **Clear-gap stratum (300 pairs):** one response from `claude-sonnet-5`
  (strong arm) and one from `claude-haiku-4-5` (weak arm), same prompt, same
  generic generation instruction. Quality variation is real model-capability
  variation, not artificial degradation (injected errors or handicapped
  prompts would make the "worse" response detectably unnatural and would
  poison later reuse).
- **Near-tie stratum (300 pairs):** two independent samples from
  `claude-sonnet-5` at its default decoding. The two arms are exchangeable by
  construction, so expected content preference is exactly ½ — this stratum is
  where position bias has the most room to act and doubles as a negative
  control for content effects.

Rationale for the mixture: β_pos averaged over an all-decisive bank would be
attenuated toward whatever the judge does on easy calls; over an all-coin-flip
bank it measures maximal bias. The 50/50 mixture measures β over a bank with
genuine quality variation and supports the exploratory bias-vs-decisiveness
contrast within prompt. Primary β_pos is over the full 600-pair bank, items
equally weighted (spec D3 analog); the bank composition is a disclosed
curation choice.

**DESIGN DECISION — pairs nested in prompts rather than 600 independent
prompts.** Alternative: 600 prompts × 1 pair maximizes independent clusters
and power (no design effect). Chosen structure costs ~10% MDE inflation at
ρ ≈ 0.2 but (a) halves prompt-curation effort, (b) allows the stratum
contrast to be within-prompt, and (c) keeps the prompt bank small enough to
curate carefully. The power analysis in §6 charges the design effect.

### 3.3 Confounding controls

- **Position vs content:** eliminated by construction — every pair is judged
  in both orders and D10 excludes any pair that ends up single-order (excluded
  count reported; see §8 for the failure-asymmetry check).
- **Position vs length / style:** any pair-level attribute (length, format,
  style) is a content attribute and cancels in the order-balanced recode
  exactly as content preference does. Length is still measured and reported
  per arm and per stratum, because (a) the stratified interpretation
  ("bias on close calls") assumes the strata differ mainly in quality gap, and
  (b) the reuse plan wants length distributions on record.
- **Position vs generation model identity:** judges will judge pairs
  containing outputs of their own model family (unavoidable under the
  Anthropic-only constraint). Self-preference is a content preference and
  cancels in β̂_pos. What does not cancel is a position × generator
  interaction, which is a special case of F13 and carries the same
  disclosure.
- **Reproducibility:** the API offers no generation seed, so `gen_seed` in
  the data model cannot be honored; reproducibility is at the artifact level —
  all responses are generated once, frozen, stored with full request config,
  and every judge sees the identical frozen pairs. Recorded as a data-model
  deviation.

---

## 4. Factors, counterbalancing, clustering

| Role | Factor | Levels |
|---|---|---|
| Design factor | presentation order | both orders per pair, exhaustive (not sampled) |
| Design factor | judge model | 3, fully crossed with all pairs |
| Stratification | quality-gap stratum | clear-gap / near-tie, within prompt |
| Held fixed | judge prompt template, decoding config, slot labels, item bank | one level each |
| Unit / cluster | pair (item) nested in prompt | 600 pairs in C = 300 prompt clusters |

Judge calls: 600 pairs × 2 orders × 3 judges = 3,600 primary calls, plus the
replicate subset (§5). Every judge sees every pair in both orders — the
cross-judge secondary contrasts are fully paired at the pair level.

Clustering: `cluster(i) = prompt_id` (spec §1.4; the two pairs from one
prompt share topic and share the near-tie arms' generator). C = 300 ≥ 50, so
by DECISION D9 the **cluster percentile bootstrap CI is primary** and the
analytic cluster-robust interval is the always-reported cross-check. The
primary test is the §2.2 cluster sign-flip permutation test, which is exact
under the §14.1 exchangeability null (no position effect on any pair).

Analysis is per judge: three independent runs of the same reduction and
engine over the same pairs. Cross-judge contrasts (§7) reuse the paired
structure.

---

## 5. Judge noise: replicate subset (DECISION D4)

A seed-drawn random subset of **60 pairs per judge** (10%, > D4's floor of
30 responses) receives one additional call **in each order** — same-order
replicates (m = 2 in both orders for the subset), per the §8.2 caveat that
cross-order spread would upper-bound σ²_J by absorbing item-specific position
sensitivity. Cost: 60 pairs × 2 orders × 1 extra call × 3 judges = 360 extra
calls.

**Replicates do not enter the primary β estimate.** This is load-bearing, not
bookkeeping: a pair with unequal call counts across its two orders is exactly
the F11 side-unbalanced condition — under the null with content preference
`w`, a (2, 1)-call pair has `P(g = 1) = w²(1−w)` vs `P(g = 0) = (1−w)²w`,
equal only at `w = ½`, and the clear-gap stratum is built to sit far from ½.
`E[g] = ½` still holds, so the point estimate would survive, but `d` is then
mean-zero-asymmetric and the §2.2 sign-flip test loses its exactness. Even
the balanced (2, 2) subset design is kept out of the primary reduction so
that every item enters with an identical (r = 1, m = 1)-per-order structure.
The rule:

- Every judge call is labeled **primary** or **replicate** at request-creation
  time (by the run seed — never by completion order, which could correlate
  with load or content).
- The primary reduction consumes **exactly one call per (pair, order)**: the
  primary-labeled call. If a primary call fails and its retries are exhausted,
  the replicate call for that (pair, order) may be promoted — promotion is a
  relabeling that preserves one-call-per-side, and the count of promotions is
  reported.
- Replicate calls feed **only** σ̂²_J (pooled within-(pair, order) variance,
  §8.2), ICC_judge, and the achieved-MDE inputs.

**Pre-launch assertion on realized data:** before the engine runs, the
reduction's F11 side-balance check (spec §2.2) is executed over the realized
judgment records as a hard gate, not a warning — every item entering the
primary reduction must have exactly one primary call in each order. Any
violation (double-submitted retries, batch duplicates) is repaired to one
call per side by the request-time labels, or the pair is excluded under D10
and counted. §8 item (c) covers the exclusion-asymmetry consequence.

The subset yields σ̂²_J and ICC_judge per judge for the report (§14.6) and
feeds the achieved-MDE calculation. It is not needed for validity (§1.1 of
the spec: item-level analysis absorbs judge noise); it is needed so a
near-zero β̂ from a high-noise judge is not over-interpreted (§8
falsification, item f).

---

## 6. Sample size, MDE, and budget

### 6.1 Planning variance

`d_i = 2g_i − 1` with, at one call per order, `g_i ∈ {0, ¼, ½, ¾, 1}`.
Deterministically decided pairs contribute `g_i = ½` (zero variance); pure
coin-flip pairs contribute Var(g) = ⅛ (SD ≈ 0.35); position-sensitivity
heterogeneity adds between-pair variance. **Planning value σ_g = 0.40**
(σ_d = 0.80), the D5 planning default carried over to the g scale — labeled
an assumption, deliberately above the all-coin-flip value and below the
all-or-nothing worst case σ_g = 0.5. Sensitivity grid below spans 0.25–0.50.

### 6.2 Analytic MDE at the chosen design

Computed with the library's own machinery — `mde_paired_mean(n=600, sd_d=2·σ_g,
df=C−1=299)` and `design_effect(2, ρ)`, MDE mapped to the β scale by halving
(power 0.80 throughout, two-sided):

| σ_g | α = 0.05, ρ = 0 | α = 0.05, ρ = 0.2 | α = 0.05/3, ρ = 0 | **α = 0.05/3, ρ = 0.2** |
|---|---|---|---|---|
| 0.25 | 0.029 | 0.031 | 0.033 | 0.036 |
| 0.35 | 0.040 | 0.044 | 0.046 | 0.051 |
| **0.40** | 0.046 | 0.050 | 0.053 | **0.058** |
| 0.50 | 0.057 | 0.063 | 0.066 | 0.073 |

α = 0.05/3 is the Bonferroni-worst step of the Holm correction (§7) —
conservative planning for the smallest-p judge's threshold. The headline
planning claim:

> **MDE ≈ 0.05–0.06 on the β scale** (5–6 percentage points of preference
> probability) per judge at 80% power under the primary planning assumptions
> (σ_g = 0.40, ρ = 0.2, Holm-worst α), improving to ≈ 0.04 if pair-level
> variance comes in at σ_g = 0.35 and the nominal-α single-judge view is
> taken.

### 6.3 Is that MDE defensible?

A practically meaningful position effect is judged against what the harness
exists to measure: spec §9.3's worked example treats a 5-point pass-rate
difference as "worth shipping." A position bias of comparable size (β ≥ 0.05)
distorts any non-counterbalanced comparison by more than the effects teams
ship on, so it must be detectable — and it is, by design. Published position
effects for LLM judges are typically far larger (first-position preference
rates of 60–75%, i.e. β ≈ 0.10–0.25), well inside the detectable region.

What the budget cannot buy: detecting β = 0.03 at the same standards requires
~1,700–2,200 pairs (`n_paired_mean(0.06, 2·σ_g, alpha=0.05/3, deff=1.2)` →
1,714 at σ_g = 0.35, 2,237 at 0.40), roughly $80+ of model calls. So the
study is powered to detect *practically meaningful* bias, not to certify its
absence below ~4–5 points; a null result is reported with exactly that bound
(§8). The design is not underpowered for its stated question; it would be
underpowered for "is there any position bias at all," which it does not claim
to answer.

### 6.4 Simulation gate (normative, spec §9.2)

The outcome is discrete and clustered, so per §9.2 the analytic grid above is
planning-only. **Before spending API budget**, run `mde_simulated` /
`simulated_power` with a generative model matching the design (5-point g
support from a decisive/coin-flip pair mixture, C = 300 × 2, prompt-level ρ,
injected additive β), through the *actual* §2 pipeline. Launch gate: simulated
MDE at Holm-worst α within +0.01 of the analytic value (i.e. ≤ ~0.06);
otherwise revisit n before generating anything. This shares the §11.9
simulator (see §11 prerequisites).

### 6.5 Budget

Token planning values: judge call ≈ 1,100 input (instructions + prompt + two
150–350-word responses) + ≤ 64 output (structured verdict); generation ≈ 120
input + 350 output.

| Line | Calls | Est. cost |
|---|---|---|
| Generation, Sonnet 5 (3 responses/prompt × 300) | 900 | ≈ $5.10 |
| Generation, Haiku 4.5 (1/prompt × 300) | 300 | ≈ $0.60 |
| Judging, Opus 5 (1,200 + 120 replicates) | 1,320 | ≈ $9.40 |
| Judging, Sonnet 5 (1,320) | 1,320 | ≈ $5.70 |
| Judging, Haiku 4.5 (1,320) | 1,320 | ≈ $1.90 |
| **Total (list price)** | | **≈ $23** |

Headroom ≈ $7 (23%) for retries, longer-than-planned responses, and the
excluded-pair re-runs in §8. Two cost levers if estimates run hot: the
Batches API (50% discount; judging is not latency-sensitive and co-batching
per judge is what §7.1-style drift hygiene wants anyway) and Sonnet 5 intro
pricing if the run lands before 2026-08-31. The levers are headroom, not
load-bearing: the design fits list price.

---

## 7. Multiplicity (DECISION D6, FLAGGED F8)

**Pre-declared primary family (m = 3):** `H0: β_pos^(J) = 0`, two-sided, one
per judge. Correction: **Holm step-down**, FWER 0.05. CIs reported at the
Bonferroni level `1 − 0.05/3` ≈ 98.33% alongside Holm-adjusted p-values, with
the F8 disclosure that a Holm rejection need not match a Bonferroni CI
excluding zero; unadjusted 95% intervals also shown, labeled descriptive.

**Secondary, labeled exploratory, uncorrected, no confirmatory claims:**

- Pairwise cross-judge differences `β^(J) − β^(K)` (3 contrasts, paired at
  the pair level, same engine on the per-pair difference of recodes).
- Per-stratum β̂ (clear-gap vs near-tie) per judge, and their within-prompt
  contrast.
- β̂ vs response-length-gap bins; tie-rate comparisons.

The capability-trend claim ("bias shrinks with capability") is exploratory:
with three points on an unquantified capability axis, the study reports the
three estimates side by side and does not fit a trend.

---

## 8. Falsification

**What would show no position bias:** all three Holm-adjusted tests fail to
reject and every Bonferroni CI for β lies inside ±0.05, with achieved MDE
(computed from realized s_d, realized C, D9-primary interval) ≤ 0.06. The
honest null statement is then "for these judges under this configuration, any
additive position effect is smaller than ~5 points" — an informative bound,
not proof of zero. A null with achieved MDE > 0.06 (variance came in above
plan) is reported as underpowered, per spec §9/§10.

**What would indicate the design failed rather than the hypothesis** — checked
in order, before any β̂ is interpreted:

a. **Pre-launch validation red.** Spec §11.4 (counterbalanced reduction
   removes an injected additive effect exactly) and §11.9 (position-bias
   estimator recovers injected β with correct coverage at C ∈ {20, 30, 50})
   must be green on the exact code the study runs. A red cell is an
   implementation failure; nothing downstream is interpretable.
b. **Simulation gate miss** (§6.4): analytic and simulated MDE disagree by
   more than 0.01 — planning model wrong; re-plan, don't launch.
c. **Side-balance assertion (F11 gate, §5).** Every item entering the primary
   reduction has exactly one primary-labeled call per order, verified on the
   realized records before the engine runs. A violation that cannot be
   repaired by the request-time labels is a pipeline failure: replicate calls
   leaking into the primary reduction would make clear-gap `d_i`
   mean-zero-asymmetric and void the sign-flip test's exactness.
d. **Order-asymmetric exclusions.** D10 drops single-order pairs. If > 5% of
   pairs are excluded for any judge, or exclusions are asymmetric in which
   order failed (binomial test on failure order), selection correlates with
   position and β̂ is compromised for that judge. Mitigation at run time:
   retry failures within budget headroom before excluding.
e. **Judge competence gate.** On the clear-gap stratum the order-balanced win
   rate of the strong arm should be decisively above ½ (gate: point estimate
   ≥ 0.65 for every judge; the tier gap makes this conservative). A judge
   below the gate is not discriminating quality; its β̂ remains internally
   valid (β needs no quality signal) but the "position bias in a competent
   judge" interpretation and the stratified analysis are void for it.
f. **Degenerate judge behavior.** Tie rate > ~30% (D2: win-scale compression
   — a finding about the judge, but β̂ is attenuated and the report must say
   so); or ICC_judge ≈ 0 with large σ̂²_J (the run measured coin flips).
g. **Interval disagreement.** Large bootstrap-vs-analytic CI disagreement
   (§2.3) — skew or outlier clusters; report both, investigate before
   headlining.
h. **Near-tie stratum content check.** Within the near-tie stratum the two
   arms are exchangeable, so the order-balanced *content* win rate for
   arm-1-as-generated must be consistent with ½. Deviation means arm labeling
   leaked into presentation (pipeline bug), not a judge property.

A large β̂ passes falsification only if (a)–(h) are clean; the design cannot
manufacture position bias out of content, length, or generator effects
because all of those cancel in the order-balanced recode by construction —
the residual ways to fake it are exactly the pipeline failures listed above.

---

## 9. Analysis outputs and plots

Per-judge report block per spec §14.6 (quantity name, identified-bias status,
D9-primary CI + non-primary interval + measured-coverage disclosure, p and
null tested, n/C/cluster sizes, exclusion counts with reasons, judge models,
assumption list, F-flags F13 and the §2 flags fired, achieved MDE).

Figures (all with the numbers they plot available as CSV):

1. **Headline forest plot.** β̂_pos per judge, ordered by capability tier;
   Bonferroni-level CI (bold) and unadjusted 95% CI (light) per judge;
   reference line at 0 and shaded ±0.05 "practical materiality" band;
   Holm-adjusted p annotated. One glance answers: which judges are biased,
   which direction, and does it shrink with capability.
2. **Cross-order verdict patterns.** Per judge, a stacked bar over pairs
   classified by their two-order verdict pattern: content-consistent (same
   response wins both orders), position-consistent (first-shown wins both),
   anti-position (second-shown wins both), tie-involved/mixed. This is the
   mechanism picture: β̂ lives in the imbalance between the position-consistent
   and anti-position segments, and the content-consistent share shows how
   often content dominates position. The most legible single figure for a
   non-statistical reader.
3. **Stratum contrast (exploratory).** Grouped forest: β̂ on clear-gap vs
   near-tie pairs per judge with unadjusted CIs. Shows whether position bias
   concentrates on close calls — the operationally important pattern, since
   close calls are where evals are decided.
4. **Distribution of g_i.** Per judge, histogram over the five attainable
   g values. Shows the discreteness the CIs must live with, where the mass
   sits (½ = content-decided; ¼/¾ and 0/1 = order-flipped or
   position-locked), and makes any degenerate judge behavior visible.
5. **Validity panel.** Per judge: clear-gap competence win rate with CI
   (gate line at 0.65), tie rate, exclusion count by order, σ̂_J and
   ICC_judge. The figure a skeptical reviewer checks before believing
   figure 1.

---

## 10. Reuse for a later self-preference study (§14.4)

Not designed here; the constraint honored is that nothing in this design
makes its artifacts unusable for a cross-family generation pass.

**Carries over:**

- **Prompt bank + strata metadata.** Model-agnostic, no self-identification
  elicitors, no Anthropic-specific content, license-clean, category-balanced
  — a cross-family pass can generate new arms on the identical prompts.
- **Frozen Anthropic response sets** with full generation config — usable as
  the comparison arm ("self" vs other-family, or as panel-shared material);
  the near-tie Sonnet samples are natural-quality responses, deliberately not
  artificially degraded.
- **Infrastructure:** both-orders-or-exclude enforcement (D10), co-batched
  judging, structured-verdict schema, caching, the judge prompt template.
- **Analysis:** §14.1 reduction and engine are shared; §14.4's estimator
  consumes the same order-balanced pair scores.
- **This study's estimates:** per-judge β̂_pos and σ̂²_J. Position effects
  cancel in the §14.4 cross-judge contrast given counterbalancing, but the
  measured magnitudes justify that design choice quantitatively, and σ̂²_J
  feeds the self-preference power analysis.

**Does not carry:** the judgments themselves (§14.4 needs every panel judge
scoring identical pairs that include each self model's responses — new pairs,
new judging); the clear-gap weak arm as a "self" arm (self-preference wants
natural-quality arms, and Haiku-as-weak-arm was a capability-gap device); the
Holm family (a new study pre-declares its own).

**Choices made here specifically for that reuse:** prompt filters in §3.1;
generic generation instructions (no persona, no model identity in system
prompts); artifact-level freezing with full config; 300 prompts is enough to
sub-sample a fresh, judgment-uncontaminated item set if the self-preference
study wants pairs no judge has seen.

---

## 11. Prerequisites and open items

Blocking, in order:

1. **Runner complete** (CLAUDE.md status: in progress), including the D10
   both-orders-or-exclude rule and the §12.1 pairwise-record resolution in
   `docs/data-model.md` — the §14.1 reduction is not implementable from
   records until the pairwise schema choice is committed.
2. **Validation green:** §11.4 counterbalancing identity and §11.9
   position-bias recovery/coverage cells, run against the shipping code.
3. **Simulation gate** (§6.4) passes on the design's generative model.
4. **Prompt bank curated and frozen** (§3.1), with licensing recorded.

Open items flagged for the run, not blocking design sign-off:

- Judge-model pinning residual risk (§2) — disclosed, mitigated by co-batched
  windows.
- No API generation seed (§3.3) — artifact-level reproducibility recorded as
  a data-model deviation.
- σ_g = 0.40 is a planning assumption (D5 analog); the achieved MDE from
  realized variance is the number the write-up must headline next to any
  null.
