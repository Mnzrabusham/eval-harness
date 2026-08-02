# Statistics specification

This document defines every estimator, test, interval, and power calculation
in `stats/`. It is the source of truth: if code disagrees with this document,
the code is wrong or this document must be amended in the same commit.

Companion document: `docs/data-model.md` defines the judgment record. Section
12 records the schema gaps this spec originally flagged and how
`docs/data-model.md` resolved each one.

**Normative language.** MUST / MUST NOT are requirements; SHOULD is a default
that may be overridden with documented justification. Three markers flag
places where a choice was made or remains open:

- **DECISION** — more than one approach is defensible; the spec picks one and
  records the alternative and the tradeoff. Implementers follow the pick.
- **FLAGGED** — a known limitation or approximation the report must disclose.
- **OPEN** — genuinely unsettled; the spec describes options but does not
  commit. Code MUST NOT silently implement one; raise it first.

Section 13 indexes all of these.

**Scope limitation, stated up front.** Every estimand in this document is
defined relative to a single judge configuration. A result reads "variant A
beats variant B *as scored by judge J*", never "A is better than B." No
number of items or replicate judgments removes judge-level systematic error
(a judge that likes verbose answers, likes its own outputs, etc.). That
error is measured in `bias/` and calibrated against humans in `agreement/`;
it is not correctable inside `stats/` and every report MUST carry this
caveat verbatim or equivalent.

---

## 0. Normative defaults

All defaults are overridable per run; the values used MUST appear in the
report.

| Parameter | Default | Notes |
|---|---|---|
| `alpha` | 0.05 | two-sided (DECISION D1) |
| `n_boot` | 10,000 | bootstrap replicates |
| `n_perm` | 10,000 | Monte Carlo permutation draws |
| `enumerate_threshold` | 13 | exhaustive sign-flip enumeration when clusters ≤ 13 (2¹³ = 8,192 flips) |
| CI method | cluster percentile bootstrap | see §2.1; primary only at C ≥ 50 (DECISION D9); BCa is FLAGGED F2 |
| Quantile definition | `numpy.quantile(..., method="linear")` | pinned so two implementations agree |
| RNG | `numpy.random.Generator(numpy.random.PCG64(seed))` | explicit seed, threaded through every stochastic call; no global state |
| Tie score (pairwise) | 0.5 | DECISION D2 |
| Multiplicity | Holm step-down | DECISION D6; family defined in §6.1 |
| Min clusters before warning | 20 | FLAGGED F4, rule of thumb |

Reproducibility requirement: given identical input records, seed, and
parameters, two independent implementations of this spec MUST produce
bit-identical point estimates and test decisions, and identical CIs and
p-values up to the shared RNG stream (same seed → same draws, since both use
PCG64 and the resampling procedures below are fully specified, including
iteration order: clusters sorted lexicographically by cluster id, items
sorted lexicographically by `item_id` within cluster, before any resampling).

---

## 1. Estimands and the unit of analysis

### 1.1 The unit of analysis is the item

The judgment record is one judge call on one response, but judge calls on the
same item are not independent — they share the item's difficulty, and
replicate calls share the response itself. Treating judgment rows as i.i.d.
observations is pseudo-replication and is the single most common way eval
statistics come out anti-conservatively wrong. All inference in this library
therefore operates on **item-level summaries**, computed by the reduction in
§1.2, and treats the item (or the source-document cluster, §1.4) as the unit
of resampling and degrees of freedom.

A consequence worth stating: once the analysis is at item level, judge
nondeterminism and generation sampling variance cannot invalidate the test —
they only inflate within-item noise, which the item-level variance absorbs.
Replicates buy precision, not validity. Section 8 quantifies this.

### 1.2 Item-level reduction (normative pipeline)

For every scenario, raw judgment rows reduce to one number per item,
`d_i ∈ [−1, 1]`, with `E[d_i] = 0` under "no difference." All three outcome
types map onto this one engine:

**Scalar and binary outcomes** (scenarios B, C). For variant `v` and item
`i`:

1. Response-level mean: for each response `r`, average its judgment values
   over the judge calls on it: `y_r = mean_j(y_rj)`.
2. Item-level mean: average over responses: `x_iv = mean_r(y_r)`.
   Responses get equal weight regardless of how many judge calls each
   received (unbalanced judging must not silently reweight responses).
3. `d_i = x_iA − x_iB` for scalar (raw score units, see §4), and the same
   for binary, where pass = 1, fail = 0, so `d_i ∈ [−1, 1]`.

**Pairwise preference** (scenario A). Each judge call sees a pair of
responses in a presentation order and returns `first` / `second` / `tie`.

1. Recode each call to preference-for-A: `y = 1` if A preferred, `0` if B
   preferred, `0.5` if tie (DECISION D2).
2. Pair-and-order mean: for each response pair `p` and order
   `o ∈ {A-first, B-first}`, average calls: `y_po`.
3. Order-balanced pair score: `s_p = (y_{p,A-first} + y_{p,B-first}) / 2`
   when both orders exist. Averaging within order first, then across orders,
   is required: it counterbalances position bias even when the two orders
   got unequal numbers of judge calls. If only one order exists for a pair,
   `s_p` is the available order's mean and the run MUST be flagged
   "not counterbalanced; position bias uncorrected" (FLAGGED F1).
4. Item score: `s_i = mean_p(s_p)`; then `d_i = 2·s_i − 1 ∈ [−1, 1]`.

The reported estimand for scenario A is the win rate `θ = E[s_i]`
(H0: θ = 0.5), which is a linear transform of `E[d_i]`; the inference engine
runs on `d_i` and the report maps back.

**Missing sides.** If an item lacks responses for one variant, it has no
`d_i` and is excluded, and the count of excluded items MUST be reported.
Systematic missingness (e.g., variant B errors out on hard items) biases the
estimand toward the surviving items; the report must say how many items were
dropped and why. No imputation (imputing outcomes for a variant that failed
to answer is a modeling choice this library does not make; a hard failure
should instead be scored as the worst outcome *at generation time* if the
run's rubric says so).

### 1.3 What the estimand is

`Δ = E[d_i]`, the expected item-level difference, where the expectation is
over (a) the item bank actually used, treated as the population of interest,
and (b) the judge's and generator's sampling randomness. Two things follow:

- **Items are equally weighted** (DECISION D3). If 40 of 100 items come from
  one source document, that document contributes 40% of the estimand. The
  alternative — weight clusters equally — changes the estimand, not just the
  variance. The spec picks item-weighting because the item bank's
  composition is a deliberate curation choice; the report MUST show the
  cluster size distribution so a reader can see concentration.
- **The estimand conditions on the judge** (see scope limitation above).

### 1.4 Clusters

`cluster(i) = source_doc_id` when non-null, else the item is its own
singleton cluster. Records sharing a source document are positively
correlated (shared topic, shared difficulty), so the cluster — not the item —
is the independent unit for variance estimation, resampling, and degrees of
freedom. Ignoring this makes standard errors too small by roughly
`sqrt(1 + (m̄ − 1)·ρ)` where `m̄` is mean cluster size and `ρ` the
within-cluster correlation of `d_i` (the design effect, §9.3).

Let `C` = number of clusters. All warnings and small-sample rules key on
`C`, not on the number of items.

---

## 2. The shared inference engine

Defined once; every scenario references it. Input: the vector
`d = (d_1, …, d_n)` with cluster assignments. Output: point estimate, CI,
p-value, and diagnostics.

### 2.1 Point estimate and cluster percentile bootstrap CI (primary)

- `Δ̂ = mean(d_i)` over all items.
- Bootstrap: for `b = 1 … n_boot`: draw `C` clusters uniformly with
  replacement from the `C` clusters; concatenate the `d_i` of the sampled
  clusters (a cluster drawn twice contributes its items twice); compute
  `Δ*_b = mean` of that vector. Note this is an item-weighted mean within
  each resample; resamples have varying item counts — that is correct and
  intentional.
- CI: `(quantile(Δ*, α/2), quantile(Δ*, 1 − α/2))` with the pinned linear
  quantile method. Also report bootstrap SE `= std(Δ*, ddof=1)`.
- Degenerate case: if all `d_i` are identical the CI collapses to a point;
  report it as such with a warning.

**FLAGGED F2** — percentile intervals under-cover slightly with few clusters
or heavy skew (empirically ~1–3 points below nominal at C ≈ 20). BCa corrects
some of this at the cost of implementation complexity (jackknife
acceleration, more room for two implementations to diverge). The spec keeps
percentile as normative and requires the validation suite (§11) to measure
and publish the actual coverage; if measured coverage at realistic C is
unacceptable, revisit this decision rather than patching ad hoc.

**DECISION D9 — analytic interval primary below C = 50.** The evidence F2 was
deferred pending has arrived (§11.1, reps = 10,000; figures are read from
`evalkit/stats/coverage_evidence.py`, the single source for this table —
neither this document nor the code hardcodes a second copy):

| C | bootstrap coverage | analytic coverage (§2.3) |
|---|---|---|
| 20 | 92.53% | 94.94% |
| 30 | 93.48% | 95.02% |
| 50 | 94.15% | 95.05% |
| 200 (unclustered) | 94.72% | 95.00% |

(worst cell per C; full grid in `validation/results/coverage_reps10000.csv`)

MC-SE at 10,000 reps is ≈0.22 points, so the C = 20 and C = 30 shortfalls
(2.47 and 1.52 points — roughly 11σ and 7σ) are real drift, not noise; the
C = 50 shortfall (0.85 points, ~4σ) is smaller but still measurable; the
analytic interval tracks nominal throughout the grid.

Below C = 50, the analytic cluster-robust interval (§2.3) is therefore
*primary* — it is what a report headlines as `ci_low` / `ci_high`. The
bootstrap is still always computed and disclosed alongside it (§10), never
dropped; large disagreement between the two remains a diagnostic. At
C ≥ 50 the roles are as originally specified: bootstrap primary, analytic
the cross-check.

*Acceptable shortfall and threshold.* The spec treats a 1-percentage-point
shortfall (a 94.0% floor against 95% nominal) as the acceptable practical
limit — comfortably above the ~0.22-point MC-SE floor at 10k reps, so it is
a real, decidable cutoff rather than an artifact of measurement precision,
and it matches the §11.1 acceptance-band philosophy (D8) of tolerating
small, controlled undercoverage while treating larger drift as not
acceptable. C = 50 is the smallest measured C at which the bootstrap
shortfall (0.85 points) first falls under that floor; C = 30's 1.52-point
shortfall does not.

*Alternatives considered and rejected:*

- **Threshold at C = 20**, matching the existing generic small-sample
  warning (§2.4, F4). Rejected: the C = 30 cell still shows a real
  1.52-point (~7σ) shortfall, so aligning the analytic-primary threshold with
  F4 would headline C = 20–49 runs with an interval the evidence says
  under-covers.
- **A pure significance threshold** (switch primary wherever the measured
  shortfall exceeds 3×MC-SE, with no minimum-magnitude floor). Rejected:
  MC-SE shrinks as the validation suite's replication count grows, so a
  significance-only rule would keep moving the threshold upward over time
  even if the true shortfall at a given C never changes — it conflates
  measurement precision with practical importance. A fixed percentage-point
  floor does not have that failure mode.
- **A more conservative threshold (e.g. C = 100)**. Rejected: at C = 50 the
  measured shortfall (0.85 points) is already within the accepted floor and
  close to the unclustered C = 200 baseline (0.28 points, plausibly noise);
  moving the threshold higher would discard the bootstrap's role with no
  coverage evidence behind it.

### 2.2 Cluster sign-flip permutation test (primary test)

Null hypothesis: within each item, the variant labels are exchangeable —
equivalently, the joint distribution of each cluster's difference vector
`(d_i)_{i ∈ c}` is symmetric about 0. Under H0, flipping the sign of all
differences in a cluster simultaneously leaves the distribution unchanged.

- Test statistic `T = Δ̂` (the mean; equivalent to the t statistic for this
  randomization distribution up to monotone transform, and simpler).
- If `C ≤ enumerate_threshold`: enumerate all `2^C` sign assignments;
  `p = #{assignments: |T*| ≥ |T|} / 2^C`. The identity assignment is
  included, so `p ≥ 2^−C > 0`.
- Else: draw `n_perm` independent uniform sign vectors (one sign per
  cluster, applied to every item in the cluster);
  `p = (1 + #{b: |T*_b| ≥ |T|}) / (n_perm + 1)`.
- Comparisons `|T*| ≥ |T|` are exact float comparisons; the statistic is a
  mean of the same summands with flipped signs, so exact equality occurs
  naturally (e.g., the identity flip) and is counted, which is the
  conservative direction.

Why this is the primary test: it is finite-sample valid under the stated
null with no normality assumption, it handles clustering by construction,
and with binary single-judgment data it reduces to the exact McNemar
randomization distribution (zeros contribute nothing; the flips of the
discordant items reproduce the conditional binomial — see §5). One engine,
validated once, instead of a per-scenario zoo.

Assumption honesty: the symmetry null is stronger than "E[d] = 0". Under
the exchangeability null, symmetry of each cluster's difference vector
holds by construction for *any* marginal score distribution — skewed
rubrics, floor/ceiling compression, ordinal discreteness all afflict both
variants identically — **provided the replicate structure is
side-balanced**: the same (r, m) design on both sides of every item. Two
real situations fall outside the symmetry null while "E[d] = 0" may still
hold:

- **Side-unbalanced designs (FLAGGED F11).** If one variant has more
  responses or judge calls per item than the other, `x_iA` and `x_iB` are
  averages over different numbers of draws; with skewed score
  distributions their difference is mean-zero but asymmetric under "no
  difference." The reduction MUST detect differing per-side (r, m)
  structure at runtime and warn, naming the affected items; when the
  observed `d_i` are also materially skewed (default: |sample skewness|
  > 1) the warning MUST be prominent, because sign-flip exactness for the
  mean-zero null is then compromised.
- **Distribution-different, mean-equal variants.** Equal means but
  different shapes (e.g., one variant high-variance, the other piled at a
  floor). Outside the null as stated; a rejection is then a correct
  rejection of exchangeability, not evidence about means.

Consequences for the p-value when only "E[d] = 0" holds: the sign-flip
variance matches the true variance of `Δ̂` (`Σ_c S_c²` is flip-invariant),
so the **two-sided** test remains asymptotically valid, with residual level
error O(1/C) — roughly ±1 point at C ≈ 20–50 under strong skew, in either
direction, not guaranteed conservative. **One-sided** p-values inherit an
O(γ/√C) error (γ = skewness of the cluster sums) that does not cancel:
anti-conservative when rejecting into the long tail, conservative in the
other direction (FLAGGED F12). These magnitudes are measured, not assumed:
§11.8 defines null-robustness probe cells. The report states the null as
"no difference between variants," which is the exchangeability null — that
is what is actually being tested.

One-sided tests (permitted by DECISION D1 when pre-declared): the
one-sided p-value uses the same randomization distribution without
absolute values, `p = #{T* ≥ T} / 2^C` for the upper-tailed test (`≤` for
lower-tailed); the Monte Carlo version applies the same
`(1 + #exceed) / (n_perm + 1)` correction. One-sided reports MUST carry
the F12 caveat when `d` is materially skewed.

### 2.3 Analytic cluster-robust interval

Always computed and always reported alongside the bootstrap (§2.1), never
one instead of the other. Which of the two is *primary* — the interval a
report headlines as `ci_low` / `ci_high` — depends on the achieved cluster
count C: analytic below C = 50, bootstrap at or above (DECISION D9, §2.1;
measured coverage in `evalkit/stats/coverage_evidence.py`). The non-primary
interval is still always carried in the report (§10) and disclosed with the
measured coverage that motivates the choice. Large disagreement between the
two remains a diagnostic in its own right (skew, outlier cluster, small C)
regardless of which is primary.

- `S_c = Σ_{i ∈ c} d_i`, `n_c = |c|`, `n = Σ n_c`.
- `SE² = [C / (C − 1)] · Σ_c (S_c − n_c·Δ̂)² / n²`.
- Interval: `Δ̂ ± t_{1−α/2, C−1} · SE`; p-value from `t = Δ̂ / SE` with
  `C − 1` degrees of freedom.
- With all-singleton clusters this reduces exactly to the classical paired
  t-test on `d_i` — that identity is a required unit test.

### 2.4 Small-sample rules

- `C < 20`: emit a warning on the report — bootstrap and cluster-robust
  intervals are both unreliable here; the permutation p-value remains valid
  but has granularity `≥ 2^−C` (FLAGGED F4; 20 is a rule of thumb, not a
  theorem).
- Scenario C with fewer than 10 discordant items: the CI is unstable; report
  the discordant counts directly and state that the run is underpowered
  (§9's worked example shows why this happens constantly at n = 50).
- Effective sample size reporting: always report both `n` items and `C`
  clusters. If `C ≪ n`, the honest headline is C.

---

## 3. Scenario A — two variants, same items, pairwise judge preference

**Data.** Judge sees (response_A, response_B) per item, counterbalanced
across presentation orders, possibly with replicate calls and multiple
response pairs per item.

**Estimand.** Win rate `θ = E[s_i]`, ties scored 0.5. H0: θ = 0.5.

**Estimator.** Reduction §1.2 → `θ̂ = mean(s_i)`; engine §2 runs on
`d_i = 2s_i − 1`; CI for θ maps back by `(1 + CI_d) / 2`.

**Test.** §2.2 sign-flip permutation; §2.3 as cross-check.

**Why the naive choice is wrong.** The naive analysis pools all judgment
rows and runs a binomial test of "wins / total comparisons" against 0.5.
Three failures:

1. *Pseudo-replication.* Replicate judgments and multiple pairs per item are
   correlated through the item; the binomial variance `θ(1−θ)/N_rows`
   understates the truth, sometimes badly (if judge noise is small, 5
   replicate judgments are ~1 effective observation, not 5). This is
   anti-conservative — the direction that produces fake wins.
2. *Position bias.* Without order-balanced averaging, a judge that favors
   the first-shown response by even a few points shifts θ̂ directly. The
   §1.2 reduction removes any *additive* position effect by construction.
   What survives is a position × variant interaction (position bias that is
   stronger for one variant); that is measured in `bias/` and is FLAGGED
   (F1) in reports, not corrected here.
3. *Silent tie handling.* Dropping ties estimates
   `P(A wins | not a tie)` — a different estimand that depends on the
   judge's tie propensity, which varies across judge models. Dropping must
   never be silent.

**DECISION D2 — ties count 0.5.** Alternatives: (a) drop ties and report the
conditional win rate — defensible, matches the sign test; (b) model ties
explicitly (Bradley–Terry–Davidson) — heavier machinery, adds a model
assumption. The spec scores ties 0.5 because it keeps θ interpretable as an
expected score, keeps the estimand stable across judges with different tie
propensities, and degrades gracefully when ties are rare. The tie rate MUST
be reported; if it exceeds ~30% the win rate compresses toward 0.5 and the
report should say the judge rarely discriminates — that is a finding about
the judge, not evidence of variant equivalence.

**Assumptions and failure modes.**

- Both orders judged per pair. If not: F1, position bias flows into θ̂ at
  full strength.
- Items exchangeable under H0 (§2.2). Broken if e.g. the A responses were
  generated at a different time/config than B — enforce same-run generation.
- Judge stability over the run. A judge model updated mid-run breaks
  exchangeability; `judge_model` must be an immutable snapshot (§12).

**Report.** θ̂, CI, p, n items, C clusters, tie rate, number of judge calls
per pair, order balance, excluded items, and the judge-relativity caveat.

---

## 4. Scenario B — two variants, same items, scalar rubric score

**Data.** Each response scored on a rubric scale `[lo, hi]` (e.g., 1–5).
Judged independently per response (not side by side), possibly with
replicates.

**Estimand.** `Δ = E[x_iA − x_iB]` in raw rubric units.

**Estimator / test.** Reduction §1.2 → `d_i`; engine §2. Report Δ̂ in rubric
units with CI; never report a bare standardized effect (§10).

**Why the naive choice is wrong.** The naive analysis is a two-sample
(unpaired) t-test of all A scores vs all B scores. Because both variants
answered the *same* items, the scores share the item-difficulty component:
`Var(x̄_A − x̄_B)` for paired data is `(σ_A² + σ_B² − 2·Cov)/n`, and the
covariance is typically large — item difficulty usually dominates variant
effects. The unpaired test uses `(σ_A² + σ_B²)/n`, overstating the variance
by a factor `1/(1 − ρ)` where ρ is the item-level correlation. At a typical
ρ = 0.5 the unpaired test needs **twice** the items for the same power; at
ρ = 0.8, five times. Separately, if replicates are pooled as rows, the
unpaired test is anti-conservative on that axis (pseudo-replication) — the
two errors do not cancel, they just make the result uninterpretable.

**FLAGGED F3 — ordinal scores treated as interval.** A 1–5 rubric is
ordinal; taking means assumes the steps are equally spaced. This is the
universal practice and usually harmless for *comparison* (both variants are
distorted identically), but it is an assumption: a judge that only ever
outputs 4s and 5s makes the scale effectively binary. Alternatives:
Wilcoxon signed-rank (tests symmetry of differences, not the mean; behaves
poorly with the heavy ties of discrete scales) or ordinal (proportional
odds) regression (a real model with its own assumptions; statsmodels
supports it). The spec stays with the mean on the raw scale and requires the
report to show the score *distribution* per variant (histogram or counts per
level), so scale degeneracy is visible. If more than ~50% of the mass sits
on one scale point, treat the outcome as effectively binary and say so.

**Assumptions and failure modes.**

- Same rubric prompt, same judge config for both variants — otherwise Δ
  confounds variant with rubric drift.
- Scale bounds known and constant across the run (needed to normalize for
  §7's bounded-outcome methods and to detect ceiling effects). Ceiling:
  when one variant's mean is near `hi`, differences compress and Δ
  understates the true gap; report the fraction of maximal scores.
- Scores judged side-by-side instead of independently would reintroduce
  position effects; this scenario assumes independent scoring. If scoring
  was comparative, use scenario A instead.

**Report.** Δ̂ with CI in rubric units, p, per-variant means and score
distributions, n, C, replicate counts, σ̂_judge (§8) so the reader can see
how much of the noise is the judge.

---

## 5. Scenario C — two variants, same items, binary pass/fail

**Data.** Pass/fail per response, same items; possibly replicated judgments
or multiple responses per item (then `x_iv ∈ [0,1]` is a pass *probability*
estimate and the general engine applies unchanged).

**Estimand.** Pass-rate difference `Δ = p_A − p_B` (risk difference).

**Estimator / test.** Reduction §1.2 → `d_i`; engine §2. In the special case
of exactly one response and one judgment per side, `d_i ∈ {−1, 0, +1}` and
the classical structure appears: `n₊ = #{d_i = +1}` (A passed, B failed),
`n₋ = #{d_i = −1}`, and `Δ̂ = (n₊ − n₋)/n`.

**Required classical cross-check (no clustering, single judgments only):**
exact McNemar. Conditional on the discordant count `n_d = n₊ + n₋`, under
H0 `n₊ ~ Binomial(n_d, ½)`;
`p = min(1, 2 · P(X ≤ min(n₊, n₋)))`, and `p = 1` when `n₊ = n₋`.
The sign-flip permutation test (with singleton clusters) has exactly this
null distribution — concordant items flip to themselves and drop out — so
the two MUST agree up to Monte Carlo error; the validation suite asserts it.
Mid-p McNemar (subtract half the point mass) is closer to nominal on average
but not guaranteed conservative; FLAGGED F5, not used. Measured (§11.2,
reps = 10,000): the exact test's true rejection rate under the null is 2.79%
at ψ = 0.3 and 2.08% at ψ = 0.1 against nominal α = 0.05 — conservative by
construction, a consequence of the McNemar statistic's discreteness (not a
bug; anti-conservatism is what would fail the build here, and this is the
opposite). This conservatism is why the exact test's realized power falls
short of the Connor formula's nominal-level prediction; see §9.1.

**Why the naive choice is wrong.** The naive analysis puts pass counts in a
2×2 (variant × pass/fail) and runs a chi-square or two-proportion z-test.
That test assumes the two samples are independent; here they are the same
items. Concretely: the information about Δ lives entirely in the
*discordant* items — an item both variants pass (or both fail) says nothing
about their difference — and the unpaired test dilutes those `n_d` items
across all `n`, mis-stating the variance (usually conservative, but not
reliably, and the estimated correlation structure is simply wrong). With
`n = 50` items and typical agreement rates, `n_d` might be 10–15; the
unpaired test behaves as if it had 50 independent observations per arm.

**CI for Δ.** Primary: cluster bootstrap (§2.1). With single judgments this
is a bootstrap over ±1/0 values — valid, but discrete; when `n_d < 10` the
interval is unstable and the small-sample rule in §2.4 applies. Analytic
alternatives for the paired risk difference (Wald with continuity
correction, Agresti–Min add-2, Tango score) — FLAGGED F6: Tango has the best
coverage in the literature but requires an iterative solve; Agresti–Min is
trivial and decent. The spec keeps the bootstrap as primary for consistency
with clustering support; the validation suite measures its coverage at
small `n_d` and publishes it, and if it is unacceptable, Agresti–Min becomes
the small-sample fallback — do not implement a fallback until the simulation
says it is needed.

**Assumptions and failure modes.** Same as scenario A minus position issues
(no side-by-side presentation). Pass criteria drift within a run (judge
prompt edited midway) invalidates pooling — `run_id` boundaries exist
precisely so this cannot happen silently.

**Report.** Δ̂ with CI, p, pass rates per variant, the 2×2 concordance table
(n₊₊, n₊₋, n₋₊, n₋₋), n, C, and explicitly `n_d` — a reader who sees
`n_d = 8` knows the run was underpowered regardless of the p-value.

---

## 6. Scenario D — k variants compared simultaneously

**Data.** k variants, same items; outcome type from A, B, or C.

**Design decision first: what question is being asked?**

1. *Many-to-one:* k−1 candidates vs one designated control (current
   production prompt). The control MUST be designated before looking at
   results.
2. *All-pairs:* all k(k−1)/2 contrasts.
3. *Pick-the-best:* a selection problem, not a testing problem — see the
   winner's-curse paragraph below.

**Estimator.** Each pairwise contrast is computed exactly as in scenarios
A/B/C (same reduction, same engine). No omnibus test (Friedman,
repeated-measures ANOVA) is required or used: omnibus-then-post-hoc gating
answers "is anything different?" when the decision is always "which one do I
ship?", costs power, and provides no additional error control beyond the
correction below (DECISION D7; the omnibus is available in scipy if a user
insists, but it is not part of this spec).

**Multiplicity — DECISION D6.**

- **Family definition (§6.1):** the family is all *primary* comparisons
  reported from one run. Many-to-one: k−1 tests. All-pairs: k(k−1)/2.
  Secondary/exploratory analyses are labeled as such, uncorrected, and no
  ship/no-ship claim may cite them.
- **Correction: Holm step-down**, controlling family-wise error rate (FWER).
  Sort p-values ascending; adjusted
  `p̃_(j) = min(1, max_{k ≤ j} (m − k + 1) · p_(k))` for m tests. Holm
  dominates Bonferroni, requires no dependence assumptions, and FWER is the
  right criterion here because the use case is a small number of
  confirmatory ship decisions, where any single false "A beats B" claim is
  the failure mode this library exists to prevent.
- **When FDR instead:** screening many variants (say k > 10) where the
  output is a shortlist for further testing, not a ship decision.
  Benjamini–Hochberg (`p̃_(j) = min over k ≥ j of (m/k) · p_(k)`, capped
  at 1) is then defensible — but the report must say "controls expected
  proportion of false discoveries, not the chance of any false claim," and
  the shortlist must be confirmed on fresh data. The choice FWER-vs-FDR
  MUST be declared before the run.
- **FLAGGED F7 — resampling-based max-T** (Westfall–Young): reuse the §2.2
  engine, flipping cluster signs *once per iteration shared across all
  contrasts*, and compare each observed `|T_j|` to the distribution of
  `max_j |T*_j|`. This exploits the (strong, positive) dependence between
  contrasts sharing a variant and is more powerful than Holm while still
  controlling FWER. It is a natural extension of the existing engine but
  adds implementation surface; deferred, Holm is normative. Dunnett's test
  for many-to-one is the classical parametric analogue — same remark.
- **FLAGGED F8 — CIs under Holm.** Holm adjusts tests, not intervals. The
  spec reports Bonferroni-level CIs (each at `1 − α/m`) alongside
  Holm-adjusted p-values, and discloses that a Holm rejection may not match
  a Bonferroni CI excluding zero. This mismatch is inherent, not a bug; do
  not "fix" it by reporting unadjusted CIs next to adjusted p-values.

**Winner's curse — must appear in any k-variant report.** The empirical best
of k variants is biased upward: conditional on being selected as max, its
Δ̂ overstates its true Δ, and the more variants and the noisier the
measurement, the worse the bias. Consequences the spec enforces:

1. The selected winner's reported estimate carries the caveat that it is a
   post-selection estimate and biased upward.
2. A ship decision based on "best of k" SHOULD be confirmed with a fresh run
   (new items or at minimum new generations/judgments) testing only
   winner-vs-control — that confirmation is scenario A/B/C with no
   multiplicity burden and an unbiased estimate.
3. Testing "empirical best vs the rest" *on the same data that selected it*
   is circular and MUST NOT be reported as a valid test.

**Why the naive choice is wrong.** Running all pairs uncorrected at α = 0.05
with k = 5 variants is 10 tests; under a complete null the chance of at
least one false win is ≈ 40%. Combined with winner's curse, the standard
"eval leaderboard" workflow — run everything, publish the best, quote its
unadjusted p-value and its raw Δ̂ — is wrong three separate ways at once.

**Report.** The full contrast matrix (all computed contrasts, not just
significant ones), adjusted p-values, Bonferroni-level CIs, the declared
family and correction method, and the winner's-curse caveat where relevant.

---

## 7. Scenario E — comparing against a fixed baseline over time

**Data.** A frozen baseline variant; candidate runs at times t = 1, 2, …
each judged against it. Two distinct goals hide here and need different
treatment: (i) *per-run decision* — "is candidate_t better than baseline?";
(ii) *monitoring* — "alarm if the system regresses."

### 7.1 Design rules (these do more work than any test)

1. **Baseline responses may be cached; baseline judgments may not.** The
   judge drifts (model updates, prompt edits, even provider-side changes).
   Every run MUST re-judge the baseline's frozen responses in the same
   judging batch, same `judge_model` snapshot, same config as the
   candidate. Comparing fresh candidate judgments to months-old cached
   baseline judgments confounds "candidate improved" with "judge got more
   generous" and is the most common way tracking dashboards lie.
2. **Judge identity must be pinned.** `judge_model` MUST be an immutable
   snapshot identifier, not an alias that silently updates (§12). When the
   judge is upgraded intentionally, both arms are re-judged and the
   discontinuity is marked on any time series.
3. **Item bank versioning.** If items are added/removed over time, each run
   compares on its own bank (fine, scenario A/B/C per run), but *cross-run*
   comparisons of the tracking metric are only paired — and only meaningful
   — on the intersection of items. Report which bank version each run used.

### 7.2 Per-run decision

Each run is exactly scenario A, B, or C. If the question is "did we
regress?", the appropriate test is one-sided (DECISION D1 allows pre-declared
one-sided tests at α = 0.05 — note that one-sided sign-flip p-values carry
O(γ/√C) level error when the item differences are materially skewed and the
truth is mean-zero-but-asymmetric; the F12 caveat MUST accompany one-sided
regression reports in that regime, see §2.2); if the question is "is the candidate not worse
than baseline by more than δ", that is non-inferiority: pre-register the
margin δ (in estimand units — e.g., 3 points of pass rate) and claim
non-inferiority iff the CI lower bound exceeds −δ. The margin is a product
decision, not a statistical one; the spec only requires that it be fixed
before the run.

### 7.3 Repeated testing over time — OPEN O1

Testing "candidate vs baseline" at α every run guarantees false alarms in
the long run: after 20 independent-null runs, P(≥1 false alarm) ≈ 64%. This
is sequential testing, and there is no assumption-free fix; three defensible
regimes, in increasing order of rigor and cost:

- **(a) Per-run α with honest framing** (pragmatic default). Each run's
  claim is per-run; the report of any time series MUST state the expected
  false-alarm count (`runs × α`) alongside any flagged runs. No claim of
  the form "no regression across the quarter" may be made from per-run
  tests.
- **(b) Alpha spending over a planned horizon.** If decisions are made at K
  planned looks (e.g., weekly for a quarter, K = 13), spend α across looks
  (Pocock: equal per-look thresholds; O'Brien–Fleming: strict early, loose
  late; both available via standard formulas). Requires committing to K in
  advance; adding looks invalidates it.
- **(c) Anytime-valid confidence sequences.** Because every outcome here is
  bounded (`d_i ∈ [−1, 1]`), empirical-Bernstein confidence sequences
  (Waudby-Smith & Ramdas, 2021+) give intervals valid at *every* time
  simultaneously — peek freely, stop whenever. This is the principled
  answer for continuous monitoring. Cost: wider intervals per look, an
  additional estimator whose formulas must be transcribed carefully from
  the literature and validated by simulation like everything else, and a
  clustering story (apply at cluster level) that needs care.

The spec does not commit among these because the right choice depends on how
the harness is actually operated (scheduled decision points vs continuous
dashboard), which is a product question. Implement (a) first with its
mandatory framing; implement (c) when monitoring becomes a real use case; do
not implement (b) unless someone actually has fixed decision points. Any
implementation of (c) MUST cite the exact paper/equation implemented and add
a validation study of its time-uniform coverage before use.

**Why the naive choice is wrong.** The naive dashboard tests every run at
α = 0.05, alarms on any p < 0.05, and compares against stale cached
judgments. Failure (1): guaranteed eventual false alarm — "sampling to a
foregone conclusion." Failure (2): judge drift shows up as system
regression (or masks one). Failure (3): when the item bank grew between
runs, the metric moved because the items changed, not the system.

---

## 8. Judge nondeterminism as a variance component

A judge at temperature > 0 (or an API that is nondeterministic even at
temperature 0 — common) returns different verdicts on identical input.
This is *within-response* variance, distinct from item variance, and the two
have opposite remedies: judge noise shrinks with replicate judgments, item
variance only shrinks with more items.

### 8.1 Model

For the item-level difference:
`Var(d_i) = σ²_B + σ²_W,i`, where `σ²_B` is the between-item variance of the
true (expected-over-judge) difference, and `σ²_W,i` the within-item noise
from judging and generation sampling. With `r` responses per side per item
and `m` judge calls per response:

`σ²_W = 2·(σ²_G / r + σ²_J / (r·m))`

with `σ²_J` = judge-call variance (same response, replicate calls) and
`σ²_G` = generation variance (same item and variant, different responses;
identifiable only when both `r ≥ 2` on some items *and* judge replicates
exist to subtract σ²_J). The factor 2 is the two sides of the difference.
For pairwise preference (one call judges the pair), drop the factor 2 and
read σ²_J as the pairwise-call variance.

### 8.2 Estimation (normative)

- **σ̂²_J** — pooled within-response variance over all responses with
  `m_r ≥ 2` judge calls:
  `σ̂²_J = Σ_r Σ_j (y_rj − ȳ_r)² / Σ_r (m_r − 1)`.
  Unbiased under any imbalance. Requires a replicate subset: DECISION D4 —
  every run SHOULD replicate-judge (m = 2) a random subset of at least 30
  responses (or 20% of responses, whichever is smaller, but never fewer
  than 10); the subset is drawn with the run's seed. Without it, σ̂²_J is
  unavailable and the report says so.
- **σ̂²_B** — method of moments from the item-level differences:
  `σ̂²_B = max(0, s²_d − mean_i(σ̂²_W,i))` where `s²_d` is the sample
  variance (ddof=1) of `d_i` and `σ̂²_W,i` plugs σ̂²_J (and σ̂²_G when
  identified) into the §8.1 formula with item i's actual `r` and `m`.
  Truncation at 0 is reported when it triggers. (A REML mixed model —
  statsmodels `MixedLM` — is the defensible heavier alternative; FLAGGED
  F9: method of moments is normative because it is transparent and
  sufficient for the two uses below.)
- **Judge reliability**, reported per run:
  `ICC_judge = σ̂²_B / (σ̂²_B + σ̂²_W)`. A judge with ICC near 0 is mostly
  noise at the current replicate level — the run measures the judge's coin
  flips, not the variants.

Caveat for pairwise data: within-item spread across the two presentation
orders contains any item-specific position sensitivity in addition to judge
noise, so σ̂²_J estimated from order-pairs is an upper bound on pure judge
noise. Use same-order replicates for a clean estimate.

### 8.3 What replicates are worth (budget guidance, reported by the power tool)

Precision of the run: `Var(Δ̂) ≈ (σ²_B + σ²_W(m)) / n` (singleton clusters).
Under a fixed total judgment budget `N = n·m` with items free to add,
`Var(Δ̂) = (m·σ²_B + σ²_W-terms) / N` is **minimized at m = 1**: more items
always beat replicate judgments for estimating Δ. Replicates are for (i)
estimating σ²_J itself (D4's subset), (ii) when items are genuinely scarce
or expensive, and (iii) pairwise counterbalancing, which needs both orders
(effectively m = 2 per pair — this is a validity requirement, not a
precision optimization, and takes priority). The power tool (§9) accepts
(n, r, m) and reports the tradeoff explicitly rather than hard-coding this
rule.

Validity reminder: none of this section is needed for the tests in §2 to be
valid — item-level analysis absorbs judge noise automatically. It exists so
reports can say *how much* of the observed variance is the judge, and so
budget decisions are made with eyes open.

---

## 9. A priori power analysis and minimum detectable effect

Every run SHOULD be preceded by a power computation, and every report MUST
include the achieved MDE (the effect the run could have detected at 80%
power given its actual n, C, and variance estimates) — this is the number
that makes "we found no significant difference" honest: "no difference" at
MDE = 0.21 means "we couldn't have seen anything smaller than a landslide."

### 9.1 Analytic formulas (normative for planning; validated by §11)

All formulas below are normal approximations, then refined by iterating with
t quantiles (`df = n − 1`, or `C − 1` under clustering) until n is stable —
two iterations suffice; implementers MUST iterate to convergence
(`|n_new − n_old| < 1`).

**Scalar / general (paired mean):**
`n = (z_{1−α/2} + z_{1−power}′)² · σ_d² / Δ²` where `z′ = z_{power}`
(e.g., 1.96 + 0.84 at α = .05, power .80), `σ_d` = SD of item-level
differences. MDE inverts: `MDE = (z_{1−α/2} + z_{power}) · σ_d / √n`.
σ_d comes from a pilot run or a stated assumption; the report records which.

**Pairwise preference:** same formula on the `s_i` scale
(H0 at 0.5, effect `θ − 0.5`, σ = SD of `s_i`). Worst case σ = 0.5
(every item all-or-nothing, no ties); ties and replicate averaging shrink
it. With no pilot, use 0.4 as the planning default and label it an
assumption (DECISION D5).

**Binary paired (Connor 1987):** with discordance rate `ψ = p₊ + p₋` and
`Δ = p₊ − p₋`:
`n = [z_{1−α/2}·√ψ + z_{power}·√(ψ − Δ²)]² / Δ²`.
Requires a guess of ψ — from a pilot or prior runs; ψ is as important as Δ
and must be reported as an input. Measured (§11.3, reps = 10,000): the
simulated power of the binary/Connor cell (`binary-connor-n200-psi0.3-delta0.10`)
falls 3.63 points short of this formula's prediction (analytic 73.65%,
simulated 70.02%) — within the §11.3 agreement tolerance (±3 points + MC
error ≈ ±4.3 points at this scale, so not a build failure), but not noise
either. The cause is §5/F5: the exact McNemar test's true level measures
2.08–2.79% against the nominal 5%, and a
test that spends less than its allotted α loses power relative to a formula
that assumes the nominal level is achieved. Connor's formula is not wrong;
it is answering "what n hits 80% power at a test that rejects exactly 5% of
true nulls," and the exact test undershoots that premise by construction.

**Clustering:** multiply n by the design effect
`DEFF = 1 + (m̄_c − 1)·ρ_c` (m̄_c = mean cluster size, ρ_c = within-cluster
ICC of `d_i`, estimated from pilot data or assumed and labeled).

**Replicates:** enter through `σ_d²(r, m) = σ²_B + σ²_W(r, m)` per §8.

### 9.2 Simulation-based power (normative for anything the formulas don't cover)

The general tool: specify a generative model (n, C and cluster sizes, σ²_B,
σ²_J, σ²_G, r, m, effect Δ, outcome type, tie rate), simulate R = 1,000
datasets with a seed, run the *actual* §2 pipeline on each, and report the
rejection fraction, with its binomial MC error. MDE by bisection on Δ to
hit the target power (tolerance: power within ±0.01). Any configuration
with clustering + replicates + discrete outcomes SHOULD use this rather
than stacking approximations. This tool shares its generative code with the
validation suite (§11) — one simulator, two consumers.

### 9.3 The worked example this library exists for

50 items, binary pass/fail, single judgments, no clustering, 30% discordance
(typical when variants share a base model), α = 0.05, power 0.80. Connor's
formula solved for Δ gives **MDE ≈ 0.21**: with 50 items you cannot reliably
detect anything smaller than a *21-percentage-point* pass-rate difference.
A real 5-point improvement — worth shipping — has roughly a 9% chance of
reaching significance, and (winner's curse) conditional on reaching it, its
estimate is wildly inflated. Preference version: 50 items, σ_s = 0.4 →
MDE on win rate ≈ 0.5 ± 0.16, i.e. only win rates outside [0.34, 0.66] are
detectable. This example (or the run's own equivalent) belongs in the
README and in every underpowered report.

---

## 10. Effect sizes and honest reporting

**Primary effect measure per scenario — always in interpretable units, always
with a CI:**

| Scenario | Primary | Secondary |
|---|---|---|
| A (preference) | win rate θ̂ with CI; net preference 2θ̂−1 | conditional win rate excluding ties (with tie rate) |
| B (scalar) | Δ̂ in rubric units with CI | probability of superiority `PS = P(x_iA > x_iB) + ½·P(=)` (estimated as the fraction of items, ties half) |
| C (binary) | pass-rate difference Δ̂ with CI | discordant-pair split n₊ : n₋ |

**Standardized effects (Cohen-style) — FLAGGED F10.** `d_z = Δ̂ / s_d` is
reported only as a supplement and never as the headline, for two reasons:
(1) it conflates effect and noise — adding judge replicates shrinks `s_d`
and inflates `d_z` with no change in the real effect; (2) rubric units and
pass-rate points are already interpretable, which standardization destroys.
When a standardized paired effect is reported, it is `d_z` (mean of
differences over SD of differences), named as such — not Cohen's d on the
pooled scale, which is a different quantity.

**The report block.** Every comparison emits (machine-readable, one schema):

- estimand name and definition version (spec section)
- point estimate, CI (level, method, `n_boot`, seed), p-value (test, `n_perm` or "exact"), one/two-sided
- which CI is primary at the achieved C and why (DECISION D9, §2.1), the
  non-primary interval alongside it, and the measured-coverage disclosure —
  stated as what validation measured under simulated conditions, not a
  claim about this run's own data (`extras["ci_report"]`)
- n items, C clusters, cluster size distribution, excluded items with reason counts
- outcome-specific: tie rate (A), score distributions (B), concordance table and n_d (C)
- judge: `judge_model` snapshot, replicate design (r, m), σ̂²_J, ICC_judge, order balance
- multiplicity: family, method, adjusted values (D)
- power: a priori inputs if declared; achieved MDE at 80% power — always
- caveats triggered: F-flags fired, small-sample warnings, judge-relativity note

A "significant" result reported without its MDE, or a null reported without
it, is noncompliant with this spec.

---

## 11. Validation requirements (binding on `validation/`)

Per CLAUDE.md rule 1, correctness is established by simulation, not by unit
tests. Each estimator/test ships with studies that MUST fail loudly on
drift. Minimum grid, all seeded, all in-process synthetic data:

1. **Coverage.** 95% CIs (bootstrap §2.1 and analytic §2.3): empirical
   coverage over ≥ 10,000 simulated datasets per cell. Grid: n ∈ {20, 50,
   200}; outcome ∈ {scalar-normal, scalar-skewed, binary(ψ ∈ {0.1, 0.3}),
   preference-with-ties}; clustering ∈ {none, C = 10 × 5 items with ρ ∈
   {0.2, 0.5}}. Acceptance: coverage in [93.5%, 96.5%] for unclustered
   n ≥ 50 cells (MC error at 10k reps ≈ 0.22 points, so this is a real
   tolerance, not slack). Cells outside the band do not necessarily fail
   the build for known cases (percentile at C = 10 will under-cover — F2)
   but MUST be recorded in a published coverage table; a *previously
   passing* cell leaving the band fails the build. Thresholds are
   calibration DECISIONs (D8), revisable with justification in the same
   commit.
2. **False positive rate.** Sign-flip test and McNemar cross-check under
   true nulls across the same grid: rejection rate at α = 0.05 MUST NOT
   exceed 0.05 + 3·MC-SE in any cell (anti-conservatism fails the build;
   conservatism is reported, not failed).
3. **Power agreement.** Simulated power (§9.2 machinery) vs analytic
   formulas (§9.1) on the formulas' home turf (no clustering, normal-ish):
   agreement within ±3 points, else the analytic implementation is wrong.
4. **Identities.** Cluster-robust SE with singleton clusters ≡ paired t
   (exact); sign-flip with binary singletons ≡ exact McNemar distribution;
   counterbalanced reduction removes an injected additive position effect
   exactly (bias < MC error).
5. **Variance components.** σ̂²_J and σ̂²_B recover known simulation truth
   (bias and RMSE reported) across balanced and unbalanced replicate
   designs, including the truncation-at-zero regime.
6. **Multiplicity.** Holm FWER ≤ α under complete and partial nulls, k ∈
   {3, 5, 10}, dependent contrasts (shared control arm).
7. **Winner's curse demonstration.** Selection bias of the empirical best
   of k measured and published — this doubles as documentation.

8. **Null-robustness probes (weaker mean-zero null; recorded, not
   build-failing).** Measured rejection rates of the sign-flip test when
   E[d] = 0 but d is asymmetric — outside the exchangeability null (§2.2).
   Grid: C ∈ {20, 50}; skewed mean-zero d generated by both routes:
   (a) shape-different mean-equal variants (side-balanced) and
   (b) side-unbalanced replicate structure (F11); one-sided and two-sided.
   These cells measure the F11/F12 level error empirically rather than
   assuming Edgeworth magnitudes. Because they probe a null the test does
   not claim exactness for, anti-conservatism here is a published property,
   not a build failure; only a gross-error tripwire (rejection rate ≤ 0.25
   at α = 0.05) fails the build. The exact-null cells in item 2 remain
   build-failing.

9. **Bias estimator recovery (§14.5).** Every `bias/` estimator: inject a
   known bias into simulated judgment records and recover it within Monte
   Carlo tolerance, with interval coverage checked at the same cluster
   counts as the §11.1 clustered grid (C ∈ {20, 30, 50}). Position:
   additive order effect. Verbosity: controlled padded/condensed arms
   recovering a known γ, with the manipulation check covering 0 when no
   artifact effect is injected, plus an artifact cell where the check must
   recover the injected artifact contrast, plus an observational cell with
   a known quality–length confound where the association estimator covers
   the *association* (not γ) and the injected-γ-vs-association gap is
   published. Self-preference: cross-judge design with a genuine quality
   gap; naive vs corrected published. The published demonstrations double
   as documentation (cf. item 7).

Validation studies are marked (`pytest -m validation`) and skipped by
default per CLAUDE.md.

---

## 12. Gaps in `docs/data-model.md` (resolved)

The statistics above needed the following, which the schema did not
originally provide. These were stated here per instructions rather than
silently worked around. All eight gaps are now resolved in
`docs/data-model.md`; each item is kept below for history and points at the
record and field that resolves it.

1. **Pairwise judgments don't fit "one response per row."** A preference
   verdict references *two* responses, but a record has a single
   `response_id`. Two resolutions: (a) convention — a pairwise judge call
   emits two rows sharing a `judge_call_id`, one per response, each with a
   recoded per-response outcome; (b) schema — a pairwise record with
   `response_id_first` / `response_id_second` and
   `judgment ∈ {first, second, tie}`. Either works; (b) is less error-prone
   (the pairing is structural, not reconstructed by join) — but the data
   model must pick one explicitly. Until then, §1.2's reduction is not
   implementable from records. This also resolves `response_tokens` for
   verbosity analysis, which needs *both* responses' lengths for a pairwise
   call.
   **Resolved:** option (b). `PairwiseJudgment` carries
   `response_id_first` / `response_id_second`, `tokens_first` /
   `tokens_second`, and `judgment ∈ {first, second, tie}`.
2. **`presentation_order` semantics are underdetermined.** "Which variant
   appeared first" — as a bare field on a single-response row, it cannot be
   validated against the pair. Under resolution 1(b) it becomes derivable
   (first/second are structural) and should be dropped or defined as
   derived.
   **Resolved:** dropped. `docs/data-model.md` Constraints: "`presentation_order`
   is removed. Under `PairwiseJudgment` the ordering is structural, so the
   field is redundant and unverifiable."
3. **Judgment type and scale metadata.** Nothing records whether `judgment`
   is a preference, a scalar on [1, 5], a scalar on [0, 100], or binary —
   or whether ties were permitted. Pooling, normalization (§4, §7.3), and
   ceiling detection all need `judgment_type` and `scale_min` /
   `scale_max` / `ties_allowed`, either per record or in a run-level config
   that records reference. Run-level is sufficient; it must exist and be
   immutable per `run_id`.
   **Resolved:** run-level, as anticipated. `RunConfig` carries
   `judgment_type`, `scale_min`, `scale_max`, `ties_allowed`, and is
   immutable per `run_id`.
4. **Judge sampling parameters and seed.** `gen_seed` covers generation;
   there is no analogue for the judge call. Defining σ²_J (§8) as "variance
   across replicate calls *under the same judge config*" requires knowing
   the config: `judge_seed` / judge temperature (or a `judge_config_id`).
   **Resolved:** the `judge_config_id` alternative. Both `ResponseJudgment`
   and `PairwiseJudgment` carry `judge_config_id` ("judge temperature +
   seed + params").
5. **Timestamp.** No `created_at`. Scenario E's drift diagnostics and the
   "judgments must be co-batched" rule (§7.1) cannot be audited without it.
   **Resolved:** `created_at` on `RunConfig`, `ResponseJudgment`, and
   `PairwiseJudgment`.
6. **`judge_model` must be pinned.** The field exists but nothing says it
   must be an immutable snapshot id rather than a floating alias
   ("...-latest"). Scenario E (§7.1) and within-run exchangeability (§3)
   require the pin; the data model should state it as a constraint.
   **Resolved:** `docs/data-model.md` Constraints: "`judge_model` must be
   an immutable snapshot id. Floating aliases ending in `-latest` are
   rejected at write time."
7. **Human judgments.** `agreement/` needs human labels in the same store:
   a way to mark a record as human (e.g., `judge_model = "human"` plus an
   `annotator_id`) — kappa and Krippendorff's alpha need per-annotator
   identity, not just "a human said so."
   **Resolved:** `annotator_id` on both judgment records ("human labels
   only; null for model judges"); Constraints: "Human labels use the same
   records with `annotator_id` populated. Per-annotator identity is
   required for kappa and Krippendorff's alpha."
8. **Baseline designation (scenario E).** Which `variant_id` is the frozen
   baseline is run-level configuration; fine to keep out of the record
   schema, but the run config referenced in gap 3 should carry it.
   **Resolved:** `RunConfig.baseline_variant_id` ("scenario E only; the
   frozen baseline").

---

## 13. Decision log

| ID | Section | Status | Summary |
|---|---|---|---|
| D1 | §0, §7.2 | DECISION | α = 0.05 two-sided default; one-sided allowed when pre-declared (regression/non-inferiority) |
| D2 | §1.2, §3 | DECISION | ties score 0.5 in the win rate; conditional win rate secondary; tie rate always reported |
| D3 | §1.3 | DECISION | items equally weighted (cluster-equal weighting changes the estimand; cluster sizes reported) |
| D4 | §8.2 | DECISION | replicate-judging subset (m = 2, ≥ 10–30 responses) per run to estimate σ²_J |
| D5 | §9.1 | DECISION | planning default σ_s = 0.4 for preference when no pilot exists; labeled assumption |
| D6 | §6 | DECISION | Holm/FWER for confirmatory comparisons; BH/FDR only for declared screening |
| D7 | §6 | DECISION | no omnibus test gating; direct corrected pairwise contrasts |
| D8 | §11 | DECISION | coverage/FPR acceptance bands; revisable with same-commit justification |
| D9 | §2.1, §2.3 | DECISION | analytic cluster-robust interval primary below C = 50, bootstrap primary at C ≥ 50 (measured coverage, `evalkit/stats/coverage_evidence.py`); acceptable shortfall = 1 point below nominal |
| F1 | §1.2, §3 | FLAGGED | uncounterbalanced runs carry uncorrected position bias; position × variant interaction survives even counterbalancing (measured in `bias/`) |
| F2 | §2.1 | FLAGGED | percentile bootstrap under-covers at small C / heavy skew; evidence arrived (§11.1, 10k reps) — see D9, which makes the analytic interval primary below C = 50 as the mitigation; BCa itself remains deferred |
| F3 | §4 | FLAGGED | ordinal rubric treated as interval; distribution reported; degenerate scales called out |
| F4 | §2.4 | FLAGGED | C < 20 warning is a rule of thumb, not a theorem |
| F5 | §5, §9.1 | FLAGGED | mid-p McNemar better calibrated on average but not conservative; not used. Measured (§11.2, 10k reps): exact test's true level 2.79% (ψ=0.3) / 2.08% (ψ=0.1) vs nominal 5%; this conservatism explains the Connor power cell (§11.3) falling 3.63 points short of its analytic prediction |
| F6 | §5 | FLAGGED | analytic paired-RD intervals (Tango, Agresti–Min) deferred unless simulation shows bootstrap inadequate at small n_d |
| F7 | §6 | FLAGGED | Westfall–Young max-T more powerful than Holm, same engine; deferred |
| F8 | §6 | FLAGGED | Holm p-values vs Bonferroni-level CIs can disagree; inherent, disclosed |
| F9 | §8.2 | FLAGGED | method-of-moments variance components normative; REML (MixedLM) defensible alternative |
| F10 | §10 | FLAGGED | standardized effects (d_z) supplementary only; replicate count changes d_z without changing the effect |
| F11 | §2.2 | FLAGGED | side-unbalanced (r, m) structure breaks symmetry-by-construction under skewed scores; reduction detects it at runtime and warns, prominently when |d̂ skewness| > 1 |
| F12 | §2.2, §7.2 | FLAGGED | one-sided sign-flip p-values have O(γ/√C) level error under a mean-zero-but-asymmetric null (two-sided: O(1/C)); measured by the §11.8 probes; one-sided reports with materially skewed d carry the caveat |
| O1 | §7.3 | OPEN | sequential error control regime (per-run α vs alpha spending vs anytime-valid confidence sequences) — depends on how monitoring is operated; per-run-with-framing first, no silent implementation of the others |
| D10 | §14 | DECISION | bias estimands on the additive probability scale via order-balanced recodes, ties ½; both presentation orders required for bias estimands (single-order pairs excluded and counted, unlike §1.2/F1) |
| F13 | §14.2 | FLAGGED | position bias is the additive marginal effect; position × variant interaction and non-additive position structure are not separately identified |
| F14 | §14.3 | FLAGGED | verbosity bias unidentified from observational data: only the association is reported there, and it is named "association" in every public symbol and report field; controlled-manipulation γ̂ assumes quality-preserving, artifact-free rewrites (checked by the padded-vs-condensed manipulation check) and transport to natural pairs is assumed, not identified |
| F15 | §14.4 | FLAGGED | self-preference cross-judge contrast assumes panel neutrality; panel-shared style preference and self-aligned idiosyncratic taste survive the contrast; the naive self win rate is a labeled diagnostic, never a bias estimate |

---

## 14. Bias measurement (`bias/`)

The preamble's scope limitation: judge-level systematic error is not
correctable inside `stats/`. This section defines what `bias/` measures,
exactly, and what it cannot. Every estimator reduces judgment records to an
item-level vector and hands it to the §2 engine unchanged — `bias/`
contains no resampling or permutation code of its own. All estimands are
judge-configuration-relative: a bias estimate describes the judge
configuration that produced the records, and pooling records from several
judge configurations estimates a mixture (the report lists the judge models
present and warns when there is more than one).

(This section follows the decision log so that existing §13 references in
code and documents stay valid; new D/F entries above index into it.)

### 14.1 Shared reduction: order-balanced recodes

Each estimator applies the §1.2 pipeline shape with a different call-level
recode `z ∈ {0, ½, 1}` (ties ½ throughout, consistent with D2): average `z`
within (pair, order) first, then across the two presentation orders, then
across pairs, then items, giving `g_i ∈ [0, 1]`; the engine runs on
`d_i = 2·g_i − 1` and the report maps back (self-preference runs on a
difference of two win-rate-scale scores, already in [−1, 1]). One
difference from §1.2 (DECISION D10): **both presentation orders are
required, not preferred.** In §1.2 a single-order pair still estimates the
win rate (flagged F1); here the across-order average is what cancels the
nuisance term (content preference in §14.2, position effects in
§14.3–§14.4), so pairs judged in only one order are excluded from bias
estimands and the excluded count MUST be reported.

Under each estimator's null — stated in exchangeability form, i.e. no
effect on *any* pair — the sign-flip test is exact by the §2.2 symmetry
argument: `z ↦ 1 − z` swaps the roles of the two orders, so `g_i ↦ 1 − g_i`
is distribution-preserving under the null and `d_i` is symmetric about 0.
A mean-zero-but-heterogeneous null (some pairs biased up, others down,
averaging to zero) is outside the exact null exactly as discussed in §2.2.

### 14.2 Position bias

**Definition (what a stated magnitude means).** Under the additive model
`P(prefer the first-shown response | pair p, order o) = μ_{p,o} + β_p`,
where `μ_{p,o}` is the order-free preference for whichever response order
`o` shows first (`μ_{p,U-first} = w_p(U≻V)`, `μ_{p,V-first} = w_p(V≻U)`,
with `w_p(U≻V) + w_p(V≻U) + P_p(tie) = 1`), position bias is
`β_pos = E_i[β_i]`: the average increment, in probability points, to a
response's chance of being preferred from being displayed first. A stated
`β̂_pos = 0.05` means being shown first adds 5 percentage points to a
response's preference probability, over and above content preference,
averaged over the run's pairs and items.

**Estimator.** Recode every call to preference-for-first (`first → 1`,
`second → 0`, `tie → ½`); reduce per §14.1 to `g_i`;
`β̂_pos = mean_i(g_i) − ½`. The algebra that makes this work: for a pair
judged in both orders,
`g_p = (w_p(U≻V) + w_p(V≻U) + P_p(tie))/2 + β_p = ½ + β_p` — content
preference and tie propensity cancel exactly, whatever they are. Engine on
`d = 2g − 1` (so `E[d] = 2·β_pos`); CI and SE map back by halving; the
p-value tests `β_pos = 0`.

**FLAGGED F13 — what is not identified.** `β_pos` is the additive marginal
effect. A position × variant interaction (position bias stronger for one
variant, F1) and any non-additive position structure are not separately
identified from counterbalanced preference data; they enter the item-level
variance and widen the CI without moving the point estimate. The report
carries this limit.

### 14.3 Verbosity: association vs bias

**The estimand people want is causal:** `γ_verb` = the increment in
preference probability attributable to greater length, quality held fixed.

**FLAGGED F14 — `γ_verb` is not identified from observational judgment
data.** The observable is the length–preference association
`A_len = P(longer response preferred)` (order-balanced, ties ½,
equal-length pairs excluded and counted), and
`A_len − ½ = γ_verb + (quality–length component)` with the decomposition —
including the sign of `γ_verb` — unknown. Conditioning fails (no quality
covariate exists in the record; the judge's own verdict is circular),
within-item contrasts fail (quality varies with length inside items through
the same generation process), and no credible instrument shifts length
without shifting quality (truncation destroys quality mechanically).
Therefore:

- `bias/` reports `A_len` with an engine CI as a **descriptive
  association**, never as a bias. **Naming rule (normative):** every public
  symbol and every report field for the observational quantity says
  "association"; no public symbol containing "verbosity_bias" may return
  it. An observational report MUST disclose `A_len`, the per-variant length
  distributions, and the statement that the bias/quality decomposition is
  unidentified from this data.
- A bias estimate requires a **controlled length-manipulation study**:
  content-matched pairs that differ only in length — an original response
  vs a padded/expanded rewrite, and an original vs a condensed rewrite —
  counterbalanced as usual. Then `γ̂ = Â_len − ½` on manipulated pairs,
  under assumptions: (i) the rewrite is quality-preserving; (ii) the judge
  responds to length, not to detectable rewrite artifacts.

**Manipulation check (normative for controlled runs).** Assumption (ii) is
the weakest link and is checkable rather than merely disclosed: estimate
`γ̂_padded` (original vs padded; longer = manipulated) and `γ̂_condensed`
(original vs condensed; longer = original) separately. Under a pure length
response the two agree. Artifact detection pushes them apart in opposite
directions: a judge that penalizes padding artifacts attenuates
`γ̂_padded`, one that penalizes condensing artifacts inflates
`γ̂_condensed` — both drive `γ̂_padded − γ̂_condensed` negative, and
quality-changing rewrites load on the same contrast. The report MUST show
both arm estimates and their paired per-item difference with its own CI as
a manipulation check, not only a pooled `γ̂`; a check interval away from 0
means the manipulation is detected and the pooled number is not a clean
length effect.

**What remains unidentified even then:** transportability. `γ̂` measured at
the manipulated length gaps on the manipulated item set need not equal the
judge's length effect on natural pairs (the effect may be nonlinear in the
gap and content-specific), so a run's observed `A_len` still cannot be
decomposed into bias and quality without assuming `γ` transports; and a
pairwise-manipulation `γ̂` says nothing about verbosity bias in scalar
scoring, which is a distinct quantity this spec does not define.

### 14.4 Self-preference bias

**Definition.** For judge `J` whose underlying model produced the "self"
variant's responses:
`σ_self = θ_self^{(J)} − mean_{K≠J} θ_self^{(K)}`, where `θ_self^{(X)}` is
the order-balanced win rate (§1.2, ties ½) of the self variant as scored by
judge `X` **on the same pairs**. A stated `σ̂_self = 0.08` means judge `J`
scores its own model's win rate 8 points above what the rest of the judge
panel scores for identical comparisons.

**Identification.** The naive quantity `θ_self^{(J)} − ½` confounds
self-preference with the self model's genuine quality (its own outputs may
actually be better); it MUST NOT be reported as a bias estimate and appears
only as a labeled diagnostic. The cross-judge contrast removes genuine
quality because it appears in every judge's score of the same pairs.
Estimator: `d_i = s_i^{(J)} − mean_K s_i^{(K)}` per item (judges equally
weighted; items lacking either side excluded and counted); engine on `d`
directly (win-rate scale).

**Assumption: panel neutrality.** The other judges are, on average,
unbiased toward or against the self model on these pairs. **FLAGGED F15 —
residual confounds:** (a) any panel-shared style preference correlated with
the self model's style survives the contrast and biases `σ̂_self`
one-for-one (a same-family panel is the worst case; prefer heterogeneous
panels); (b) judge-idiosyncratic taste that happens to align with the self
model's style is operationally indistinguishable from self-recognition —
the contrast attributes it to self-preference; (c) with a single other
judge, "panel neutrality" rests entirely on that judge and the report MUST
say so. Position effects cancel provided each judge saw the same
counterbalanced orders.

### 14.5 Validation

Section 11 item 9: recovery and coverage studies for every §14 estimator at
the §11.1 clustered grid's cluster counts, including a padded/condensed
artifact cell demonstrating the manipulation check recovers an injected
artifact contrast, observational cells publishing the injected-bias vs
measured-association gap, and self-preference cells publishing naive vs
corrected.

### 14.6 Report block (per bias estimate)

Quantity name (per the F14 naming rule), whether it is an identified bias
or a descriptive association, point estimate with primary CI per DECISION
D9 plus the non-primary interval and the measured-coverage disclosure,
p-value and the null tested, n items / C clusters / cluster sizes,
excluded-pair and excluded-item counts with reasons, judge models present,
the estimand's assumption list and F-flags fired, and — for controlled
verbosity runs — both arm estimates and the manipulation check.
