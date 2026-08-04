# eval-harness

Statistically rigorous evaluation for LLM systems.

## Purpose

Most teams compare prompt variants on 50 examples and declare a winner. This
library makes that comparison honest: correct estimators, uncertainty on every
number, power analysis before the run, measured judge bias, and calibration
against human labels.

This is a public portfolio project. The intended reader is a skeptical
statistician or a senior engineer deciding whether to hire the author. Assume
every claim will be checked. Overclaiming is worse than a missing feature.

## Source of truth

- `docs/statistics-spec.md` defines every estimator, test, and assumption.
- `docs/data-model.md` defines the judgment record schema.

Read both before touching `stats/`, `bias/`, or `agreement/`. If a task
conflicts with either document, stop and say so rather than improvising. If a
change requires amending them, amend them in the same commit as the code.

## Structure

evalkit/
  runner/     executes variants against models; caching, concurrency, seeds
  judge/      LLM-as-judge with counterbalanced presentation
  stats/      estimators, tests, intervals, power, multiplicity correction
  bias/       position, verbosity, self-preference bias measurement
  agreement/  judge-vs-human calibration (kappa, Krippendorff's alpha)
tests/        unit tests
validation/   simulation studies proving the statistics are correct
docs/

## Non-negotiable rules

1. Statistical correctness is established by simulation, not unit tests.
   A test that checks return types and shapes is not evidence. Every estimator
   needs a simulation study in `validation/` demonstrating interval coverage,
   false positive rate under a true null, and power matching the analytic
   function. These must fail loudly if the implementation drifts.

2. No point estimate ships without an uncertainty measure.

3. Flag uncertainty instead of choosing something plausible. If more than one
   statistical approach is defensible, present the options and the tradeoff.
   A confidently wrong statistical choice is the primary failure mode of this
   project and it is invisible in code review.

4. Never silently substitute a different test than the spec specifies.

5. Do not add a dependency to make something convenient. numpy, scipy,
   statsmodels, and pytest are the core set.

6. Never run `git commit` or `git push`. Stage changes with `git add` and stop.
   The author reviews and commits manually.

## Conventions

- Python 3.11+, `.venv` in repo root.
- Every stochastic function accepts an explicit `seed`. No global random state.
- Tests and validation studies make no network calls. Simulation studies
  generate synthetic data with known ground truth in-process.
- Validation studies may be slow; mark them so `pytest` can skip them by
  default and run them explicitly.
- Setup commands in README and docs target macOS/Linux (bash). Local
  development is Windows; keep that out of public-facing docs.
- Windows + Git Bash. The venv is at `.venv/Scripts/`, not `.venv/bin/`.


## Working style

Sessions here are budget-constrained. Prefer reading the specific files named
in the task over exploring the repository. Ask before running a broad search
across the codebase. Do not refactor code you were not asked to touch.

## Status

- [x] docs/data-model.md
- [x] docs/statistics-spec.md
- [x] stats module + validation suite
- [x] judge + bias modules
- [x] runner
- [x] D4 replicate draw is item-level and mirrored across variants
      (spec §8.2 as amended); ship with a test asserting the drawn
      subset is side-balanced, so a per-response reimplementation fails
      loudly instead of just firing F11 warnings downstream
- [ ] agreement module
- [ ] position bias study
- [ ] README and writeup