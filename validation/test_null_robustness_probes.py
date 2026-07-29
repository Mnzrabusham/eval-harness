"""§11.8 — Null-robustness probes: the WEAKER mean-zero null (recorded, not
build-failing).

These cells measure the sign-flip test's actual rejection rate when
E[d] = 0 but d is asymmetric — OUTSIDE the §2.2 exchangeability null, where
the test claims no exactness. Both realistic routes are probed (§2.2):

  shape-diff       — variants differ in distribution with equal means
                     (side-balanced; d = centered gamma, skewness ~ +1.4)
  side-unbalanced  — identical skewed score distribution, unbalanced
                     replicate structure (F11; d = mean-of-3 minus 1 draw,
                     negative skew)

Grid: C in {20, 50} x {two-sided, greater, less}. Expectation from theory
(F12): two-sided error O(1/C), one-sided error O(skew/sqrt(C)),
anti-conservative into the long tail. The point of this file is to MEASURE
those rates rather than assume the Edgeworth magnitudes; results are
published. Only a gross-error tripwire (rate <= 0.25) fails the build; the
exact-null cells in test_false_positive_rate.py remain build-failing.
"""

import pytest

from evalkit.stats import run_engine
from evalkit.stats.simulate import simulate_d_meanzero_skewed
from validation._tolerances import mc_se, publish, rep_seeds

pytestmark = pytest.mark.validation

ALPHA = 0.05
N_PERM = 2000
N_BOOT = 100
TRIPWIRE = 0.25  # gross-error only (§11.8); measured rates are the deliverable


def test_meanzero_asymmetric_null_probes(reps):
    rows, gross = [], []
    idx = 0
    for kind in ("shape-diff", "side-unbalanced"):
        for C in (20, 50):
            data_seeds, eng_seeds = rep_seeds(930_000 + idx, reps)
            idx += 1
            # One simulation stream per (kind, C); all three alternatives are
            # evaluated on the same datasets with the same engine seeds.
            rates = {"two-sided": 0, "greater": 0, "less": 0}
            for j in range(reps):
                d, clusters = simulate_d_meanzero_skewed(kind=kind, n_items=C,
                                                         seed=int(data_seeds[j]))
                for alt in rates:
                    res = run_engine(d, clusters, seed=int(eng_seeds[j]),
                                     n_boot=N_BOOT, n_perm=N_PERM, alternative=alt)
                    rates[alt] += res.p_value <= ALPHA
            for alt, hits in rates.items():
                rate = hits / reps
                rows.append(dict(kind=kind, C=C, alternative=alt,
                                 rejection_rate=f"{rate:.4f}",
                                 mc_se=f"{mc_se(max(rate, ALPHA), reps):.4f}",
                                 nominal=ALPHA,
                                 note="probes weaker mean-zero null; recorded, not build-failing"))
                if rate > TRIPWIRE:
                    gross.append(f"{kind} C={C} {alt}: rate {rate:.4f} > tripwire {TRIPWIRE}")
    publish("null_robustness_probes", reps, rows)
    assert not gross, (
        "Gross-error tripwire (§11.8): rejection under the mean-zero probe "
        "exceeded 0.25, far beyond the F12 theory:\n" + "\n".join(gross)
    )
