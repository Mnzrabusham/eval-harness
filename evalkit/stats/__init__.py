"""stats/: estimators, tests, intervals, power, multiplicity.

Source of truth: docs/statistics-spec.md. If code here disagrees with that
document, the code is wrong or the document must be amended in the same
commit. Correctness is established by the simulation studies in
validation/, not by unit tests.
"""

from .coverage_evidence import (
    MIN_CLUSTERS_ANALYTIC_PRIMARY,
    coverage_disclosure,
    primary_ci_method,
)
from .engine import AnalyticResult, EngineResult, make_rng, run_engine
from .mcnemar import discordant_counts, mcnemar_exact
from .multiplicity import benjamini_hochberg, holm
from .reduction import (
    MATERIAL_SKEW_THRESHOLD,
    PairwiseReduction,
    ScoreReduction,
    reduce_pairwise,
    reduce_scores,
    sample_skewness,
)
from .scenarios import (
    JUDGE_RELATIVITY_CAVEAT,
    ComparisonReport,
    compare_binary,
    compare_pairwise,
    compare_scalar,
)
from .power import (
    SimulatedPower,
    design_effect,
    mde_binary_paired,
    mde_paired_mean,
    mde_simulated,
    n_binary_paired,
    n_paired_mean,
    power_binary_paired,
    power_paired_mean,
    simulated_power,
)
from .variance import (
    JudgeVariance,
    VarianceComponents,
    sigma2_judge,
    sigma2_within_item,
    variance_components,
)

__all__ = [
    "AnalyticResult",
    "ComparisonReport",
    "EngineResult",
    "JUDGE_RELATIVITY_CAVEAT",
    "JudgeVariance",
    "MATERIAL_SKEW_THRESHOLD",
    "MIN_CLUSTERS_ANALYTIC_PRIMARY",
    "PairwiseReduction",
    "ScoreReduction",
    "SimulatedPower",
    "VarianceComponents",
    "benjamini_hochberg",
    "compare_binary",
    "compare_pairwise",
    "compare_scalar",
    "coverage_disclosure",
    "design_effect",
    "discordant_counts",
    "holm",
    "make_rng",
    "mcnemar_exact",
    "mde_binary_paired",
    "mde_paired_mean",
    "mde_simulated",
    "n_binary_paired",
    "n_paired_mean",
    "power_binary_paired",
    "power_paired_mean",
    "primary_ci_method",
    "reduce_pairwise",
    "reduce_scores",
    "run_engine",
    "sample_skewness",
    "sigma2_judge",
    "sigma2_within_item",
    "simulated_power",
    "variance_components",
]
