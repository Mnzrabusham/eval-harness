"""The §6.4 simulation gate (docs/position-bias-study.md §6.4, spec §9.2).

Before spending API budget, the design requires proof that the *actual*
analysis pipeline -- the real §14.1/§14.2 order-balanced reduction
(``evalkit.bias._core.order_balanced_reduce``, the same function
``evalkit.bias.position.position_bias`` calls) and the real §2 engine
(``evalkit.stats.engine.run_engine``, invoked internally by
``evalkit.stats.power.simulated_power``/``mde_simulated``) -- does not lose
power the analytic formula would not predict. A gate that reimplements the
reduction or the engine as a shortcut tests nothing: it can only detect bugs
in the copy, not in the pipeline the study will actually run. This script
does not reimplement either; it only supplies raw synthetic judgment
records and lets the production code do the rest.

Two independently measured numbers, both from the same generative model:

1. **Realized sigma_g** -- mean over replications of sd(g_i) under the null
   (beta=0) design (300 prompt clusters x 2 pairs each: one decisive pair,
   one coin-flip pair, per §3.2's strata; both presentation orders; one
   call per order, matching the primary reduction's design, §5).
2. **Simulated MDE** -- ``evalkit.stats.power.mde_simulated`` bisecting the
   injected beta until simulated power hits 80%, routing every replicate
   through the real reduction and engine.

The comparator is ``mde_paired_mean`` recomputed at the realized sigma_g
(not the deliberately conservative sigma_g = 0.40 planning value, §6.1),
at Holm-worst alpha = 0.05/3 (§7, three judges), design effect
``design_effect(2, 0.2)`` = 1.2, mapped to the beta scale by halving --
exactly the construction that produced §6.2's table (verified to reproduce
those published numbers before this script was written). The gate passes
iff simulated MDE <= comparator + 0.01 (§6.4's allowance for the
discreteness of the 5-point g support and the bisection tolerance -- not
headroom for pipeline power loss).

No API calls. Exits 0 on pass, 1 on fail (or on a plumbing error), so it can
gate a launch script:

    python study/simulation_gate.py [--reps N] [--sigma-g-reps N] [--seed N]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from math import sqrt

import numpy as np

from evalkit.bias._core import order_balanced_reduce
from evalkit.stats.engine import make_rng
from evalkit.stats.power import SimulatedPower, design_effect, mde_paired_mean, mde_simulated

__all__ = [
    "GateResult",
    "make_simulate_fn",
    "realized_sigma_g",
    "run_gate",
    "simulate_records",
]

# --- design constants (docs/position-bias-study.md §3.2, §4, §5, §6.4) -----

N_PROMPTS = 300           # §4: C = 300 prompt clusters
PAIRS_PER_PROMPT = 2      # §3.2: one clear-gap pair, one near-tie pair
N_PAIRS = N_PROMPTS * PAIRS_PER_PROMPT  # 600, spec n
CLUSTER_DF = N_PROMPTS - 1              # 299, spec C - 1

TIE_RATE = 0.10                   # "a tie propensity in the plausible range"
DECISIVE_CONTENT = 0.32           # clear-gap stratum: strong, order-cancelling content preference
NEARTIE_CONTENT = 0.0             # near-tie stratum: exactly exchangeable arms (§3.2)
BETA_CLUSTER_SCALE = 0.015        # prompt-level position-effect heterogeneity (shared by a prompt's 2 pairs)
BETA_ITEM_SCALE = 0.005           # pair-level heterogeneity on top of the cluster term
# "modest heterogeneity" (§6.4); the two scales above keep every implied
# per-order probability inside (0, 1 - TIE_RATE) across the bisection range
# below -- see _check_range, which is a hard assertion, not a clip.

HOLM_WORST_ALPHA = 0.05 / 3       # §7: Bonferroni-worst Holm step, 3 judges
RHO = 0.2                         # §6.2/§6.4's planning rho
MEAN_CLUSTER_SIZE = float(PAIRS_PER_PROMPT)  # n / C = 2
DESIGN_EFFECT = design_effect(MEAN_CLUSTER_SIZE, RHO)  # 1.2, matches §6.4's stated value

GATE_TOLERANCE = 0.01             # §6.4: discreteness + bisection allowance only
TARGET_POWER = 0.8
BISECT_LO, BISECT_HI = 0.0, 0.08  # beta scale; see module docstring for margin check

DEFAULT_SEED = 20260804           # study/build_prompt_set.py's seed, reused for continuity
DEFAULT_REPS = 1000
DEFAULT_SIGMA_G_REPS = 1000


def _base(tie_rate: float) -> float:
    return (1.0 - tie_rate) / 2.0


def _check_range(p: float, tie_rate: float, label: str) -> None:
    if p < 0.0 or p + tie_rate > 1.0:
        raise ValueError(
            f"{label}: implied first-shown-win probability {p:.4f} "
            f"(+ tie_rate {tie_rate}) is out of [0, 1] -- generative model "
            f"constants need tightening for this beta range"
        )


def _draw(rng: np.random.Generator, p_first: float, tie_rate: float) -> str:
    u = rng.random()
    if u < p_first:
        return "first"
    if u < p_first + tie_rate:
        return "tie"
    return "second"


def simulate_records(*, beta: float, seed: int) -> list[dict]:
    """300 prompts x 2 pairs, both orders, one call per order.

    Matches §3.2's two strata: a clear-gap ("decisive") pair whose content
    preference is strong enough that each order's outcome is only mildly
    stochastic (low within-pair variance -- "g concentrated at 1/2" once
    order-balanced), and a near-tie ("coin-flip") pair with exactly
    exchangeable arms (zero content preference, full stochastic variance).
    ``beta`` is the injected additive position effect (§14.2), with
    prompt-level heterogeneity shared across a prompt's two pairs and
    pair-level heterogeneity on top -- both drawn uniformly, following the
    same construction ``evalkit.stats.simulate.simulate_position_bias_records``
    uses for the general position-bias validation cells.

    Record shape matches what ``evalkit.bias._core.order_balanced_reduce``
    (the real reduction) requires: ``item_id`` = pair (§4's unit pinning),
    ``source_doc_id`` = prompt (the cluster key), presentation-terms
    ``judgment``.
    """
    base = _base(TIE_RATE)
    rng = make_rng(seed)
    records: list[dict] = []
    for c in range(N_PROMPTS):
        doc = f"doc{c:05d}"
        v_c = rng.uniform(-BETA_CLUSTER_SCALE, BETA_CLUSTER_SCALE)
        for stratum, w in (("decisive", DECISIVE_CONTENT), ("neartie", NEARTIE_CONTENT)):
            item = f"i{c:05d}-{stratum}"
            pair = f"{item}-p0"
            f_i = rng.uniform(-BETA_ITEM_SCALE, BETA_ITEM_SCALE)
            beta_i = beta + v_c + f_i
            for a_first in (True, False):
                w_signed = w if a_first else -w
                p_first = base + w_signed + beta_i
                _check_range(p_first, TIE_RATE, f"{item}/{'A' if a_first else 'B'}-first")
                judgment = _draw(rng, p_first, TIE_RATE)
                records.append({
                    "item_id": item,
                    "source_doc_id": doc,
                    "pair_id": pair,
                    "variant_first": "A" if a_first else "B",
                    "variant_second": "B" if a_first else "A",
                    "judgment": judgment,
                    "judge_model": "sim-judge",
                })
    return records


def _pref_first(rec, judgment: str) -> float:
    """The §14.2 position-bias recode -- identical to
    ``evalkit.bias.position``'s private recode of the same name, reproduced
    here because it is a one-line callback the real reduction requires, not
    a piece of the reduction itself."""
    if judgment == "first":
        return 1.0
    if judgment == "second":
        return 0.0
    return 0.5


def _reduce(records: list[dict]):
    """Run the real §14.1 reduction and assert the design's shape held."""
    red = order_balanced_reduce(records, _pref_first)
    if red.n_items != N_PAIRS:
        raise ValueError(
            f"reduction produced {red.n_items} items, expected {N_PAIRS}; "
            f"the generative model should never produce single-order pairs "
            f"({red.n_pairs_single_order} were excluded) -- this is a bug in "
            f"the simulator, not a real exclusion"
        )
    return red


def make_simulate_fn(beta: float):
    """``simulate_fn(data_seed) -> (d, clusters)`` for ``mde_simulated``.

    Generation and reduction happen here, inside the callback
    ``simulated_power``/``mde_simulated`` invoke per replicate; the engine
    itself is run by that library code, not here (§9.2's seeding discipline
    keeps data seeds and engine seeds on separate streams -- see
    ``evalkit.stats.power.simulated_power``'s docstring).
    """
    def simulate_fn(data_seed: int):
        records = simulate_records(beta=beta, seed=data_seed)
        red = _reduce(records)
        d = 2.0 * red.g - 1.0
        return d, list(red.clusters)
    return simulate_fn


def realized_sigma_g(*, seed: int, reps: int) -> tuple[float, list[float]]:
    """Mean over ``reps`` replications of sd(g_i) under the null (beta=0).

    sigma_g is a property of the pair-mixture design's variance, not of the
    injected effect (§6.4: "this generative model's pair-level SD is
    sigma_g ~ 0.25 ... not the deliberately conservative planning value"),
    so it is measured once, independently of the MDE bisection's trial
    points at various beta.
    """
    state = np.random.SeedSequence(seed).generate_state(reps)
    sds = []
    for i in range(reps):
        records = simulate_records(beta=0.0, seed=int(state[i]))
        red = _reduce(records)
        sds.append(float(np.std(red.g, ddof=1)))
    return float(np.mean(sds)), sds


@dataclass(frozen=True)
class GateResult:
    realized_sigma_g: float
    simulated_mde_beta: float
    simulated_power: SimulatedPower
    comparator_beta: float
    gap: float
    passed: bool


def run_gate(*, seed: int = DEFAULT_SEED, reps: int = DEFAULT_REPS,
            sigma_g_reps: int = DEFAULT_SIGMA_G_REPS) -> GateResult:
    sigma_g, _ = realized_sigma_g(seed=seed, reps=sigma_g_reps)

    beta_mde, sim_power = mde_simulated(
        make_simulate_fn, BISECT_LO, BISECT_HI,
        target_power=TARGET_POWER, reps=reps, seed=seed, alpha=HOLM_WORST_ALPHA,
    )

    comparator_d = mde_paired_mean(
        n=N_PAIRS, sd_d=2.0 * sigma_g * sqrt(DESIGN_EFFECT),
        alpha=HOLM_WORST_ALPHA, power=TARGET_POWER, df=CLUSTER_DF,
    )
    comparator_beta = comparator_d / 2.0

    gap = beta_mde - comparator_beta
    passed = gap <= GATE_TOLERANCE

    return GateResult(
        realized_sigma_g=sigma_g,
        simulated_mde_beta=beta_mde,
        simulated_power=sim_power,
        comparator_beta=comparator_beta,
        gap=gap,
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Position-bias study launch gate (position-bias-study.md section 6.4). "
                    "See the module docstring for the full explanation."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="replications per simulated-power evaluation (mde_simulated bisects "
                             "up to 20 times, each costing this many replicates)")
    parser.add_argument("--sigma-g-reps", type=int, default=DEFAULT_SIGMA_G_REPS,
                        help="replications for the independent realized-sigma_g measurement")
    args = parser.parse_args()

    print(f"section 6.4 simulation gate -- seed={args.seed}, reps={args.reps}, "
          f"sigma_g_reps={args.sigma_g_reps}")
    print(f"design: n={N_PAIRS} pairs, C={N_PROMPTS} clusters, "
          f"mean_cluster_size={MEAN_CLUSTER_SIZE}, DEFF=design_effect({MEAN_CLUSTER_SIZE}, {RHO})="
          f"{DESIGN_EFFECT}, alpha=Holm-worst={HOLM_WORST_ALPHA:.5f} (0.05/3), "
          f"target_power={TARGET_POWER}")
    print("routing every replicate through the real reduction "
          "(evalkit.bias._core.order_balanced_reduce) and the real engine "
          "(evalkit.stats.engine.run_engine, via simulated_power/mde_simulated) "
          "-- no shortcut reimplementation.\n")

    t0 = time.monotonic()
    result = run_gate(seed=args.seed, reps=args.reps, sigma_g_reps=args.sigma_g_reps)
    elapsed = time.monotonic() - t0

    print(f"realized sigma_g (mean over {args.sigma_g_reps} reps of sd(g_i), beta=0): "
          f"{result.realized_sigma_g:.4f}")
    print(f"  (section 6.4's planning approximation was sigma_g ~ 0.25 -- not compared here; "
          f"the comparator below uses the realized value directly)")
    print(f"simulated MDE (beta scale): {result.simulated_mde_beta:.4f}")
    print(f"  achieved simulated power at that beta: "
          f"{result.simulated_power.power:.3f} +/- {result.simulated_power.mc_se:.3f} "
          f"(MC-SE, {result.simulated_power.reps} reps, target {TARGET_POWER})")
    print(f"analytic comparator (mde_paired_mean at realized sigma_g, "
          f"Holm-worst alpha, DEFF={DESIGN_EFFECT}): {result.comparator_beta:.4f}")
    print(f"gap (simulated - comparator): {result.gap:+.4f}  "
          f"(tolerance +{GATE_TOLERANCE:.2f}, per section 6.4 -- discreteness + bisection "
          f"allowance, not pipeline headroom)")
    print(f"elapsed: {elapsed:.1f}s\n")

    if result.passed:
        print(f"PASS: simulated MDE ({result.simulated_mde_beta:.4f}) <= comparator "
              f"({result.comparator_beta:.4f}) + tolerance ({GATE_TOLERANCE:.2f}) "
              f"= {result.comparator_beta + GATE_TOLERANCE:.4f}.")
        print("The pipeline is not losing power beyond what the like-for-like "
              "analytic formula predicts. Cleared to proceed to the next section 11 "
              "prerequisite -- this gate alone does not authorize launch.")
        return 0

    print(f"FAIL: simulated MDE ({result.simulated_mde_beta:.4f}) exceeds comparator "
          f"({result.comparator_beta:.4f}) + tolerance ({GATE_TOLERANCE:.2f}) "
          f"= {result.comparator_beta + GATE_TOLERANCE:.4f} by {result.gap - GATE_TOLERANCE:.4f}.")
    print("Per section 6.4 this is an implementation or design problem in the pipeline "
          "(reduction, exclusion handling, or engine), not a variance surprise -- "
          "the study MUST NOT launch until the discrepancy is explained and resolved.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
