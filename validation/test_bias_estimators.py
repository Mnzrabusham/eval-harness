"""§11 item 9 / §14.5 — bias estimator recovery and interval coverage.

Inject a known bias of each type into simulated judgment records; the
estimators must recover the injected truth within Monte Carlo tolerance,
with CI coverage (bootstrap and analytic, on the reporting scale) checked
at the same cluster counts as the §11.1 clustered grid (C in {20, 30, 50}).

Published demonstrations (documentation, per §11 item 9):
- observational verbosity cells check coverage of the ASSOCIATION (their
  own estimand) and publish ``association_excess_minus_gamma`` — the
  quality-length confound, known by construction, i.e. exactly how far off
  anyone treating the association as a bias would be (F14);
- self-preference cells publish the naive theta_self^(J) - 1/2, which
  recovers quality + bias, not bias (F15);
- the verbosity-controlled-C{20,30,50} cells inject NO artifact
  (artifact_padded = artifact_condensed = 0, the simulator's default), so
  gamma_padded and gamma_condensed agree by construction and
  manipulation_check truth is exactly 0 — this is the clean/specificity
  half of §11 item 9's "manipulation check covering 0 when no artifact
  effect is injected", checked at the same C in {20, 30, 50} as gamma
  recovery itself;
- a separate artifact cell requires the same padded-vs-condensed
  manipulation check to instead recover an injected artifact contrast
  (§14.3) — together the two show the check has both power (artifact cell)
  and specificity (clean cells) rather than firing on everything.

extra_low allowances per C mirror test_coverage's clustered cells, widened
by 0.01 because the item-level scores here are discrete (one call per
pair/order), which coarsens the percentile bootstrap the way §11.1's
binary cells do.
"""

import numpy as np
import pytest

from evalkit.bias import (
    position_bias,
    self_preference_bias,
    verbosity_association,
    verbosity_bias_controlled,
)
from evalkit.stats.simulate import (
    simulate_position_bias_records,
    simulate_self_preference_records,
    simulate_verbosity_controlled_records,
    simulate_verbosity_observational_records,
)
from validation._tolerances import coverage_lower_bound, publish, rep_seeds

pytestmark = pytest.mark.validation

N_BOOT = 2000
N_PERM = 199  # permutation p not under test here; keep cheap
CLUSTER_SIZE = 5
# (C, extra_boot, extra_t): same cluster counts as §11.1's clustered cells.
CLUSTER_GRID = ((20, 0.045, 0.03), (30, 0.035, 0.025), (50, 0.025, 0.02))

BETA = 0.06        # injected position bias
GAMMA = 0.10       # injected verbosity bias (controlled) / length effect (observational)
CONFOUND = 0.08    # injected quality-length confound (observational cells)
SIGMA_SELF = 0.08  # injected self-preference bias
QUALITY_DELTA = 0.05  # genuine quality edge of the self model


def _measured(quantity, truth, report):
    return dict(quantity=quantity, truth=truth, estimate=report.estimate,
                boot_ci=report.bootstrap_ci, analytic_ci=report.analytic_ci)


def _scalar(quantity, truth, value):
    """A published-and-asserted scalar with no interval (diagnostics)."""
    return dict(quantity=quantity, truth=truth, estimate=value,
                boot_ci=None, analytic_ci=None)


def _run_cells(cells, reps, base_seed):
    rows, failures = [], []
    for ci, cell in enumerate(cells):
        data_seeds, eng_seeds = rep_seeds(base_seed + ci, reps)
        acc: dict[str, dict] = {}
        for j in range(reps):
            records, truth = cell["make"](int(data_seeds[j]))
            for q in cell["measure"](records, truth, int(eng_seeds[j])):
                slot = acc.setdefault(q["quantity"], dict(
                    truth=q["truth"], ests=[], hit_b=0, hit_t=0,
                    with_ci=q["boot_ci"] is not None))
                slot["ests"].append(q["estimate"])
                if q["boot_ci"] is not None:
                    slot["hit_b"] += q["boot_ci"][0] <= q["truth"] <= q["boot_ci"][1]
                    slot["hit_t"] += q["analytic_ci"][0] <= q["truth"] <= q["analytic_ci"][1]
        for quantity, slot in acc.items():
            ests = np.asarray(slot["ests"], dtype=float)
            truth = slot["truth"]
            mean_est = float(np.mean(ests))
            bias = mean_est - truth
            tol = 3.0 * float(np.std(ests, ddof=1)) / np.sqrt(reps) + 0.002
            row = dict(cell=cell["name"], quantity=quantity, truth=f"{truth:.4f}",
                       mean_est=f"{mean_est:.4f}", bias=f"{bias:+.4f}",
                       recovery_tol=f"{tol:.4f}")
            if abs(bias) > tol:
                failures.append(f"{cell['name']}/{quantity}: recovery bias "
                                f"{bias:+.4f} exceeds tolerance {tol:.4f}")
            if slot["with_ci"]:
                cov_b = slot["hit_b"] / reps
                cov_t = slot["hit_t"] / reps
                lo_b = coverage_lower_bound(reps, extra_low=cell["extra_boot"])
                lo_t = coverage_lower_bound(reps, extra_low=cell["extra_t"])
                row.update(boot_coverage=f"{cov_b:.4f}", boot_lower_bound=f"{lo_b:.4f}",
                           analytic_coverage=f"{cov_t:.4f}", analytic_lower_bound=f"{lo_t:.4f}")
                if cov_b < lo_b:
                    failures.append(f"{cell['name']}/{quantity}: bootstrap coverage "
                                    f"{cov_b:.4f} < {lo_b:.4f}")
                if cov_t < lo_t:
                    failures.append(f"{cell['name']}/{quantity}: analytic coverage "
                                    f"{cov_t:.4f} < {lo_t:.4f}")
            else:
                row.update(boot_coverage="", boot_lower_bound="",
                           analytic_coverage="", analytic_lower_bound="")
            rows.append(row)
    return rows, failures


def test_position_bias_recovery(reps):
    cells = [dict(
        name=f"position-C{C}", extra_boot=eb, extra_t=et,
        make=lambda s, C=C: simulate_position_bias_records(
            n_clusters=C, cluster_size=CLUSTER_SIZE, beta=BETA, seed=s),
        measure=lambda recs, truth, s: [_measured(
            "beta_pos", truth["beta"],
            position_bias(recs, seed=s, n_boot=N_BOOT, n_perm=N_PERM))],
    ) for C, eb, et in CLUSTER_GRID]
    rows, failures = _run_cells(cells, reps, 920_000)
    publish("bias_position", reps, rows)
    assert not failures, "position-bias validation drifted:\n" + "\n".join(failures)


def _controlled_measure(recs, truth, s):
    rep = verbosity_bias_controlled(
        recs, original_variant="orig", padded_variant="padded",
        condensed_variant="condensed", seed=s, n_boot=N_BOOT, n_perm=N_PERM)
    return [
        _measured("gamma_pooled", truth["gamma_pooled"], rep.gamma_pooled),
        _measured("gamma_padded", truth["gamma_padded"], rep.gamma_padded),
        _measured("gamma_condensed", truth["gamma_condensed"], rep.gamma_condensed),
        _measured("manipulation_check", truth["check"], rep.manipulation_check),
    ]


def _observational_measure(recs, truth, s):
    rep = verbosity_association(recs, seed=s, n_boot=N_BOOT, n_perm=N_PERM)
    return [
        _measured("association", truth["association"], rep),
        # The confound is known by construction: treating the association
        # excess as a bias estimate would overstate gamma by exactly
        # quality_confound (F14). Published and asserted as a recovery of
        # the confound itself.
        _scalar("association_excess_minus_gamma", truth["quality_confound"],
                rep.estimate - 0.5 - truth["gamma"]),
    ]


def test_verbosity_recovery(reps):
    cells = []
    for C, eb, et in CLUSTER_GRID:
        # No artifact injected (artifact_padded = artifact_condensed = 0,
        # the default): this is the clean/specificity cell for the
        # manipulation check (§11 item 9) — gamma_padded and gamma_condensed
        # agree by construction and manipulation_check truth is 0, checked
        # alongside gamma_pooled recovery at this same C.
        cells.append(dict(
            name=f"verbosity-controlled-C{C}", extra_boot=eb, extra_t=et,
            make=lambda s, C=C: simulate_verbosity_controlled_records(
                n_clusters=C, cluster_size=CLUSTER_SIZE, gamma=GAMMA, seed=s),
            measure=_controlled_measure))
        cells.append(dict(
            name=f"verbosity-observational-C{C}", extra_boot=eb, extra_t=et,
            make=lambda s, C=C: simulate_verbosity_observational_records(
                n_clusters=C, cluster_size=CLUSTER_SIZE, gamma=GAMMA,
                quality_confound=CONFOUND, seed=s),
            measure=_observational_measure))
    # Artifact cell (§11 item 9): the manipulation check must recover an
    # injected artifact contrast, truth = -(0.05 + 0.03) = -0.08.
    cells.append(dict(
        name="verbosity-artifact-C50", extra_boot=CLUSTER_GRID[2][1],
        extra_t=CLUSTER_GRID[2][2],
        make=lambda s: simulate_verbosity_controlled_records(
            n_clusters=50, cluster_size=CLUSTER_SIZE, gamma=GAMMA,
            artifact_padded=0.05, artifact_condensed=0.03, seed=s),
        measure=_controlled_measure))
    rows, failures = _run_cells(cells, reps, 930_000)
    publish("bias_verbosity", reps, rows)
    assert not failures, "verbosity validation drifted:\n" + "\n".join(failures)


def _self_measure(recs, truth, s):
    rep = self_preference_bias(
        recs, self_judge="judge-self", self_variant="self",
        other_variant="other", seed=s, n_boot=N_BOOT, n_perm=N_PERM)
    return [
        _measured("sigma_self", truth["sigma_self"], rep),
        # Naive diagnostic recovers quality + bias, not bias (F15):
        # published next to the corrected estimate as documentation.
        _scalar("naive_theta_excess", truth["naive_theta_excess"],
                rep.diagnostics["naive_self_theta_minus_half"]),
    ]


def test_self_preference_recovery(reps):
    cells = [dict(
        name=f"self-preference-C{C}", extra_boot=eb, extra_t=et,
        make=lambda s, C=C: simulate_self_preference_records(
            n_clusters=C, cluster_size=CLUSTER_SIZE, sigma_self=SIGMA_SELF,
            quality_delta=QUALITY_DELTA, seed=s),
        measure=_self_measure,
    ) for C, eb, et in CLUSTER_GRID]
    rows, failures = _run_cells(cells, reps, 940_000)
    publish("bias_self_preference", reps, rows)
    assert not failures, "self-preference validation drifted:\n" + "\n".join(failures)
