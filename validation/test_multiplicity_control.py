"""§11.6 — Holm controls FWER and BH controls FDR at the stated level under
complete and partial nulls with DEPENDENT contrasts (k variants sharing a
control arm), k in {3, 5, 10}.

Each replication simulates per-item scores for k variants on the same items,
forms the k-1 paired contrasts against the designated control (dependent
through the shared control scores), runs the actual §2 sign-flip test per
contrast, and applies the §6 corrections.
"""

import numpy as np
import pytest

from evalkit.stats import benjamini_hochberg, holm, run_engine
from validation._tolerances import fpr_bound, publish, rep_seeds

pytestmark = pytest.mark.validation

ALPHA = 0.05
N_ITEMS = 40
EFFECT = 0.6  # detectable at n=40 so partial-null power is non-trivial
N_PERM = 999
N_BOOT = 50


def _contrast_pvalues(rng_seed, eng_seeds, deltas):
    """Simulate k variants' item scores; return p-values for k-1 contrasts
    against variant 0 (the control), dependent through the shared control."""
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    k = len(deltas)
    x = rng.normal(0.0, 1.0, (k, N_ITEMS)) + np.asarray(deltas)[:, None]
    pvals = []
    for j in range(1, k):
        res = run_engine(x[j] - x[0], seed=int(eng_seeds[j - 1]),
                         n_boot=N_BOOT, n_perm=N_PERM)
        pvals.append(res.p_value)
    return np.asarray(pvals)


@pytest.mark.parametrize("k", [3, 5, 10])
def test_holm_fwer_and_bh_fdr(k, reps):
    m = k - 1
    bound = fpr_bound(reps, ALPHA)
    rows, failures = [], []
    for scenario in ("complete-null", "partial-null"):
        deltas = [0.0] * k
        if scenario == "partial-null":
            # Half the contrasts (rounded down, at least 1) get a real effect.
            n_eff = max(1, m // 2)
            for j in range(k - n_eff, k):
                deltas[j] = EFFECT
        null_idx = np.asarray([j for j in range(m) if deltas[j + 1] == 0.0])
        data_seeds, *eng_streams = rep_seeds(950_000 + 17 * k + (scenario == "partial-null"),
                                             reps, streams=1 + m)
        holm_fw = 0
        bh_fdp = 0.0
        holm_power_hits = 0
        power_denom = 0
        for i in range(reps):
            eng_seeds = [int(stream[i]) for stream in eng_streams]
            p = _contrast_pvalues(int(data_seeds[i]), eng_seeds, deltas)
            adj_holm = holm(p)
            adj_bh = benjamini_hochberg(p)
            holm_fw += bool(np.any(adj_holm[null_idx] <= ALPHA))
            n_rej = int(np.count_nonzero(adj_bh <= ALPHA))
            v = int(np.count_nonzero(adj_bh[null_idx] <= ALPHA))
            bh_fdp += v / max(1, n_rej)
            nonnull = [j for j in range(m) if j not in null_idx]
            if nonnull:
                power_denom += len(nonnull)
                holm_power_hits += int(np.count_nonzero(adj_holm[nonnull] <= ALPHA))
        fwer = holm_fw / reps
        fdr = bh_fdp / reps
        power = holm_power_hits / power_denom if power_denom else float("nan")
        rows.append(dict(k=k, scenario=scenario, holm_fwer=f"{fwer:.4f}",
                         bh_fdr=f"{fdr:.4f}", bound=f"{bound:.4f}",
                         holm_power_nonnull=f"{power:.4f}"))
        if fwer > bound:
            failures.append(f"k={k} {scenario}: Holm FWER {fwer:.4f} > {bound:.4f}")
        if fdr > bound:
            failures.append(f"k={k} {scenario}: BH FDR {fdr:.4f} > {bound:.4f}")
    publish(f"multiplicity_k{k}", reps, rows)
    assert not failures, "Multiplicity control drift:\n" + "\n".join(failures)
