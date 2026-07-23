"""Paired significance test for the Step A evaluation.

BOULDER's `run_significance_test.py` compares two setups with an **unpaired**
z-test on aggregate scores. This project's design is **paired** (the same examples
in both setups), so we test the per-example differences directly: a two-sided
bootstrap over `d_i = score_a[i] - score_b[i]` yields the mean delta, a 95 %
confidence interval, and a p-value for H0: mean delta = 0. Works for any metric
(binary accuracy, precision, MAE) because it only needs the per-example scores.
"""
from __future__ import annotations

import numpy as np

from boulder.evaluation.bias_correction import significance_stars


def paired_bootstrap(
    diffs, n_bootstrap: int = 2000, alpha: float = 0.05, seed: int = 42,
) -> dict:
    """Two-sided paired bootstrap over per-example score differences.

    Returns delta (mean difference), the (1-alpha) CI, a bootstrap p-value for
    H0: delta == 0, and significance stars. CI/p are None when n < 2.
    """
    d = np.asarray(list(diffs), dtype=float)
    n = len(d)
    estimate = float(d.mean()) if n else 0.0
    if n < 2:
        return {"delta": estimate, "ci_lo": None, "ci_hi": None,
                "p_value": None, "sig": "", "n": n}

    rng = np.random.RandomState(seed)
    resampled = d[rng.randint(0, n, size=(n_bootstrap, n))].mean(axis=1)
    ci_lo = float(np.percentile(resampled, 100 * alpha / 2))
    ci_hi = float(np.percentile(resampled, 100 * (1 - alpha / 2)))
    # Two-sided bootstrap test: how often a resample mean, recentred at the null,
    # is at least as extreme as the observed delta.
    p_value = float(np.mean(np.abs(resampled - estimate) >= abs(estimate)))
    return {"delta": estimate, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p_value": p_value, "sig": significance_stars(p_value), "n": n}
