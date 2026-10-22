"""
Statistical utilities for repeated cross-validation comparisons.

Implements:
1. Nadeau-Bengio corrected paired t-test for repeated k-fold CV.
2. Holm-Bonferroni step-down correction for multiple testing.
3. Wilcoxon signed-rank paired test.
"""

from typing import Sequence
import numpy as np
from scipy import stats


def nadeau_bengio_correction_factor(k: int, n_splits: int = 5) -> float:
    """
    Computes Nadeau-Bengio correction factor:
        c = (1 / k) + (n_test / n_train)
    where n_test / n_train = 1 / (n_splits - 1).
    For 5-fold CV, n_test / n_train = 1/4 = 0.25.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if n_splits <= 1:
        raise ValueError("n_splits must be greater than 1.")
    return (1.0 / k) + (1.0 / (n_splits - 1))


def nadeau_bengio_paired_test(
    diffs: Sequence[float] | np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 3,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    Computes Nadeau-Bengio corrected paired t-test for repeated k-fold CV.

    Correction factor:
        c = 1/k + (n_test / n_train) = 1/k + 1/(n_splits - 1)
    For 5-fold CV: n_test/n_train = 0.25.
    Total folds k = len(diffs).

    Parameters:
        diffs: Sequence of paired fold differences (strategy - baseline).
        n_splits: Number of folds per repeat (default 5).
        n_repeats: Number of repeats (default 3).
        alpha: Significance level for confidence interval (default 0.05 -> 95% CI).

    Returns:
        (mean_diff, ci_lower, ci_upper, p_value)
    """
    d = np.asarray(diffs, dtype=float)
    k = len(d)
    if k == 0:
        raise ValueError("diffs cannot be empty.")

    mean_d = float(np.mean(d))
    if k <= 1:
        return mean_d, mean_d, mean_d, 1.0

    s2 = float(np.var(d, ddof=1))
    c = nadeau_bengio_correction_factor(k=k, n_splits=n_splits)
    var_corrected = c * s2
    se_corrected = np.sqrt(max(0.0, var_corrected))

    df = k - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    ci_lower = mean_d - t_crit * se_corrected
    ci_upper = mean_d + t_crit * se_corrected

    if se_corrected < 1e-12:
        p_val = 1.0 if np.isclose(mean_d, 0.0) else 0.0
    else:
        t_stat = mean_d / se_corrected
        p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df)))

    return mean_d, float(ci_lower), float(ci_upper), float(p_val)


def wilcoxon_test(diffs: Sequence[float] | np.ndarray) -> float:
    """
    Computes Wilcoxon signed-rank test p-value.
    Safely handles all-zero differences and small samples.
    """
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return 1.0
    if np.all(np.isclose(d, 0.0)):
        return 1.0
    try:
        res = stats.wilcoxon(d, alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return 1.0


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """
    Applies Holm-Bonferroni step-down correction to a sequence of p-values.
    Guarantees monotonicity: adjusted_p[i] <= adjusted_p[i+1] when sorted.
    """
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    if m == 0:
        return []

    order = np.argsort(p_arr)
    sorted_p = p_arr[order]

    adjusted = np.empty(m, dtype=float)
    cum_max = 0.0
    for i in range(m):
        raw_p = sorted_p[i]
        adj_p = min(1.0, (m - i) * raw_p)
        cum_max = max(cum_max, adj_p)
        adjusted[i] = cum_max

    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return [float(x) for x in out]
