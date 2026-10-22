"""
Unit tests for src/stats_utils.py.
"""

import numpy as np
import pytest
from scipy import stats

from src.stats_utils import (
    holm_bonferroni,
    nadeau_bengio_correction_factor,
    nadeau_bengio_paired_test,
    wilcoxon_test,
)


def test_nadeau_bengio_correction_factor():
    # 5-fold CV with 3 repeats (k=15): 1/15 + 1/4 = 0.31666666...
    c_15 = nadeau_bengio_correction_factor(k=15, n_splits=5)
    assert np.isclose(c_15, (1.0 / 15.0) + 0.25)

    # 5-fold CV with 1 repeat (k=5): 1/5 + 1/4 = 0.45
    c_5 = nadeau_bengio_correction_factor(k=5, n_splits=5)
    assert np.isclose(c_5, 0.45)

    with pytest.raises(ValueError):
        nadeau_bengio_correction_factor(k=0)
    with pytest.raises(ValueError):
        nadeau_bengio_correction_factor(k=5, n_splits=1)


def test_nadeau_bengio_known_synthetic_data():
    # Synthetic diffs with known sample variance
    diffs = np.array([-1.0, -2.0, -1.5, -0.5, -2.5,
                      -1.2, -1.8, -1.4, -0.8, -2.2,
                      -1.1, -1.9, -1.3, -0.7, -2.3])
    k = len(diffs)
    mean_d = float(np.mean(diffs))
    s2 = float(np.var(diffs, ddof=1))
    c = (1.0 / k) + 0.25
    expected_se = np.sqrt(c * s2)
    t_crit = float(stats.t.ppf(0.975, df=k - 1))
    exp_ci_lower = mean_d - t_crit * expected_se
    exp_ci_upper = mean_d + t_crit * expected_se

    m_d, ci_l, ci_u, p_val = nadeau_bengio_paired_test(diffs, n_splits=5, n_repeats=3)

    assert np.isclose(m_d, mean_d)
    assert np.isclose(ci_l, exp_ci_lower)
    assert np.isclose(ci_u, exp_ci_upper)
    assert ci_l < ci_u
    # Standard paired t-test se would be sqrt(s2 / k)
    std_se = np.sqrt(s2 / k)
    assert expected_se > std_se  # Correction must widen standard error and CI
    assert (ci_u - ci_l) > 2.0 * t_crit * std_se


def test_nadeau_bengio_edge_cases():
    # All zero diffs
    zeros = np.zeros(15)
    m_d, ci_l, ci_u, p_val = nadeau_bengio_paired_test(zeros)
    assert np.isclose(m_d, 0.0)
    assert np.isclose(p_val, 1.0)

    # Single value
    m_d, ci_l, ci_u, p_val = nadeau_bengio_paired_test([1.5])
    assert np.isclose(m_d, 1.5)
    assert np.isclose(p_val, 1.0)


def test_holm_bonferroni_properties():
    # Empty
    assert holm_bonferroni([]) == []

    # Single
    assert holm_bonferroni([0.02]) == [0.02]

    # Multiple p-values: test monotonicity and bounding
    raw_p = [0.005, 0.04, 0.01, 0.20, 0.03]
    adj_p = holm_bonferroni(raw_p)

    assert len(adj_p) == len(raw_p)
    # Check that sorting raw_p preserves sorted order in adj_p
    order = np.argsort(raw_p)
    sorted_raw = np.array(raw_p)[order]
    sorted_adj = np.array(adj_p)[order]
    # Check monotonicity
    for i in range(len(sorted_adj) - 1):
        assert sorted_adj[i] <= sorted_adj[i + 1]

    # Check bounds
    for p in adj_p:
        assert 0.0 <= p <= 1.0

    # Hand calculation check for [0.005, 0.01, 0.03, 0.04, 0.20]
    # m = 5
    # i=0: 0.005 * 5 = 0.025
    # i=1: max(0.025, 0.01 * 4) = max(0.025, 0.04) = 0.04
    # i=2: max(0.04, 0.03 * 3) = max(0.04, 0.09) = 0.09
    # i=3: max(0.09, 0.04 * 2) = max(0.09, 0.08) = 0.09
    # i=4: max(0.09, 0.20 * 1) = max(0.09, 0.20) = 0.20
    expected_sorted = [0.025, 0.04, 0.09, 0.09, 0.20]
    assert np.allclose(sorted_adj, expected_sorted)


def test_wilcoxon_test():
    # Consistent positive diffs
    diffs = np.array([1.0, 2.0, 1.5, 0.8, 1.2, 2.1, 1.9, 0.9, 1.4, 2.2])
    p_val = wilcoxon_test(diffs)
    assert p_val < 0.05

    # All zeros
    assert np.isclose(wilcoxon_test(np.zeros(10)), 1.0)
