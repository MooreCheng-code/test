"""
Unit tests for E5 Explainability and Compact Feature Selection.
"""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier

from src.explain import (
    compute_pr_auc,
    compute_three_point_log_loss,
    compute_permutation_importance,
    map_financial_dimension,
    is_directly_computable,
)


def test_compute_pr_auc():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    pr_auc = compute_pr_auc(y_true, y_score)
    assert 0.0 <= pr_auc <= 1.0
    assert pr_auc > 0.8


def test_compute_three_point_log_loss():
    y_true = np.array([0] * 90 + [1] * 10)
    scores = np.linspace(0, 1, 100)
    loss = compute_three_point_log_loss(y_true, scores)
    assert not np.isnan(loss)
    assert isinstance(loss, float)


def test_financial_dimension_mapping():
    assert "獲利能力" in map_financial_dimension("Net Income to Total Assets")
    assert "獲利能力" in map_financial_dimension("Persistent EPS in the Last Four Seasons")
    assert "槓桿與償債能力" in map_financial_dimension("Debt ratio %")
    assert "槓桿與償債能力" in map_financial_dimension("Total debt/Total net worth")
    assert "流動性" in map_financial_dimension("Current Ratio")
    assert "流動性" in map_financial_dimension("Quick Ratio")
    assert "營運效率" in map_financial_dimension("Total Asset Turnover")
    assert "成長能力" in map_financial_dimension("Total Asset Growth Rate")
    assert "現金流" in map_financial_dimension("Cash Flow to Sales")


def test_is_directly_computable():
    assert "是" in is_directly_computable("Debt ratio %")
    assert "否" in is_directly_computable("Persistent EPS in the Last Four Seasons")


def test_permutation_importance_synthetic():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(100, 4))
    # Make feature 0 strongly predictive
    y = (X[:, 0] > 0.0).astype(int)

    clf = ExtraTreesClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    mean_pr, std_pr, mean_roc, std_roc = compute_permutation_importance(
        clf, X, y, n_repeats=3, seed=42
    )

    assert len(mean_pr) == 4
    assert not np.isnan(mean_pr).any()
    # Feature 0 should have the largest drop in PR-AUC
    assert mean_pr[0] >= mean_pr[1]
    assert mean_pr[0] >= mean_pr[2]
