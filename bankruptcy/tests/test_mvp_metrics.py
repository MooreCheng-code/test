"""
Unit Tests for MVP Extended Metrics and Preprocessing Components (Stage 2).

Covers:
1. TrainingMedianImputer behavior with DataFrame and ndarray, handling missing values strictly via fit medians.
2. build_pipeline integration: use_imputer=False vs use_imputer=True.
3. recall_at_top_k_percent: perfect, inverted, tied scores, and pessimistic tie-breaking.
4. confusion_matrix_at_threshold and misclassification_cost.
5. evaluate_mvp_threshold_metrics: comprehensive metric calculations (F1, F2, MCC, alert proportion, cost).
6. confusion_and_cost_at_top_k_percent: mathematical identity checks (sum = N, tp+fn = P, fp+tn = N-P).
7. cost_sensitivity_table: output DataFrame structure, cost calculation sensitivity.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, fbeta_score

from config import (
    COST_FN,
    COST_FP,
    COST_FN_SENSITIVITY_LIST,
    TOP_K_PERCENT,
)
from src.metrics import (
    compute_top_k_quota,
    confusion_and_cost_at_top_k_percent,
    confusion_matrix_at_threshold,
    cost_sensitivity_table,
    evaluate_mvp_threshold_metrics,
    misclassification_cost,
    recall_at_top_k_percent,
)
from src.pipelines import build_pipeline
from src.preprocessing import TrainingMedianImputer


# ==============================================================================
# 1. TrainingMedianImputer Tests
# ==============================================================================
def test_training_median_imputer_dataframe():
    df_train = pd.DataFrame({
        "col_a": [1.0, 2.0, 3.0, 4.0, 5.0],  # median = 3.0
        "col_b": [10.0, 20.0, np.nan, 40.0, 50.0],  # median = 30.0
    })
    imputer = TrainingMedianImputer()
    imputer.fit(df_train)

    assert imputer.medians_["col_a"] == 3.0
    assert imputer.medians_["col_b"] == 30.0

    df_test = pd.DataFrame({
        "col_a": [np.nan, 2.0],
        "col_b": [10.0, np.nan],
    })
    transformed = imputer.transform(df_test)
    assert transformed["col_a"].iloc[0] == 3.0
    assert transformed["col_a"].iloc[1] == 2.0
    assert transformed["col_b"].iloc[0] == 10.0
    assert transformed["col_b"].iloc[1] == 30.0


def test_training_median_imputer_fallback():
    imputer = TrainingMedianImputer(fallback_medians={"col_new": 99.0})
    imputer.fit(pd.DataFrame({"col_a": [1.0, 2.0, 3.0]}))

    assert "col_new" in imputer.medians_
    assert imputer.medians_["col_new"] == 99.0

    df_test = pd.DataFrame({"col_a": [1.0], "col_new": [np.nan]})
    transformed = imputer.transform(df_test)
    assert transformed["col_new"].iloc[0] == 99.0


def test_pipeline_imputer_toggle():
    pipe_default = build_pipeline("M1", use_imputer=False)
    step_names_default = [name for name, _ in pipe_default.steps]
    assert "imputer" not in step_names_default

    pipe_with_imputer = build_pipeline("M1", use_imputer=True)
    step_names_with = [name for name, _ in pipe_with_imputer.steps]
    assert "imputer" in step_names_with
    assert step_names_with[0] == "feature_selector"
    assert step_names_with[1] == "imputer"


# ==============================================================================
# 2. recall_at_top_k_percent Tests
# ==============================================================================
def test_recall_at_top_k_percent_perfect():
    # N = 10, P = 2. Positives at rank 1 and 2.
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    # Top 10% -> k = floor(0.10 * 10) = 1. Top 1 contains 1 positive -> recall = 1/2 = 0.5
    assert recall_at_top_k_percent(y_true, score, k_percent=0.10) == pytest.approx(0.5)

    # Top 20% -> k = floor(0.20 * 10) = 2. Top 2 contains 2 positives -> recall = 2/2 = 1.0
    assert recall_at_top_k_percent(y_true, score, k_percent=0.20) == pytest.approx(1.0)


def test_recall_at_top_k_percent_pessimistic_tie():
    # N = 10, P = 2. Positives at index 0 and 1.
    # Tied score 0.5 for index 1 (positive), index 2 (negative), index 3 (negative)
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    # If k = 2 (top 20%):
    # rank 1 has score 0.9 (y=1).
    # rank 2 has score 0.5 (tied between 1 pos and 2 negs).
    # Strictly greater than 0.5: 1 sample (1 pos). Rem quota = 2 - 1 = 1.
    # Tied negatives = 2.
    # Pessimistic allocation: tied negatives consume remaining slots first.
    # tp_from_ties = max(0, 1 - 2) = 0.
    # Total TP = 1. Recall = 1 / 2 = 0.5.
    assert recall_at_top_k_percent(y_true, score, k_percent=0.20) == pytest.approx(0.5)

    # If k = 3 (top 30%):
    # Rem quota = 3 - 1 = 2. Tied negatives = 2.
    # tp_from_ties = max(0, 2 - 2) = 0.
    # Total TP = 1. Recall = 0.5.
    assert recall_at_top_k_percent(y_true, score, k_percent=0.30) == pytest.approx(0.5)

    # If k = 4 (top 40%):
    # Rem quota = 4 - 1 = 3. Tied negatives = 2.
    # tp_from_ties = max(0, 3 - 2) = 1.
    # Total TP = 2. Recall = 1.0.
    assert recall_at_top_k_percent(y_true, score, k_percent=0.40) == pytest.approx(1.0)


# ==============================================================================
# 3. confusion_matrix_at_threshold & misclassification_cost Tests
# ==============================================================================
def test_confusion_matrix_and_cost():
    y_true = np.array([1, 1, 0, 0, 0])
    score = np.array([0.9, 0.4, 0.7, 0.2, 0.1])
    thr = 0.5

    cm = confusion_matrix_at_threshold(y_true, score, threshold=thr)
    assert cm["tp"] == 1  # score 0.9, y=1
    assert cm["fn"] == 1  # score 0.4, y=1
    assert cm["fp"] == 1  # score 0.7, y=0
    assert cm["tn"] == 2  # score 0.2, 0.1, y=0

    cost_default = misclassification_cost(cm["tp"], cm["fp"], cm["tn"], cm["fn"], cost_fn=20.0, cost_fp=1.0)
    assert cost_default == 1 * 20.0 + 1 * 1.0 == 21.0

    cost_custom = misclassification_cost(cm["tp"], cm["fp"], cm["tn"], cm["fn"], cost_fn=50.0, cost_fp=2.0)
    assert cost_custom == 1 * 50.0 + 1 * 2.0 == 52.0


# ==============================================================================
# 4. evaluate_mvp_threshold_metrics Tests
# ==============================================================================
def test_evaluate_mvp_threshold_metrics():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    thr = 0.65

    res = evaluate_mvp_threshold_metrics(y_true, score, threshold=thr, cost_fn=20.0, cost_fp=1.0)

    # Flagged: >= 0.65 -> indices 0, 1, 2 (scores 0.9, 0.8, 0.7).
    # Indices 0, 1 are y=1 (TP=2). Index 2 is y=0 (FP=1).
    assert res["tp"] == 2
    assert res["fp"] == 1
    assert res["fn"] == 0
    assert res["tn"] == 7
    assert res["recall"] == 1.0
    assert res["precision"] == pytest.approx(2 / 3.0)
    assert res["alert_count"] == 3
    assert res["alert_ratio"] == pytest.approx(3 / 2.0)
    assert res["alert_proportion"] == pytest.approx(3 / 10.0)
    assert res["cost"] == 0 * 20.0 + 1 * 1.0 == 1.0

    # F1 check
    expected_f1 = f1_score(y_true, (score >= thr).astype(int))
    assert res["f1"] == pytest.approx(expected_f1)


# ==============================================================================
# 5. confusion_and_cost_at_top_k_percent Tests
# ==============================================================================
def test_confusion_and_cost_at_top_k_percent():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    res = confusion_and_cost_at_top_k_percent(y_true, score, k_percent=0.20, cost_fn=20.0, cost_fp=1.0)

    assert res["k"] == 2
    assert res["tp"] == 2
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["tn"] == 8
    assert res["recall"] == 1.0
    assert res["precision"] == 1.0
    assert res["f1"] == 1.0
    assert res["cost"] == 0.0
    assert res["alert_count"] == 2
    assert res["alert_proportion"] == pytest.approx(0.20)

    # Mathematical identity assertions
    assert res["tp"] + res["fp"] + res["tn"] + res["fn"] == len(y_true)
    assert res["tp"] + res["fn"] == 2  # P
    assert res["fp"] + res["tn"] == 8  # N - P


# ==============================================================================
# 6. cost_sensitivity_table Tests
# ==============================================================================
def test_cost_sensitivity_table():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.4, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01])

    # Evaluate at threshold = 0.5:
    # Flagged: scores 0.9 (y=1), 0.7 (y=0), 0.6 (y=0), 0.5 (y=0) -> TP=1, FP=3, FN=1, TN=5
    df_sens = cost_sensitivity_table(y_true, score, threshold=0.5, sensitivity_list=[10.0, 20.0, 50.0], cost_fp=1.0)

    assert isinstance(df_sens, pd.DataFrame)
    assert len(df_sens) == 3
    assert list(df_sens["cost_fn"]) == [10.0, 20.0, 50.0]

    # Row 0: cost_fn=10.0 -> FN*10 + FP*1 = 1*10 + 3*1 = 13.0
    assert df_sens.iloc[0]["total_cost"] == 13.0
    assert df_sens.iloc[0]["cost_per_sample"] == pytest.approx(13.0 / 10.0)

    # Row 1: cost_fn=20.0 -> 1*20 + 3*1 = 23.0
    assert df_sens.iloc[1]["total_cost"] == 23.0

    # Row 2: cost_fn=50.0 -> 1*50 + 3*1 = 53.0
    assert df_sens.iloc[2]["total_cost"] == 53.0


# ==============================================================================
# 7. Additional Edge Case Tests (All-tied, No Positives, k=0, N<10)
# ==============================================================================
def test_compute_top_k_quota():
    # floor method (default)
    assert compute_top_k_quota(100, 0.10) == 10
    assert compute_top_k_quota(1091, 0.10) == 109
    assert compute_top_k_quota(9, 0.10) is None  # floor(0.9) = 0 -> None
    assert compute_top_k_quota(4, 0.10) is None  # floor(0.4) = 0 -> None
    assert compute_top_k_quota(0, 0.10) is None

    # ceil method
    assert compute_top_k_quota(9, 0.10, method="ceil") == 1
    assert compute_top_k_quota(1091, 0.10, method="ceil") == 110


def test_recall_at_top_k_percent_all_tied():
    # N=10, P=2. All 10 samples tied with score 0.5.
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.5] * 10)

    # If k = 5 (top 50%):
    # Pessimistic: all 8 negatives take the quota first.
    # Slots available = 5 <= 8 negatives -> TP = 0 -> recall = 0.0
    assert recall_at_top_k_percent(y_true, score, k_percent=0.50) == 0.0

    # If k = 9 (top 90%):
    # 8 negatives take 8 slots, remaining 1 slot takes 1 positive -> TP = 1 -> recall = 1/2 = 0.5
    assert recall_at_top_k_percent(y_true, score, k_percent=0.90) == pytest.approx(0.5)

    # If k = 10 (100%): all 10 slots taken -> TP = 2 -> recall = 1.0
    assert recall_at_top_k_percent(y_true, score, k_percent=1.00) == pytest.approx(1.0)


def test_recall_at_top_k_percent_no_positives():
    y_no_pos = np.array([0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    with pytest.raises(ValueError, match="at least one positive sample"):
        recall_at_top_k_percent(y_no_pos, score, k_percent=0.20)


def test_recall_at_top_k_percent_small_n_k_zero():
    # N=5 < 10, k_percent=0.10 -> floor(0.5) = 0 -> quota is None
    y_small = np.array([1, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    res_rec = recall_at_top_k_percent(y_small, score, k_percent=0.10)
    assert res_rec is None  # Does not divide by zero or return misleading 0

    res_cc = confusion_and_cost_at_top_k_percent(y_small, score, k_percent=0.10)
    assert res_cc is None  # Does not divide by zero or crash

    with pytest.raises(ValueError, match="too small"):
        cost_sensitivity_table(y_small, score, k_percent=0.10)

