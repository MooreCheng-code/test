"""
Unit Tests for Custom Evaluation Metrics (src/metrics.py).

Covers:
1. Perfect ranking behavior.
2. Completely inverted ranking behavior.
3. Completely tied score behavior.
4. Small hand-calculated example (N=10, P=2).
5. Boundary-crossing tie-breaking case with pessimistic guarantee.
6. Mathematical consistency (inverse function relationship).
7. Equivalence comparison with scikit-learn precision_recall_curve.
8. Comprehensive input validation and boundary checks.
9. evaluate_scores, evaluate_at_threshold, and brier score verification.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve

from src.metrics import (
    alert_ratio_at_recall,
    brier,
    evaluate_at_threshold,
    evaluate_scores,
    recall_at_alert_ratio,
)


# ==============================================================================
# Test 1: Perfect Ranking
# ==============================================================================
def test_perfect_ranking():
    """
    Test 1: All positive samples receive the highest scores.
    Hand calculation:
        N = 10, P = 3.
        Positive scores: [0.9, 0.8, 0.7].
        Negative scores: [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0].
        At any target recall t in (0, 1]:
            need = ceil(t * 3) in {1, 2, 3}.
            Threshold is pos_scores[need - 1] in {0.9, 0.8, 0.7}.
            All samples >= thr are strictly positive samples.
            Therefore flagged = need <= 3.
            For t = 1.0: need = 3, flagged = 3, alert_ratio = 3 / 3 = 1.0.
            For t = 0.8: need = 3, alert_ratio = 1.0.
        At ratio = 1.0:
            k = floor(1.0 * 3) = 3.
            Top 3 samples are all 3 positives, so recall = 3 / 3 = 1.0.
    """
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    for target in [0.1, 0.5, 0.80, 0.90, 0.95, 1.00]:
        ratio = alert_ratio_at_recall(y_true, score, target)
        expected_flagged = int(np.ceil(target * 3))
        assert ratio == pytest.approx(expected_flagged / 3.0)

    # For full recall, alert ratio must be exactly 1.0
    assert alert_ratio_at_recall(y_true, score, 1.0) == pytest.approx(1.0)

    # At ratio 1.0, recall must be 1.0
    assert recall_at_alert_ratio(y_true, score, 1.0) == pytest.approx(1.0)


# ==============================================================================
# Test 2: Inverted Ranking
# ==============================================================================
def test_inverted_ranking():
    """
    Test 2: All positive samples receive the lowest scores.
    Hand calculation:
        N = 10, P = 2.
        Negatives: scores 0.9 down to 0.2 (8 samples).
        Positives: scores 0.1, 0.0 (2 samples).
        For target_recall = 1.0:
            need = 2. The 2nd positive score is 0.0.
            Threshold is 0.0. All 10 samples have score >= 0.0.
            Flagged count = 10.
            alert_ratio = 10 / 2 = 5.0 (= N / P).
        For large N = 1000, P = 10:
            k = floor(1.5 * 10) = 15.
            Top 15 scores belong to negative samples (negatives have 990 samples).
            Therefore TP = 0, recall = 0.0.
    """
    y_small = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    score_small = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    assert alert_ratio_at_recall(y_small, score_small, 1.0) == pytest.approx(10.0 / 2.0)

    # Large dataset inverted
    N_large = 1000
    P_large = 10
    y_large = np.zeros(N_large, dtype=int)
    y_large[-P_large:] = 1  # Last 10 are positives
    score_large = np.linspace(1.0, 0.0, N_large)  # Strictly descending

    # Ratio 1.5 -> k = 15. Top 15 are all negatives.
    assert recall_at_alert_ratio(y_large, score_large, 1.5) == pytest.approx(0.0)


# ==============================================================================
# Test 3: Completely Tied Scores
# ==============================================================================
def test_completely_tied_scores():
    """
    Test 3: Every sample has the exact same score.
    Hand calculation:
        N = 10, P = 2 (Negatives = 8).
        All scores = 0.5.
        alert_ratio_at_recall:
            Threshold = 0.5. All 10 samples satisfy score >= 0.5.
            Flagged = 10.
            alert_ratio = 10 / 2 = 5.0 (= N / P) for any target_recall > 0.
        recall_at_alert_ratio (pessimistic tie-breaking):
            All 10 samples are tied at 0.5. Strictly greater = 0.
            rem = k. Tied negatives = 8.
            TP = max(0, k - 8).
            If k <= 8 (e.g. ratio = 1.5 -> k = 3, ratio = 3.0 -> k = 6, ratio = 4.0 -> k = 8):
                TP = 0, recall = 0.0.
            If k = 9 (ratio = 4.5):
                TP = 9 - 8 = 1, recall = 1 / 2 = 0.5.
            If k = 10 (ratio = 5.0):
                TP = 10 - 8 = 2, recall = 2 / 2 = 1.0.
    """
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.full(10, 0.5)

    for target in [0.1, 0.5, 0.95, 1.0]:
        assert alert_ratio_at_recall(y_true, score, target) == pytest.approx(10.0 / 2.0)

    # k <= 8 negatives -> recall = 0.0
    assert recall_at_alert_ratio(y_true, score, 1.5) == pytest.approx(0.0)  # k = 3
    assert recall_at_alert_ratio(y_true, score, 3.0) == pytest.approx(0.0)  # k = 6
    assert recall_at_alert_ratio(y_true, score, 4.0) == pytest.approx(0.0)  # k = 8

    # k = 9 -> TP = 1 -> recall = 0.5
    assert recall_at_alert_ratio(y_true, score, 4.5) == pytest.approx(0.5)

    # k = 10 -> TP = 2 -> recall = 1.0
    assert recall_at_alert_ratio(y_true, score, 5.0) == pytest.approx(1.0)


# ==============================================================================
# Test 4: Small Hand-Calculated Case (N=10, P=2)
# ==============================================================================
def test_small_hand_calculated_example():
    """
    Test 4:
        N = 10, P = 2.
        y_true = [1,   0,   1,   0,   0,   0,   0,   0,   0,   0]
        score  = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

        Rank 1: score=0.9, y=1
        Rank 2: score=0.8, y=0
        Rank 3: score=0.7, y=1
        Rank 4..10: score <= 0.6, y=0

        Hand calculation for alert_ratio_at_recall:
            - target_recall = 0.5:
                need = ceil(0.5 * 2) = 1.
                Positives sorted: [0.9, 0.7]. 1st is 0.9.
                thr = 0.9.
                score >= 0.9 has 1 sample (Rank 1).
                alert_ratio = 1 / 2 = 0.5.
            - target_recall = 1.0:
                need = ceil(1.0 * 2) = 2.
                2nd positive is 0.7.
                thr = 0.7.
                score >= 0.7 has 3 samples (Ranks 1, 2, 3: scores 0.9, 0.8, 0.7).
                alert_ratio = 3 / 2 = 1.5.

        Hand calculation for recall_at_alert_ratio:
            - ratio = 1.0:
                k = floor(1.0 * 2) = 2.
                Top 2 samples: Rank 1 (y=1), Rank 2 (y=0).
                TP = 1.
                recall = 1 / 2 = 0.5.
            - ratio = 2.0:
                k = floor(2.0 * 2) = 4.
                Top 4 samples: Ranks 1..4 contains both positives (Ranks 1 and 3).
                TP = 2.
                recall = 2 / 2 = 1.0.
            - ratio = 3.0:
                k = floor(3.0 * 2) = 6.
                Top 6 samples contains both positives.
                TP = 2.
                recall = 2 / 2 = 1.0.
    """
    y_true = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    assert alert_ratio_at_recall(y_true, score, 0.5) == pytest.approx(0.5)
    assert alert_ratio_at_recall(y_true, score, 1.0) == pytest.approx(1.5)

    assert recall_at_alert_ratio(y_true, score, 1.0) == pytest.approx(0.5)
    assert recall_at_alert_ratio(y_true, score, 2.0) == pytest.approx(1.0)
    assert recall_at_alert_ratio(y_true, score, 3.0) == pytest.approx(1.0)


# ==============================================================================
# Test 5: Boundary-Crossing Tie Case (Pessimistic Allocation)
# ==============================================================================
def test_boundary_crossing_ties():
    """
    Test 5: Ties crossing the budget boundary.
    Hand calculation:
        N = 6, P = 3.
        y_true = [1,   0,   0,   1,   0,   1]
        score  = [0.9, 0.8, 0.5, 0.5, 0.5, 0.1]

        Group score > 0.5:
            Index 0: score 0.9, y=1 (TP > thr = 1)
            Index 1: score 0.8, y=0
            Total strictly > 0.5 count = 2.
        Tied group score == 0.5:
            Index 2: y=0
            Index 3: y=1
            Index 4: y=0
            Tied count = 3 (Tied positives = 1, Tied negatives = 2).
        Score < 0.5:
            Index 5: score 0.1, y=1.

        Pessimistic recall_at_alert_ratio:
            - ratio = 1.0 -> k = floor(1.0 * 3) = 3:
                rem = 3 - 2 = 1 slot.
                Tied negatives = 2.
                Pessimistically filled by negative: max(0, 1 - 2) = 0.
                Total TP = 1 + 0 = 1.
                recall = 1 / 3.
            - ratio = 4/3 -> k = 4:
                rem = 4 - 2 = 2 slots.
                Tied negatives = 2.
                Pessimistically filled by negatives: max(0, 2 - 2) = 0.
                Total TP = 1 + 0 = 1.
                recall = 1 / 3.
            - ratio = 5/3 -> k = 5:
                rem = 5 - 2 = 3 slots.
                Tied negatives = 2.
                Negatives exhausted, positive forced: max(0, 3 - 2) = 1.
                Total TP = 1 + 1 = 2.
                recall = 2 / 3.

        Conservative alert_ratio_at_recall:
            - target_recall = 0.5 -> need = ceil(0.5 * 3) = 2.
                Positive scores sorted: [0.9, 0.5, 0.1]. 2nd positive score is 0.5.
                thr = 0.5.
                Samples with score >= 0.5: indices 0, 1, 2, 3, 4 (5 samples).
                alert_ratio = 5 / 3.
            - target_recall = 1.0 -> need = 3. 3rd positive score is 0.1.
                thr = 0.1.
                All 6 samples >= 0.1.
                alert_ratio = 6 / 3 = 2.0.
    """
    y_true = [1, 0, 0, 1, 0, 1]
    score = [0.9, 0.8, 0.5, 0.5, 0.5, 0.1]

    assert recall_at_alert_ratio(y_true, score, 1.0) == pytest.approx(1.0 / 3.0)
    assert recall_at_alert_ratio(y_true, score, 4.0 / 3.0) == pytest.approx(1.0 / 3.0)
    assert recall_at_alert_ratio(y_true, score, 5.0 / 3.0) == pytest.approx(2.0 / 3.0)

    assert alert_ratio_at_recall(y_true, score, 0.5) == pytest.approx(5.0 / 3.0)
    assert alert_ratio_at_recall(y_true, score, 1.0) == pytest.approx(2.0)


# ==============================================================================
# Test 6: Consistency Between Alert Ratio and Recall (Inverse Relation)
# ==============================================================================
def test_inverse_relationship_consistency():
    """
    Test 6: On tie-free continuous random data,
    if r = alert_ratio_at_recall(y, s, t), then recall_at_alert_ratio(y, s, r) >= t.
    """
    rng = np.random.RandomState(42)
    N = 300
    P = 20
    y_true = np.zeros(N, dtype=int)
    y_true[rng.choice(N, size=P, replace=False)] = 1
    # Distinct continuous scores
    score = rng.normal(loc=0.0, scale=1.0, size=N) + y_true * 0.5

    for target in [0.20, 0.50, 0.80, 0.90, 0.95, 1.00]:
        r = alert_ratio_at_recall(y_true, score, target)
        achieved_recall = recall_at_alert_ratio(y_true, score, r)
        # Achieved recall must meet or exceed target recall
        assert achieved_recall >= target - 1e-9


# ==============================================================================
# Test 7: Equivalence with scikit-learn precision_recall_curve
# ==============================================================================
def test_sklearn_precision_recall_curve_consistency():
    """
    Test 7: Compare recall_at_alert_ratio against sklearn precision_recall_curve
    on tie-free random data across various sample sizes.
    """
    rng = np.random.RandomState(123)
    N = 100
    P = 15
    y_true = np.zeros(N, dtype=int)
    y_true[rng.choice(N, size=P, replace=False)] = 1
    score = rng.uniform(0.0, 1.0, size=N)

    # Calculate recall using precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(y_true, score)

    # For each rank k from 1 to N, check that recall matches
    sorted_scores = np.sort(score)[::-1]
    for k in [1, 5, 10, 15, 20, 50, 80, 100]:
        ratio = k / P
        computed_rec = recall_at_alert_ratio(y_true, score, ratio)
        thr_k = sorted_scores[k - 1]
        expected_rec = (y_true[score >= thr_k]).sum() / P
        assert computed_rec == pytest.approx(expected_rec)


# ==============================================================================
# Test 8: Input Validation and Boundary Exceptions
# ==============================================================================
def test_input_validation_and_errors():
    """
    Test 8: Verify ValueErrors on invalid inputs.
    """
    valid_y = [0, 1, 0, 1]
    valid_s = [0.1, 0.4, 0.2, 0.8]

    # 1. No positive samples (P = 0)
    with pytest.raises(ValueError, match="at least one positive sample"):
        alert_ratio_at_recall([0, 0, 0], [0.1, 0.2, 0.3], 0.95)
    with pytest.raises(ValueError, match="at least one positive sample"):
        recall_at_alert_ratio([0, 0, 0], [0.1, 0.2, 0.3], 1.5)

    # 2. target_recall out of bounds: target_recall <= 0 or > 1.0
    with pytest.raises(ValueError, match="target_recall must be strictly in"):
        alert_ratio_at_recall(valid_y, valid_s, 0.0)
    with pytest.raises(ValueError, match="target_recall must be strictly in"):
        alert_ratio_at_recall(valid_y, valid_s, -0.5)
    with pytest.raises(ValueError, match="target_recall must be strictly in"):
        alert_ratio_at_recall(valid_y, valid_s, 1.2)

    # 3. ratio <= 0
    with pytest.raises(ValueError, match="ratio must be strictly positive"):
        recall_at_alert_ratio(valid_y, valid_s, 0.0)
    with pytest.raises(ValueError, match="ratio must be strictly positive"):
        recall_at_alert_ratio(valid_y, valid_s, -1.0)

    # 4. Score contains NaN
    with pytest.raises(ValueError, match="NaN"):
        alert_ratio_at_recall(valid_y, [0.1, np.nan, 0.2, 0.8], 0.95)
    with pytest.raises(ValueError, match="NaN"):
        recall_at_alert_ratio(valid_y, [0.1, np.nan, 0.2, 0.8], 1.5)

    # 5. Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        alert_ratio_at_recall([0, 1], [0.1, 0.2, 0.3], 0.95)
    with pytest.raises(ValueError, match="Length mismatch"):
        recall_at_alert_ratio([0, 1], [0.1, 0.2, 0.3], 1.5)

    # 6. Non-binary labels
    with pytest.raises(ValueError, match="binary labels"):
        alert_ratio_at_recall([0, 1, 2], [0.1, 0.2, 0.3], 0.95)
    with pytest.raises(ValueError, match="binary labels"):
        recall_at_alert_ratio([0, 1, -1], [0.1, 0.2, 0.3], 1.5)

    # 7. Empty input
    with pytest.raises(ValueError, match="empty"):
        alert_ratio_at_recall([], [], 0.95)


# ==============================================================================
# Test 9: Integration Functions (evaluate_scores, evaluate_at_threshold, brier)
# ==============================================================================
def test_evaluate_scores_and_threshold():
    """
    Test 9: Verify evaluate_scores output keys, types, and evaluate_at_threshold edge cases.
    """
    y_true = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    # 1. evaluate_scores
    res = evaluate_scores(y_true, score)
    expected_keys = [
        "alert_ratio@recall_0.80",
        "alert_ratio@recall_0.90",
        "alert_ratio@recall_0.95",
        "alert_ratio@recall_1.00",
        "recall@alert_ratio_1.5",
        "recall@alert_ratio_3",
        "recall@alert_ratio_5",
        "recall@alert_ratio_10",
        "pr_auc",
        "roc_auc",
    ]
    for key in expected_keys:
        assert key in res, f"Missing key: {key}"
        assert isinstance(res[key], float)

    # 2. evaluate_at_threshold: Normal threshold
    res_thr = evaluate_at_threshold(y_true, score, threshold=0.65)
    assert res_thr["tp"] == 2  # scores 0.9 and 0.7
    assert res_thr["fp"] == 1  # score 0.8
    assert res_thr["fn"] == 1  # score 0.3
    assert res_thr["tn"] == 6
    assert res_thr["recall"] == pytest.approx(2 / 3.0)
    assert res_thr["precision"] == pytest.approx(2 / 3.0)
    assert res_thr["alert_ratio"] == pytest.approx(3 / 3.0)
    assert isinstance(res_thr["f2"], float)
    assert isinstance(res_thr["mcc"], float)

    # 3. evaluate_at_threshold: Extremely high threshold (no warnings triggered)
    res_high = evaluate_at_threshold(y_true, score, threshold=999.0)
    assert res_high["recall"] == 0.0
    assert res_high["precision"] == 0.0  # Does not crash or divide by zero
    assert res_high["alert_ratio"] == 0.0
    assert res_high["tp"] == 0
    assert res_high["fp"] == 0
    assert res_high["fn"] == 3
    assert res_high["tn"] == 7

    # 4. evaluate_at_threshold: Extremely low threshold (all samples flagged)
    res_low = evaluate_at_threshold(y_true, score, threshold=-1.0)
    assert res_low["recall"] == 1.0
    assert res_low["precision"] == pytest.approx(3 / 10.0)
    assert res_low["alert_ratio"] == pytest.approx(10 / 3.0)
    assert res_low["tp"] == 3
    assert res_low["fp"] == 7
    assert res_low["fn"] == 0
    assert res_low["tn"] == 0

    # 5. brier score
    prob = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.1, 0.7, 0.0, 0.1, 0.2])
    brier_val = brier(y_true, prob)
    assert isinstance(brier_val, float)
    assert 0.0 <= brier_val <= 1.0
