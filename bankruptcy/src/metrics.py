"""
Custom Evaluation Metrics for Bankruptcy Prediction.

Specialized for rare-event ranking evaluation where the business objective
is to capture nearly all bankruptcies while monitoring alert costs:

1. alert_ratio_at_recall:
   Multiplier of alert count relative to actual bankrupt count to achieve target recall.
   Conservative tie-breaking: All tied samples at the score threshold are included.

2. recall_at_alert_ratio:
   Recall achieved when fixing alert count to (ratio * P).
   Pessimistic tie-breaking: Quota at the boundary is assumed to be filled with negative
   samples first (worst-case guarantee).

3. evaluate_scores:
   Computes alert ratios, fixed-ratio recalls, PR-AUC, and ROC-AUC.

4. evaluate_at_threshold:
   Computes recall, precision, F2-score, MCC, confusion matrix, and alert ratio at a fixed threshold.

5. brier:
   Brier score loss wrapper for probability calibration quality.
"""

import sys
from pathlib import Path
from typing import Sequence, Union

# Ensure config is importable regardless of working directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
)

try:
    from config import ALERT_RATIOS as DEFAULT_ALERT_RATIOS
    from config import TARGET_RECALLS as DEFAULT_TARGET_RECALLS
except ImportError:
    DEFAULT_TARGET_RECALLS = [0.80, 0.90, 0.95, 1.00]
    DEFAULT_ALERT_RATIOS = [1.5, 3, 5, 10]


ArrayLike = Union[Sequence, np.ndarray, pd.Series]


def _validate_inputs(y_true: ArrayLike, score: ArrayLike) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Validate input arrays and return (y_arr, score_arr, P, N).

    Raises:
        ValueError: If lengths mismatch, score contains NaN, y_true is not binary 0/1,
                    or y_true contains no positive samples.
    """
    if y_true is None or score is None:
        raise ValueError("y_true and score must not be None.")

    y_arr = np.asarray(y_true).ravel()
    s_arr = np.asarray(score).ravel()

    if len(y_arr) != len(s_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_arr)} elements, but score has {len(s_arr)} elements."
        )

    if len(y_arr) == 0:
        raise ValueError("y_true and score must not be empty.")

    if np.isnan(s_arr).any():
        raise ValueError("score array contains NaN values.")

    unique_vals = np.unique(y_arr)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(
            f"y_true must contain only binary labels (0 or 1), but got unique values: {unique_vals}."
        )

    p_count = int((y_arr == 1).sum())
    if p_count == 0:
        raise ValueError("y_true must contain at least one positive sample (y=1).")

    n_total = len(y_arr)
    return y_arr.astype(int), s_arr.astype(float), p_count, n_total


def alert_ratio_at_recall(y_true: ArrayLike, score: ArrayLike, target_recall: float) -> float:
    """
    Compute the alert ratio (flagged companies / actual bankrupt companies) needed to achieve target_recall.

    Lower is better (1.0 is perfect ranking).

    Tie-breaking policy:
        Conservative. When multiple samples share the score threshold of the need-th positive sample,
        all tied samples are flagged (`score >= thr`). This prevents tree-based models or coarse scores
        from underestimating alert costs.

    Parameters:
        y_true: Ground truth binary labels (0 or 1).
        score: Continuous predicted risk scores / probabilities (higher indicates higher risk).
        target_recall: Desired recall rate, must be in (0, 1].

    Returns:
        float: Alert multiplier = (flagged count) / (actual positive count).
    """
    if not (0.0 < target_recall <= 1.0):
        raise ValueError(f"target_recall must be strictly in the interval (0, 1], got {target_recall}.")

    y_arr, s_arr, P, _ = _validate_inputs(y_true, score)

    # Number of positive samples required to achieve target recall
    need = int(np.ceil(target_recall * P))
    need = max(1, min(need, P))

    # Identify the threshold score (the need-th highest score among positive samples)
    pos_scores = np.sort(s_arr[y_arr == 1])[::-1]
    thr = pos_scores[need - 1]

    # Flag all samples with score >= thr (conservative tie-breaking)
    flagged = int((s_arr >= thr).sum())
    return float(flagged / P)


def recall_at_alert_ratio(y_true: ArrayLike, score: ArrayLike, ratio: float) -> float:
    """
    Compute the recall achieved when fixing the alert budget to k = floor(ratio * P).

    Higher is better (1.0 is perfect recall).

    Tie-breaking policy:
        Pessimistic / Worst-case guarantee.
        Samples with scores strictly greater than the k-th score are taken first.
        For the remaining quota within the tied group at the k-th score threshold,
        it is assumed that negative samples are picked first (worst-case allocation).
        Therefore:
            TP = TP(strictly > thr) + max(0, remaining_quota - tied_negatives).

    Parameters:
        y_true: Ground truth binary labels (0 or 1).
        score: Continuous predicted risk scores / probabilities (higher indicates higher risk).
        ratio: Alert budget multiplier, must be > 0.

    Returns:
        float: Recall rate achieved at the fixed alert capacity.
    """
    if ratio <= 0.0:
        raise ValueError(f"ratio must be strictly positive (> 0), got {ratio}.")

    y_arr, s_arr, P, N = _validate_inputs(y_true, score)

    # Compute alert budget k
    k = int(np.floor(ratio * P))
    if k <= 0:
        return 0.0
    if k >= N:
        return 1.0

    # Sort scores descending to find score at rank k
    sorted_scores = np.sort(s_arr)[::-1]
    thr = sorted_scores[k - 1]

    # 1. Samples strictly greater than threshold
    strictly_greater = s_arr > thr
    tp_strictly_greater = int((strictly_greater & (y_arr == 1)).sum())
    n_strictly_greater = int(strictly_greater.sum())

    # Remaining quota to fill up to k
    rem = k - n_strictly_greater

    # 2. Tied group at threshold
    tied_mask = s_arr == thr
    tied_negatives = int((tied_mask & (y_arr == 0)).sum())

    # Pessimistic allocation: tied negatives consume remaining slots first
    tp_from_ties = max(0, rem - tied_negatives)

    total_tp = tp_strictly_greater + tp_from_ties
    return float(total_tp / P)


def evaluate_scores(
    y_true: ArrayLike,
    score: ArrayLike,
    target_recalls: Sequence[float] = None,
    alert_ratios: Sequence[float] = None,
) -> dict[str, float]:
    """
    Evaluate ranking performance across target recalls, alert ratios, and standard curves.

    Parameters:
        y_true: Ground truth binary labels.
        score: Continuous predicted risk scores.
        target_recalls: List of target recall levels (defaults to config.TARGET_RECALLS).
        alert_ratios: List of alert budget multipliers (defaults to config.ALERT_RATIOS).

    Returns:
        dict: Standardized metrics dictionary:
            - alert_ratio@recall_{r:.2f}
            - recall@alert_ratio_{ratio}
            - pr_auc (Average Precision Score)
            - roc_auc (ROC Area Under Curve)
    """
    if target_recalls is None:
        target_recalls = DEFAULT_TARGET_RECALLS
    if alert_ratios is None:
        alert_ratios = DEFAULT_ALERT_RATIOS

    y_arr, s_arr, P, _ = _validate_inputs(y_true, score)
    results: dict[str, float] = {}

    # Alert ratios at given recalls
    for r in target_recalls:
        key = f"alert_ratio@recall_{r:.2f}"
        results[key] = float(alert_ratio_at_recall(y_arr, s_arr, r))

    # Recalls at given alert ratios
    for ratio in alert_ratios:
        ratio_str = f"{ratio:g}"
        key = f"recall@alert_ratio_{ratio_str}"
        results[key] = float(recall_at_alert_ratio(y_arr, s_arr, ratio))

    # Continuous ranking metrics
    results["pr_auc"] = float(average_precision_score(y_arr, s_arr))

    # ROC-AUC (handle constant score case gracefully)
    if np.all(s_arr == s_arr[0]):
        results["roc_auc"] = 0.5
    else:
        results["roc_auc"] = float(roc_auc_score(y_arr, s_arr))

    return results


def evaluate_at_threshold(y_true: ArrayLike, score: ArrayLike, threshold: float) -> dict[str, Union[float, int]]:
    """
    Evaluate classification metrics at a given decision threshold (`score >= threshold`).

    Parameters:
        y_true: Ground truth binary labels.
        score: Continuous predicted risk scores.
        threshold: Decision threshold.

    Returns:
        dict: Confusion matrix counts and summary metrics:
            - recall, precision, f2, mcc, alert_ratio (floats)
            - tn, fp, fn, tp (integers)
    """
    y_arr, s_arr, P, _ = _validate_inputs(y_true, score)

    pred = (s_arr >= threshold).astype(int)

    tp = int(((pred == 1) & (y_arr == 1)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())
    tn = int(((pred == 0) & (y_arr == 0)).sum())

    recall = float(tp / P) if P > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    alert_ratio = float((tp + fp) / P) if P > 0 else 0.0

    f2 = float(fbeta_score(y_arr, pred, beta=2, zero_division=0.0))
    mcc = float(matthews_corrcoef(y_arr, pred))

    return {
        "recall": recall,
        "precision": precision,
        "f2": f2,
        "mcc": mcc,
        "alert_ratio": alert_ratio,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def brier(y_true: ArrayLike, proba: ArrayLike) -> float:
    """
    Compute Brier score loss for calibrated probability predictions.

    Parameters:
        y_true: Ground truth binary labels.
        proba: Predicted probabilities of the positive class.

    Returns:
        float: Brier score loss.
    """
    y_arr, p_arr, _, _ = _validate_inputs(y_true, proba)
    return float(brier_score_loss(y_arr, p_arr))
