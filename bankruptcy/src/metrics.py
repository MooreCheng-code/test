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
from typing import Any, Optional, Sequence, Union

# Ensure config is importable regardless of working directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
)

try:
    from config import ALERT_RATIOS as DEFAULT_ALERT_RATIOS
    from config import TARGET_RECALLS as DEFAULT_TARGET_RECALLS
    from config import (
        COST_FN as DEFAULT_COST_FN,
        COST_FP as DEFAULT_COST_FP,
        COST_FN_SENSITIVITY_LIST as DEFAULT_COST_FN_SENSITIVITY_LIST,
        TOP_K_PERCENT as DEFAULT_TOP_K_PERCENT,
        MIN_BATCH_SIZE_FOR_TOP_K as DEFAULT_MIN_BATCH_SIZE_FOR_TOP_K,
    )
except ImportError:
    DEFAULT_TARGET_RECALLS = [0.80, 0.90, 0.95, 1.00]
    DEFAULT_ALERT_RATIOS = [1.5, 3, 5, 10]
    DEFAULT_TOP_K_PERCENT = 0.10
    DEFAULT_MIN_BATCH_SIZE_FOR_TOP_K = 25
    DEFAULT_COST_FN = 20.0
    DEFAULT_COST_FP = 1.0
    DEFAULT_COST_FN_SENSITIVITY_LIST = [10.0, 20.0, 50.0]


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


def compute_top_k_quota(
    n_samples: int,
    k_percent: float = DEFAULT_TOP_K_PERCENT,
    method: str = "floor",
) -> Optional[int]:
    """
    Compute the top-k sample quota based on total population size and percentage budget.

    Shared helper across evaluation metrics and batch scoring (in_batch_top_10_pct).

    Standard policy:
        method="floor": Guarantees flagged budget does not exceed k_percent (Budget Cap constraint).
        k = floor(k_percent * n_samples).
        If k <= 0 (e.g. n_samples < 10 for k_percent=0.10), returns None to indicate
        insufficient sample size. Prevents division-by-zero or misleading 0 returns.
    """
    if n_samples <= 0 or not (0.0 < k_percent <= 1.0):
        return None

    if method == "floor":
        k = int(np.floor(k_percent * n_samples))
    elif method == "ceil":
        k = int(np.ceil(k_percent * n_samples))
    else:
        raise ValueError(f"Unknown quota method: {method}. Supported: 'floor', 'ceil'.")

    return k if k > 0 else None


def recall_at_top_k_percent(
    y_true: ArrayLike,
    score: ArrayLike,
    k_percent: float = DEFAULT_TOP_K_PERCENT,
    method: str = "floor",
) -> Optional[float]:
    """
    Compute the recall achieved when flagging the top k% highest risk samples.
    k is computed via compute_top_k_quota(N, k_percent, method=method).

    Higher is better (1.0 is perfect recall).

    Tie-breaking policy:
        Pessimistic / Worst-case guarantee (identical to recall_at_alert_ratio).
        Samples with scores strictly greater than the k-th score are taken first.
        For the remaining quota within the tied group at the k-th score threshold,
        negative samples are assumed to be picked first.

    Edge cases:
        If k is None or k <= 0 (insufficient N, e.g. N < 10 for 10% budget), returns None.
        If y_true contains no positive samples, raises ValueError.

    Parameters:
        y_true: Ground truth binary labels (0 or 1).
        score: Continuous predicted risk scores (higher indicates higher risk).
        k_percent: Top fraction of total population to flag (0, 1]. Defaults to config.TOP_K_PERCENT (0.10).
        method: Quota computation method ('floor' or 'ceil'). Defaults to 'floor'.

    Returns:
        Optional[float]: Recall rate achieved within the top k% budget, or None if quota is 0.
    """
    if not (0.0 < k_percent <= 1.0):
        raise ValueError(f"k_percent must be strictly in the interval (0, 1], got {k_percent}.")

    y_arr, s_arr, P, N = _validate_inputs(y_true, score)

    k = compute_top_k_quota(N, k_percent=k_percent, method=method)
    if k is None or k <= 0:
        return None

    if k >= N:
        return 1.0

    sorted_scores = np.sort(s_arr)[::-1]
    thr = sorted_scores[k - 1]

    strictly_greater = s_arr > thr
    tp_strictly_greater = int((strictly_greater & (y_arr == 1)).sum())
    n_strictly_greater = int(strictly_greater.sum())

    rem = k - n_strictly_greater

    tied_mask = s_arr == thr
    tied_negatives = int((tied_mask & (y_arr == 0)).sum())

    tp_from_ties = max(0, rem - tied_negatives)
    total_tp = tp_strictly_greater + tp_from_ties

    return float(total_tp / P)


def confusion_matrix_at_threshold(
    y_true: ArrayLike,
    score: ArrayLike,
    threshold: float,
) -> dict[str, int]:
    """
    Compute confusion matrix counts (tp, fp, tn, fn) at a given decision threshold (`score >= threshold`).
    """
    y_arr, s_arr, _, _ = _validate_inputs(y_true, score)
    pred = (s_arr >= threshold).astype(int)

    tp = int(((pred == 1) & (y_arr == 1)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())
    tn = int(((pred == 0) & (y_arr == 0)).sum())

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def misclassification_cost(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> float:
    """
    Compute total business misclassification cost:
        Total Cost = fn * cost_fn + fp * cost_fp

    By default, cost_fn = 20.0 (missed bankruptcy cost) and cost_fp = 1.0 (investigation cost).
    """
    return float(fn * cost_fn + fp * cost_fp)


def evaluate_mvp_threshold_metrics(
    y_true: ArrayLike,
    score: ArrayLike,
    threshold: float,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> dict[str, Any]:
    """
    Comprehensive MVP evaluation at a fixed decision threshold (`score >= threshold`).

    Includes standard metrics, confusion matrix, F1, F2, MCC, alert volume, and misclassification cost.
    """
    y_arr, s_arr, P, N = _validate_inputs(y_true, score)
    pred = (s_arr >= threshold).astype(int)

    tp = int(((pred == 1) & (y_arr == 1)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())
    tn = int(((pred == 0) & (y_arr == 0)).sum())

    recall = float(tp / P) if P > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(f1_score(y_arr, pred, zero_division=0.0))
    f2 = float(fbeta_score(y_arr, pred, beta=2, zero_division=0.0))
    mcc = float(matthews_corrcoef(y_arr, pred))

    cost = misclassification_cost(tp, fp, tn, fn, cost_fn=cost_fn, cost_fp=cost_fp)
    alert_count = int(tp + fp)
    alert_ratio = float((tp + fp) / P) if P > 0 else 0.0
    alert_proportion = float((tp + fp) / N)

    return {
        "threshold": float(threshold),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "f2": f2,
        "mcc": mcc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "cost": cost,
        "alert_count": alert_count,
        "alert_ratio": alert_ratio,
        "alert_proportion": alert_proportion,
    }


def confusion_and_cost_at_top_k_percent(
    y_true: ArrayLike,
    score: ArrayLike,
    k_percent: float = DEFAULT_TOP_K_PERCENT,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    method: str = "floor",
) -> Optional[dict[str, Any]]:
    """
    Compute confusion matrix, recall, precision, F1, and cost under fixed top k% capacity budget.
    Adheres to pessimistic tie-breaking if scores are tied at the rank-k boundary.

    If k is None or k <= 0 (insufficient N, e.g. N < 10 for 10% budget), returns None.
    """
    if not (0.0 < k_percent <= 1.0):
        raise ValueError(f"k_percent must be strictly in the interval (0, 1], got {k_percent}.")

    y_arr, s_arr, P, N = _validate_inputs(y_true, score)

    k = compute_top_k_quota(N, k_percent=k_percent, method=method)
    if k is None or k <= 0:
        return None

    if k >= N:
        tp, fp = P, N - P
        fn, tn = 0, 0
        thr = float(np.min(s_arr))
    else:
        sorted_scores = np.sort(s_arr)[::-1]
        thr = float(sorted_scores[k - 1])

        strictly_greater = s_arr > thr
        tp_strictly = int((strictly_greater & (y_arr == 1)).sum())
        n_strictly = int(strictly_greater.sum())

        rem = k - n_strictly
        tied_mask = s_arr == thr
        tied_negatives = int((tied_mask & (y_arr == 0)).sum())

        tp = int(tp_strictly + max(0, rem - tied_negatives))
        fp = int(k - tp)
        fn = int(P - tp)
        tn = int((N - P) - fp)

    recall = float(tp / P) if P > 0 else 0.0
    precision = float(tp / k) if k > 0 else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    cost = misclassification_cost(tp, fp, tn, fn, cost_fn=cost_fn, cost_fp=cost_fp)

    return {
        "k": k,
        "k_percent": float(k_percent),
        "threshold": thr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "cost": cost,
        "alert_count": k,
        "alert_ratio": float(k / P) if P > 0 else 0.0,
        "alert_proportion": float(k / N),
    }


def cost_sensitivity_table(
    y_true: ArrayLike,
    score: ArrayLike,
    threshold: Optional[float] = None,
    k_percent: Optional[float] = None,
    sensitivity_list: Sequence[float] = None,
    cost_fp: float = DEFAULT_COST_FP,
    method: str = "floor",
) -> pd.DataFrame:
    """
    Generate business misclassification cost sensitivity table across various false negative cost weights.

    Can evaluate at either a fixed score threshold (threshold != None) or a capacity percentage (k_percent != None).
    """
    if threshold is None and k_percent is None:
        k_percent = DEFAULT_TOP_K_PERCENT

    if sensitivity_list is None:
        sensitivity_list = DEFAULT_COST_FN_SENSITIVITY_LIST

    y_arr, s_arr, P, N = _validate_inputs(y_true, score)

    if threshold is not None:
        cm = confusion_matrix_at_threshold(y_arr, s_arr, threshold)
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        eval_point = f"Threshold >= {threshold:.4f}"
    else:
        k_res = confusion_and_cost_at_top_k_percent(y_arr, s_arr, k_percent=k_percent, method=method)
        if k_res is None:
            raise ValueError(
                f"Cannot compute cost sensitivity table at Top {k_percent*100:.1f}%: "
                f"sample size N={N} is too small to yield quota k >= 1."
            )
        tp, fp, tn, fn = k_res["tp"], k_res["fp"], k_res["tn"], k_res["fn"]
        eval_point = f"Top {k_percent * 100:.1f}%"

    rows = []
    for c_fn in sensitivity_list:
        cost = misclassification_cost(tp, fp, tn, fn, cost_fn=c_fn, cost_fp=cost_fp)
        rows.append({
            "evaluation_point": eval_point,
            "cost_fn": float(c_fn),
            "cost_fp": float(cost_fp),
            "cost_ratio": f"{c_fn:g}:{cost_fp:g}",
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "total_cost": float(cost),
            "cost_per_sample": float(cost / N),
        })

    return pd.DataFrame(rows)
