"""
Scoring module for frozen bankruptcy prediction models.

Provides reusable inference functions for Phase E6 and live deployment:
- to_percentile: Maps raw model probabilities to empirical percentiles against a reference distribution.
- ensemble_score: Computes rank-averaged ensemble scores across specified models.
- flag: Applies threshold to flag high-risk bankrupt companies.
- load_final_components: Loads frozen models, reference distributions, thresholds, and calibrators.
- score_dataset: End-to-end inference pipeline on unseen feature data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def to_percentile(scores: np.ndarray | list[float], reference_scores: np.ndarray) -> np.ndarray:
    """
    Maps query scores to empirical cumulative distribution percentiles in reference_scores
    using symmetric midpoint tie handling:
        P(q) = (count(ref < q) + count(ref <= q)) / (2 * len(ref))
    Guarantees monotonicity and values strictly in [0, 1].
    """
    ref_sorted = np.sort(np.asarray(reference_scores, dtype=float))
    n_ref = len(ref_sorted)
    if n_ref == 0:
        raise ValueError("reference_scores cannot be empty.")
    query = np.asarray(scores, dtype=float)
    left = np.searchsorted(ref_sorted, query, side="left")
    right = np.searchsorted(ref_sorted, query, side="right")
    return (left + right) / (2.0 * n_ref)


def ensemble_score(
    model_scores_dict: dict[str, np.ndarray],
    reference_dict: dict[str, np.ndarray],
    model_names: list[str] = ["M3", "M4", "M5"],
) -> np.ndarray:
    """
    Computes equal-weight average of ECDF percentiles across specified models.
    """
    for m in model_names:
        if m not in model_scores_dict:
            raise KeyError(f"Model '{m}' missing from model_scores_dict.")
        if m not in reference_dict:
            raise KeyError(f"Reference distribution for '{m}' missing from reference_dict.")

    percentile_arrays = []
    for m in model_names:
        pct = to_percentile(model_scores_dict[m], reference_dict[m])
        percentile_arrays.append(pct)

    return np.mean(percentile_arrays, axis=0)


def flag(score: np.ndarray | float, threshold: float) -> np.ndarray | bool:
    """
    Flags companies whose score meets or exceeds the threshold.
    """
    s = np.asarray(score, dtype=float)
    return s >= float(threshold)


def load_final_components(final_dir: Path | str) -> dict[str, Any]:
    """
    Loads all frozen components from outputs/models/final/.
    """
    f_dir = Path(final_dir)
    if not f_dir.exists():
        raise FileNotFoundError(f"Final model directory {f_dir} does not exist.")

    components: dict[str, Any] = {
        "models": {},
        "references": {},
        "calibrators": {},
    }

    # Load models
    for m in ["M3", "M4", "M5"]:
        model_path = f_dir / f"model_{m}.joblib"
        if model_path.exists():
            pipe = joblib.load(model_path)
            # Enforce single-threaded deterministic inference
            clf = pipe.named_steps.get("classifier")
            if clf is not None and hasattr(clf, "n_jobs"):
                clf.set_params(n_jobs=1)
            components["models"][m] = pipe
        ref_path = f_dir / f"reference_{m}.npy"
        if ref_path.exists():
            components["references"][m] = np.load(ref_path)

    # Load thresholds
    thresh_json = f_dir / "thresholds.json"
    if thresh_json.exists():
        with open(thresh_json, "r", encoding="utf-8") as f:
            components["thresholds"] = json.load(f)

    # Load calibrators
    cal_platt = f_dir / "calibrator_platt.joblib"
    if cal_platt.exists():
        components["calibrators"]["platt"] = joblib.load(cal_platt)

    cal_iso = f_dir / "calibrator_isotonic.joblib"
    if cal_iso.exists():
        components["calibrators"]["isotonic"] = joblib.load(cal_iso)

    metadata_path = f_dir / "version_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            components["metadata"] = json.load(f)

    return components


def score_dataset(
    X: pd.DataFrame,
    components: dict[str, Any],
    model_names: list[str] = ["M3", "M4", "M5"],
) -> pd.DataFrame:
    """
    End-to-end inference pipeline:
    1. Obtains raw predicted probabilities from frozen pipelines (enforcing deterministic n_jobs=1).
    2. Maps to ECDF percentiles and averages to produce E_A ensemble score.
    3. Produces calibrated posterior probabilities (Platt and Isotonic).
    4. Evaluates risk flags at each operating recall threshold.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input X must be a pandas DataFrame.")

    raw_scores = {}
    pct_scores = {}
    for m in model_names:
        pipe = components["models"][m]
        clf = pipe.named_steps.get("classifier")
        if clf is not None and hasattr(clf, "n_jobs"):
            clf.set_params(n_jobs=1)
        raw = pipe.predict_proba(X)[:, 1]
        raw_scores[m] = raw
        pct_scores[m] = to_percentile(raw, components["references"][m])

    # Calculate uncalibrated ensemble score
    ea_scores = np.mean([pct_scores[m] for m in model_names], axis=0)

    # Calibrate probability
    cal_platt = components.get("calibrators", {}).get("platt")
    calibrated_platt = cal_platt.predict_proba(ea_scores.reshape(-1, 1))[:, 1] if cal_platt is not None else None

    cal_iso = components.get("calibrators", {}).get("isotonic")
    calibrated_iso = cal_iso.predict(ea_scores) if cal_iso is not None else None

    data_dict = {
        "score_E_A": ea_scores,
        "score_M3": pct_scores["M3"],
    }
    if calibrated_platt is not None:
        data_dict["calibrated_prob_platt"] = calibrated_platt
    if calibrated_iso is not None:
        data_dict["calibrated_prob_isotonic"] = calibrated_iso

    for m in model_names:
        data_dict[f"raw_score_{m}"] = raw_scores[m]
        data_dict[f"pct_score_{m}"] = pct_scores[m]

    results_df = pd.DataFrame(data_dict, index=X.index)

    # Apply thresholds
    thresholds = components.get("thresholds", {})
    ea_thresh = thresholds.get("E_A", {})
    for key, info in ea_thresh.items():
        t = info["threshold"]
        results_df[f"flag_{key}"] = flag(ea_scores, t)

    return results_df
