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
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("scoring")

# Ensure config is importable regardless of working directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import (
    MAX_MISSING_FEATURE_RATIO,
    MIN_BATCH_SIZE_FOR_TOP_K,
    TOP_K_PERCENT,
)
from src.metrics import compute_top_k_quota


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
    if not cal_platt.exists():
        raise FileNotFoundError(f"Calibrator file '{cal_platt}' does not exist.")
    components["calibrators"]["platt"] = joblib.load(cal_platt)

    cal_iso = f_dir / "calibrator_isotonic.joblib"
    if not cal_iso.exists():
        raise FileNotFoundError(f"Calibrator file '{cal_iso}' does not exist.")
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
    include_threshold_flags: bool = True,
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

    # Calibrate probability: calibrators must not be missing or None
    calibrators = components.get("calibrators")
    if not calibrators:
        raise KeyError("Calibrators dictionary missing from components['calibrators'].")
    if "platt" not in calibrators or calibrators["platt"] is None:
        raise KeyError("Calibrator 'platt' missing from components['calibrators'].")
    if "isotonic" not in calibrators or calibrators["isotonic"] is None:
        raise KeyError("Calibrator 'isotonic' missing from components['calibrators'].")

    cal_platt = calibrators["platt"]
    calibrated_platt = cal_platt.predict_proba(ea_scores.reshape(-1, 1))[:, 1]

    cal_iso = calibrators["isotonic"]
    calibrated_iso = cal_iso.predict(ea_scores)

    data_dict = {
        "score_E_A": ea_scores,
        "score_M3": pct_scores["M3"],
        "calibrated_prob_platt": calibrated_platt,
        "calibrated_prob_isotonic": calibrated_iso,
    }

    for m in model_names:
        data_dict[f"raw_score_{m}"] = raw_scores[m]
        data_dict[f"pct_score_{m}"] = pct_scores[m]

    results_df = pd.DataFrame(data_dict, index=X.index)

    # Apply thresholds: raise explicit exception if requested but thresholds missing
    if include_threshold_flags:
        thresholds = components.get("thresholds")
        if not thresholds:
            raise KeyError("Thresholds configuration missing from components['thresholds'].")
        if "E_A" not in thresholds or not thresholds["E_A"]:
            raise KeyError("Thresholds configuration for 'E_A' missing from components['thresholds'].")
        ea_thresh = thresholds["E_A"]
        for key, info in ea_thresh.items():
            t = info["threshold"]
            results_df[f"flag_{key}"] = flag(ea_scores, t)

    return results_df


def check_and_deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column whitespace and strictly checks for duplicate column names:
    - Handles in-memory duplicate column names (df.columns.duplicated()).
    - Handles pandas read_csv mangled duplicate column names (e.g. 'col.1', 'col.2' when 'col' exists).
    - If duplicates have identical values across all rows, deduplicates keeping the first occurrence and logs a warning.
    - If duplicates have conflicting values, raises ValueError listing the conflicting column names.
    """
    import re
    df = df.copy()
    df.columns = df.columns.str.strip()

    # 1. In-memory exact duplicate column names
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].unique()
        conflicting = []
        for col in dup_cols:
            sub = df.loc[:, col]
            first_col = sub.iloc[:, 0]
            if not all(first_col.equals(sub.iloc[:, i]) for i in range(1, sub.shape[1])):
                conflicting.append(col)
        if conflicting:
            raise ValueError(
                f"Input data contains conflicting duplicate column(s) with differing values: {conflicting}"
            )
        for col in dup_cols:
            logger.warning(f"Duplicate column '{col}' detected with identical values; keeping first occurrence.")
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # 2. Mangled duplicate columns from CSV reading (e.g. 'Feature.1')
    mangled_dupes: dict[str, list[str]] = {}
    for col in list(df.columns):
        m = re.match(r"^(.*?)\.\d+$", col)
        if m:
            base = m.group(1).strip()
            if base in df.columns:
                mangled_dupes.setdefault(base, []).append(col)

    if mangled_dupes:
        conflicting = []
        to_drop = []
        for base, dupe_cols in mangled_dupes.items():
            base_col = df[base]
            has_conflict = False
            for d in dupe_cols:
                if not base_col.equals(df[d]):
                    has_conflict = True
                    break
            if has_conflict:
                conflicting.append(base)
            else:
                logger.warning(
                    f"Duplicate column '{base}' detected with identical values (mangled as {dupe_cols}); keeping first occurrence."
                )
                to_drop.extend(dupe_cols)
        if conflicting:
            raise ValueError(
                f"Input data contains conflicting duplicate column(s) with differing values: {conflicting}"
            )
        if to_drop:
            df = df.drop(columns=to_drop)

    return df


def load_mvp_components(
    final_dir: Path | str | None = None,
    extras_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Loads full MVP production components from outputs/models/final/ and outputs/models/mvp_extras/.

    Guarantees:
    - Zero dependency on e5_cache/ or logs/e2_cache/.
    - Supports models: E_A (ensemble of M3, M4, M5), M3 (ExtraTrees), M4 (XGBoost), M1 (Regularized Logistic Regression).
    - Includes 95-feature training medians (train_medians_95) for robust out-of-pipeline missing value imputation.
    """
    if final_dir is None:
        final_dir = project_root / "outputs" / "models" / "final"
    if extras_dir is None:
        extras_dir = project_root / "outputs" / "models" / "mvp_extras"

    f_dir = Path(final_dir)
    e_dir = Path(extras_dir)

    if not f_dir.exists():
        raise FileNotFoundError(f"Final model directory '{f_dir}' does not exist.")
    if not e_dir.exists():
        raise FileNotFoundError(f"MVP extras directory '{e_dir}' does not exist.")

    components: dict[str, Any] = {
        "models": {},
        "references": {},
        "calibrators": {},
    }

    # Load tree models (M3, M4, M5) from final/
    for m in ["M3", "M4", "M5"]:
        model_path = f_dir / f"model_{m}.joblib"
        if model_path.exists():
            pipe = joblib.load(model_path)
            clf = pipe.named_steps.get("classifier")
            if clf is not None and hasattr(clf, "n_jobs"):
                clf.set_params(n_jobs=1)
            components["models"][m] = pipe
        ref_path = f_dir / f"reference_{m}.npy"
        if ref_path.exists():
            components["references"][m] = np.load(ref_path)

    # Load linear benchmark model M1 from mvp_extras/
    m1_path = e_dir / "model_M1.joblib"
    if m1_path.exists():
        pipe_m1 = joblib.load(m1_path)
        clf = pipe_m1.named_steps.get("classifier")
        if clf is not None and hasattr(clf, "n_jobs"):
            clf.set_params(n_jobs=1)
        components["models"]["M1"] = pipe_m1
    ref_m1_path = e_dir / "reference_M1.npy"
    if ref_m1_path.exists():
        components["references"]["M1"] = np.load(ref_m1_path)

    # Load E_A reference distribution (10,910 empirical distribution)
    ref_ea_path = e_dir / "reference_E_A.npy"
    if ref_ea_path.exists():
        components["references"]["E_A"] = np.load(ref_ea_path)
    else:
        pct_m3 = to_percentile(components["references"]["M3"], components["references"]["M3"])
        pct_m4 = to_percentile(components["references"]["M4"], components["references"]["M4"])
        pct_m5 = to_percentile(components["references"]["M5"], components["references"]["M5"])
        components["references"]["E_A"] = (pct_m3 + pct_m4 + pct_m5) / 3.0

    # Load thresholds from mvp_extras/thresholds_extras.json
    thresh_path = e_dir / "thresholds_extras.json"
    if thresh_path.exists():
        with open(thresh_path, "r", encoding="utf-8") as f:
            components["thresholds"] = json.load(f)
    else:
        with open(f_dir / "thresholds.json", "r", encoding="utf-8") as f:
            components["thresholds"] = json.load(f)

    # Load Calibrators
    cal_platt = f_dir / "calibrator_platt.joblib"
    if cal_platt.exists():
        components["calibrators"]["platt"] = joblib.load(cal_platt)

    cal_iso = f_dir / "calibrator_isotonic.joblib"
    if cal_iso.exists():
        components["calibrators"]["isotonic"] = joblib.load(cal_iso)

    for m in ["M1", "M3", "M4"]:
        cal_m = e_dir / f"calibrator_{m}_platt.joblib"
        if cal_m.exists():
            components["calibrators"][m] = joblib.load(cal_m)

    # Load feature manifest (raw 95 features and 95 training medians)
    manifest_path = e_dir / "feature_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        components["raw_features_95"] = manifest.get("raw_features_95", [])
        components["train_medians_95"] = manifest.get("train_medians_95", {})

    metadata_path = f_dir / "version_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            components["metadata"] = json.load(f)

    return components


def score_mvp_batch(
    X: pd.DataFrame,
    model_name: str = "E_A",
    operating_point: str = "R90",
    allow_missing: bool = False,
    components: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    End-to-end MVP batch and single-case scoring with rigorous missingness safeguards.

    Architectural design:
    - Normalization: strips column whitespace, handles permuted order, preserves extra columns,
      and deduplicates duplicate column names keeping the first occurrence.
    - Missingness protection:
      * When allow_missing=False (default, matching b3ecc2c), any missing feature column or NaN
        in required features raises ValueError with the exact b3ecc2c exception message format.
      * When allow_missing=True, features with missingness <= 20% are imputed using train_medians_95.
      * Rows with missingness > 20% are flagged as 'rejected_high_missing'; their risk scores and ranks
        are set to None/NaN and excluded from the valid ranking population N_valid.
      * Single-record inference: if a single record has missingness > 20%, ValueError is raised directly.
    - Capacity ranking: batch_rank (pessimistic max rank) and in_batch_top_10_pct share compute_top_k_quota
      (k_percent=TOP_K_PERCENT, method='floor') and require N_valid >= MIN_BATCH_SIZE_FOR_TOP_K (25);
      otherwise in_batch_top_10_pct is set to None.
    - Methodological Notes:
      * Ties: batch_rank reflects pessimistic competitive rank; in_batch_top_10_pct resolves quota boundary
        ties stably by input appearance order (method='first') so that the number of flagged records strictly
        equals k_quota.
      * Calibrated score: calibrated_score is Platt-scaled to the training dataset's prevalence,
        NOT an unadjusted real-world population bankruptcy probability. If deployed in lower-prevalence environments,
        prior odds ratio adjustment is required.
      * Risk score scales: risk_score scale varies across models (e.g. E_A vs M4 vs M1);
        cross-model relative risk should be referenced via reference_percentile.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input X must be a pandas DataFrame.")

    # Normalize column names: strip whitespace and check duplicates with value verification
    X = check_and_deduplicate_columns(X)

    valid_models = ["E_A", "M3", "M4", "M1"]
    if model_name not in valid_models:
        raise ValueError(f"Invalid model_name '{model_name}'. Supported: {valid_models}")

    valid_ops = ["R80", "R90", "R95", "R100"]
    if operating_point not in valid_ops:
        raise ValueError(f"Invalid operating_point '{operating_point}'. Supported: {valid_ops}")

    if components is None:
        components = load_mvp_components()

    raw_features_95 = components.get("raw_features_95", [])
    train_medians = components.get("train_medians_95", {})

    # Detect extra / pass-through columns
    extra_cols = [c for c in X.columns if c not in raw_features_95 and c != "Bankrupt?"]

    # When allow_missing=False, check for missing columns or NaNs and raise b3ecc2c error format
    missing_cols_in_df = [c for c in raw_features_95 if c not in X.columns]
    nan_cols_in_df = [c for c in raw_features_95 if c in X.columns and X[c].isna().any()]
    missing_features = sorted(list(set(missing_cols_in_df + nan_cols_in_df)))

    if not allow_missing and missing_features:
        err_msg = (
            f"Input data is missing {len(missing_features)} required feature(s) for scoring: "
            f"{missing_features}"
        )
        raise ValueError(err_msg)

    if len(missing_cols_in_df) == len(raw_features_95):
        raise ValueError("Input data contains none of the required 95 financial features.")

    # Check row-level missingness
    n_rows = len(X)
    is_single_sample = (n_rows == 1)

    # Build evaluation DataFrame for the 95 features in canonical order
    X_95 = pd.DataFrame(index=X.index)
    for col in raw_features_95:
        if col in X.columns:
            X_95[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X_95[col] = np.nan

    missing_counts = X_95.isna().sum(axis=1).values
    missing_ratios = missing_counts / 95.0

    # Determine status per row
    is_rejected = missing_ratios > MAX_MISSING_FEATURE_RATIO

    if is_single_sample and is_rejected[0]:
        raise ValueError(
            f"Single sample request rejected: missing feature ratio {missing_ratios[0]:.1%} "
            f"exceeds allowed limit ({MAX_MISSING_FEATURE_RATIO:.0%})."
        )

    # Impute missing values for valid rows using training medians
    valid_mask = ~is_rejected
    n_valid = int(np.sum(valid_mask))

    risk_scores = np.full(n_rows, np.nan, dtype=float)
    ref_percentiles = np.full(n_rows, np.nan, dtype=float)
    calibrated_scores = np.full(n_rows, np.nan, dtype=float)
    alerts = [None] * n_rows
    in_ref_top10 = [None] * n_rows
    batch_ranks = [None] * n_rows
    in_batch_top10 = [None] * n_rows

    th_entry = components["thresholds"][model_name][operating_point]
    threshold_val = float(th_entry["threshold"])
    margin_validated = bool(th_entry.get("margin_validated", False))
    threshold_note = str(th_entry.get("threshold_note", th_entry.get("note", "")))

    if n_valid > 0:
        X_valid = X_95.loc[valid_mask].copy()
        for col in raw_features_95:
            if X_valid[col].isna().any():
                fill_val = train_medians.get(col, 0.0)
                X_valid[col] = X_valid[col].fillna(fill_val)

        # Predict based on selected model
        if model_name == "E_A":
            raw_m3 = components["models"]["M3"].predict_proba(X_valid)[:, 1]
            raw_m4 = components["models"]["M4"].predict_proba(X_valid)[:, 1]
            raw_m5 = components["models"]["M5"].predict_proba(X_valid)[:, 1]
            pct_m3 = to_percentile(raw_m3, components["references"]["M3"])
            pct_m4 = to_percentile(raw_m4, components["references"]["M4"])
            pct_m5 = to_percentile(raw_m5, components["references"]["M5"])
            ea_scores = (pct_m3 + pct_m4 + pct_m5) / 3.0

            valid_risk = ea_scores
            valid_pct = to_percentile(ea_scores, components["references"]["E_A"])
            cal_platt = components["calibrators"].get("platt")
            if cal_platt is not None:
                valid_cal = cal_platt.predict_proba(ea_scores.reshape(-1, 1))[:, 1]
            else:
                valid_cal = np.full(n_valid, np.nan)

        elif model_name == "M3":
            raw_m3 = components["models"]["M3"].predict_proba(X_valid)[:, 1]
            pct_m3 = to_percentile(raw_m3, components["references"]["M3"])
            valid_risk = pct_m3
            valid_pct = pct_m3
            cal_m3 = components["calibrators"].get("M3")
            valid_cal = cal_m3.predict_proba(raw_m3.reshape(-1, 1))[:, 1] if cal_m3 is not None else np.full(n_valid, np.nan)

        elif model_name == "M4":
            raw_m4 = components["models"]["M4"].predict_proba(X_valid)[:, 1]
            valid_risk = raw_m4
            valid_pct = to_percentile(raw_m4, components["references"]["M4"])
            cal_m4 = components["calibrators"].get("M4")
            valid_cal = cal_m4.predict_proba(raw_m4.reshape(-1, 1))[:, 1] if cal_m4 is not None else np.full(n_valid, np.nan)

        elif model_name == "M1":
            raw_m1 = components["models"]["M1"].predict_proba(X_valid)[:, 1]
            valid_risk = raw_m1
            valid_pct = to_percentile(raw_m1, components["references"]["M1"])
            cal_m1 = components["calibrators"].get("M1")
            valid_cal = cal_m1.predict_proba(raw_m1.reshape(-1, 1))[:, 1] if cal_m1 is not None else np.full(n_valid, np.nan)

        risk_scores[valid_mask] = valid_risk
        ref_percentiles[valid_mask] = valid_pct
        calibrated_scores[valid_mask] = valid_cal

        # Flags for valid samples
        valid_indices = np.where(valid_mask)[0]
        for idx_pos, row_idx in enumerate(valid_indices):
            alerts[row_idx] = bool(valid_risk[idx_pos] >= threshold_val)
            in_ref_top10[row_idx] = bool(valid_pct[idx_pos] >= (1.0 - TOP_K_PERCENT))

        # Batch ranking among valid samples (pessimistic max rank)
        valid_s = pd.Series(valid_risk)
        pessimistic_ranks = valid_s.rank(ascending=False, method="max").astype(int)

        # Top 10% quota within batch: flagged count strictly equals k_quota
        # Ties crossing quota boundary are stably resolved by input sequence order (method='first')
        if n_valid >= MIN_BATCH_SIZE_FOR_TOP_K:
            k_quota = compute_top_k_quota(n_valid, k_percent=TOP_K_PERCENT, method="floor")
        else:
            k_quota = None

        if k_quota is not None and k_quota > 0:
            stable_ranks = valid_s.rank(ascending=False, method="first").astype(int)
            is_top_k = (stable_ranks <= k_quota)
        else:
            is_top_k = None

        for idx_pos, row_idx in enumerate(valid_indices):
            batch_ranks[row_idx] = int(pessimistic_ranks.iloc[idx_pos])
            if is_top_k is not None:
                in_batch_top10[row_idx] = bool(is_top_k.iloc[idx_pos])
            else:
                in_batch_top10[row_idx] = None

    # Construct result dataframe with strictly standard column names and ordering
    out_dict = {}
    for col in extra_cols:
        out_dict[col] = X[col].values
    if "Bankrupt?" in X.columns:
        out_dict["Bankrupt?"] = X["Bankrupt?"].values

    out_dict["risk_score"] = risk_scores
    out_dict["reference_percentile"] = ref_percentiles
    out_dict["calibrated_score"] = calibrated_scores
    out_dict["in_reference_top_10_pct"] = in_ref_top10
    out_dict["alert"] = alerts
    out_dict["batch_rank"] = batch_ranks
    out_dict["in_batch_top_10_pct"] = in_batch_top10
    out_dict["missing_feature_count"] = missing_counts
    out_dict["missing_feature_ratio"] = missing_ratios
    out_dict["status"] = ["rejected_high_missing" if r else "ok" for r in is_rejected]
    out_dict["margin_validated"] = [margin_validated] * n_rows
    out_dict["threshold_note"] = [threshold_note] * n_rows

    return pd.DataFrame(out_dict, index=X.index)
