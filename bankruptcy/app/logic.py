"""
app/logic.py

Pure calculation functions for the bankruptcy risk interactive application.
This module is strictly decoupled from the UI framework and never accesses test data.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import scoring

DEFAULT_DEFAULTS_PATH = APP_DIR / "assets" / "defaults.json"
DEFAULT_FINAL_DIR = PROJECT_ROOT / "outputs" / "models" / "final"
DEFAULT_CV_TABLE_PATH = PROJECT_ROOT / "outputs" / "tables" / "operating_points_for_report.csv"

REQUIRED_TOP_KEYS = (
    "meta",
    "raw_columns",
    "model_columns",
    "fill_columns",
    "unused_columns",
    "medians",
    "inputs",
    "presets",
)

REQUIRED_INPUT_KEYS = (
    "min",
    "p01",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "p99",
    "max",
    "quantile_grid",
    "tier",
)


def load_defaults(path: Path | str = DEFAULT_DEFAULTS_PATH) -> dict[str, Any]:
    """Load and validate app/assets/defaults.json."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Defaults file not found at {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Defaults JSON root must be a dictionary.")

    for k in REQUIRED_TOP_KEYS:
        if k not in data:
            raise KeyError(f"Missing required top-level key in defaults.json: {k!r}")

    inputs = data["inputs"]
    if not isinstance(inputs, dict) or len(inputs) == 0:
        raise ValueError("defaults['inputs'] must be a non-empty dictionary.")

    for feat, info in inputs.items():
        for req_k in REQUIRED_INPUT_KEYS:
            if req_k not in info:
                raise KeyError(f"Missing required key {req_k!r} for input feature {feat!r}")

    return data


def load_components(final_dir: Path | str = DEFAULT_FINAL_DIR) -> dict[str, Any]:
    """Load frozen model components via src.scoring.load_final_components."""
    return scoring.load_final_components(final_dir)


def _validate_numeric_value(feat: str, val: Any) -> float:
    """Validate that a feature input value is a finite real number."""
    if isinstance(val, bool) or val is None or not isinstance(val, (int, float, np.number)):
        raise ValueError(f"Invalid non-numeric value for feature {feat!r}: {val!r}")
    fval = float(val)
    if np.isnan(fval) or np.isinf(fval):
        raise ValueError(f"NaN or Inf value is not allowed for feature {feat!r}: {val!r}")
    return fval


def build_full_row(user_inputs: dict[str, Any], defaults: dict[str, Any]) -> pd.DataFrame:
    """
    Construct a 1-row, 95-column DataFrame ordered identically to raw_columns.
    Unspecified columns are filled using training medians.
    Raises ValueError on unknown feature names, NaN, Inf, or non-numeric values.
    """
    if not isinstance(user_inputs, dict):
        raise TypeError("user_inputs must be a dictionary.")

    raw_columns: list[str] = list(defaults["raw_columns"])
    raw_set = set(raw_columns)
    medians: dict[str, float] = defaults["medians"]

    row_dict: dict[str, float] = {col: float(medians[col]) for col in raw_columns}

    for feat, val in user_inputs.items():
        if feat not in raw_set:
            raise ValueError(f"Unknown feature name in user_inputs: {feat!r}")
        row_dict[feat] = _validate_numeric_value(feat, val)

    return pd.DataFrame([row_dict], columns=raw_columns)


def score_rows(df: pd.DataFrame, components: dict[str, Any]) -> pd.DataFrame:
    """Score a DataFrame of rows using src.scoring.score_dataset."""
    return scoring.score_dataset(df, components)


def scale_warning(feature: str, defaults: dict[str, Any]) -> str | None:
    """Return warning message if feature has wide-scale encoding flag, else None."""
    inputs_meta = defaults.get("inputs", {})
    if feature in inputs_meta and inputs_meta[feature].get("scale_flag", False):
        return "此特徵在資料中有跨越多個數量級的編碼值，建議使用分位數滑桿，輸入前請確認尺度。"
    return None


def _interpolate_percentile_to_value(p: float, quantile_grid: list[float] | np.ndarray) -> float:
    """Linearly interpolate a percentile in [0, 100] to a feature value using quantile_grid."""
    q = np.clip(float(p), 0.0, 100.0)
    g = np.asarray(quantile_grid, dtype=float)
    if len(g) != 101:
        raise ValueError(f"quantile_grid must have length 101, got {len(g)}")
    k = int(np.floor(q))
    if k >= 100:
        return float(g[100])
    alpha = q - k
    g_k = g[k]
    g_k1 = g[k + 1]
    if g_k == g_k1:
        return float(g_k)
    return float(g_k + alpha * (g_k1 - g_k))


def values_from_percentiles(
    percentiles: dict[str, float] | float,
    defaults: dict[str, Any],
    feature: str | None = None,
) -> dict[str, float] | float:
    """
    Convert percentiles (0 to 100) to feature values using quantile_grid.
    Supports dictionary of {feature: percentile} or single float with feature name.
    Values outside [0, 100] are clamped.
    """
    inputs_meta = defaults["inputs"]
    if isinstance(percentiles, (int, float, np.number)):
        if feature is None:
            raise ValueError("feature name must be specified when percentiles is a single number.")
        if feature not in inputs_meta:
            raise ValueError(f"Unknown feature {feature!r}")
        return _interpolate_percentile_to_value(float(percentiles), inputs_meta[feature]["quantile_grid"])

    if not isinstance(percentiles, dict):
        raise TypeError("percentiles must be a dictionary or a number.")

    res: dict[str, float] = {}
    for feat, p in percentiles.items():
        if feat not in inputs_meta:
            raise ValueError(f"Unknown feature {feat!r}")
        res[feat] = _interpolate_percentile_to_value(float(p), inputs_meta[feat]["quantile_grid"])
    return res


def single_feature_effects(
    user_inputs: dict[str, Any],
    defaults: dict[str, Any],
    components: dict[str, Any],
) -> pd.DataFrame:
    """
    Evaluate single-feature replacement effects across all 18 input features in one batch (1 + 18 + 1 + 18 = 38 rows).
    - delta_drop: starting from user's full row, replace feature i with median (base_score - replaced_score).
    - delta_add: starting from all-median row, replace feature i with user's value (score_add - median_base_score).
    Returns a DataFrame sorted by |delta_drop| descending.
    """
    base_df = build_full_row(user_inputs, defaults)
    base_row = base_df.iloc[0].to_dict()
    raw_columns: list[str] = list(defaults["raw_columns"])
    medians: dict[str, float] = defaults["medians"]
    input_features: list[str] = list(defaults["inputs"].keys())

    # Row 0: user's base row
    batch_rows: list[dict[str, float]] = [base_row]
    # Rows 1..18: drop feature i to median
    for feat in input_features:
        mod_row = dict(base_row)
        mod_row[feat] = float(medians[feat])
        batch_rows.append(mod_row)

    # Row 19: all medians baseline row
    all_medians_row = {col: float(medians[col]) for col in raw_columns}
    batch_rows.append(all_medians_row)

    # Rows 20..37: add feature i from user's row into all medians row
    for feat in input_features:
        mod_row = dict(all_medians_row)
        mod_row[feat] = float(base_row[feat])
        batch_rows.append(mod_row)

    batch_df = pd.DataFrame(batch_rows, columns=raw_columns)
    scored_df = score_rows(batch_df, components)
    scores = scored_df["score_E_A"].to_numpy(dtype=float)

    base_score = float(scores[0])
    all_medians_score = float(scores[19])

    records: list[dict[str, Any]] = []
    for idx, feat in enumerate(input_features):
        curr_val = float(base_row[feat])
        med_val = float(medians[feat])

        score_drop = float(scores[1 + idx])
        delta_drop = 0.0 if curr_val == med_val else float(base_score - score_drop)

        score_add = float(scores[20 + idx])
        delta_add = 0.0 if curr_val == med_val else float(score_add - all_medians_score)

        records.append(
            {
                "feature": feat,
                "current_value": curr_val,
                "median_value": med_val,
                "base_score": base_score,
                "replaced_score": score_drop,
                "delta_drop": delta_drop,
                "delta_add": delta_add,
                "delta": delta_drop,
                "abs_delta": abs(delta_drop),
            }
        )

    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(by="abs_delta", ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    return result_df


def what_if_curve(
    feature: str,
    user_inputs: dict[str, Any],
    defaults: dict[str, Any],
    components: dict[str, Any],
    n: int = 40,
    grid: str = "percentile",
) -> dict[str, Any]:
    """
    Sweep `feature` across n points while holding other inputs fixed.
    - grid="percentile": n evenly spaced percentiles in [1, 99], converted via quantile_grid (deduplicated).
    - grid="value": n evenly spaced feature values in [p01, p99].
    Evaluates all points in a single batch inference call.
    """
    inputs_meta = defaults["inputs"]
    if feature not in inputs_meta:
        raise ValueError(f"Feature {feature!r} is not one of the configurable input features.")
    if int(n) < 2:
        raise ValueError("Number of grid points n must be at least 2.")

    base_df = build_full_row(user_inputs, defaults)
    base_row = base_df.iloc[0].to_dict()
    raw_columns: list[str] = list(defaults["raw_columns"])
    f_info = inputs_meta[feature]

    if grid == "percentile":
        pct_points = np.linspace(1.0, 99.0, int(n))
        pts_list: list[tuple[float, float]] = []
        seen_vals = set()
        for p in pct_points:
            v = _interpolate_percentile_to_value(float(p), f_info["quantile_grid"])
            if v not in seen_vals:
                seen_vals.add(v)
                pts_list.append((float(p), float(v)))
        if not pts_list:
            pts_list = [(50.0, float(f_info["p50"]))]
        percentiles = [p for p, _ in pts_list]
        grid_values = [v for _, v in pts_list]
    elif grid == "value":
        grid_values = np.linspace(float(f_info["p01"]), float(f_info["p99"]), int(n)).tolist()
        percentiles = [_interpolate_quantile(v, f_info["quantile_grid"]) for v in grid_values]
    else:
        raise ValueError(f"Unknown grid mode {grid!r}, must be 'percentile' or 'value'.")

    batch_rows: list[dict[str, float]] = []
    for val in grid_values:
        r = dict(base_row)
        r[feature] = float(val)
        batch_rows.append(r)

    batch_df = pd.DataFrame(batch_rows, columns=raw_columns)
    scored_df = score_rows(batch_df, components)
    scores = scored_df["score_E_A"].to_numpy(dtype=float)

    curve_df = pd.DataFrame(
        {
            "percentile": percentiles,
            "feature_value": grid_values,
            "score_E_A": scores,
        }
    )

    ea_thresholds = {
        op_key: float(op_info["threshold"])
        for op_key, op_info in components["thresholds"]["E_A"].items()
    }

    return {
        "feature": feature,
        "grid": grid,
        "percentiles": percentiles,
        "values": grid_values,
        "scores": scores.tolist(),
        "curve_df": curve_df,
        "thresholds": ea_thresholds,
        "current_value": float(base_row[feature]),
    }


def _interpolate_quantile(val: float, quantile_grid: list[float] | np.ndarray) -> float:
    """
    Piecewise linear interpolation on the 101-point quantile_grid (percentiles 0..100).
    Values <= min map to 0.0; values >= max map to 100.0; ties in the interior use the midpoint index.
    """
    g = np.asarray(quantile_grid, dtype=float)
    if len(g) != 101:
        raise ValueError(f"quantile_grid must have length 101, got {len(g)}")
    if val <= g[0]:
        return 0.0
    if val >= g[-1]:
        return 100.0

    left = int(np.searchsorted(g, val, side="left"))
    right = int(np.searchsorted(g, val, side="right"))
    if left < right:
        return float(0.5 * (left + right - 1))

    k = left - 1
    denom = g[k + 1] - g[k]
    if denom <= 0.0:
        return float(k)
    pct = float(k + (val - g[k]) / denom)
    return float(np.clip(pct, 0.0, 100.0))


def position_table(user_inputs: dict[str, Any], defaults: dict[str, Any]) -> pd.DataFrame:
    """
    Compute the training distribution percentile (0 to 100) for each of the 18 input features
    using linear interpolation on quantile_grid.
    """
    base_df = build_full_row(user_inputs, defaults)
    base_row = base_df.iloc[0].to_dict()

    records: list[dict[str, Any]] = []
    for feat, info in defaults["inputs"].items():
        val = float(base_row[feat])
        pct = _interpolate_quantile(val, info["quantile_grid"])
        records.append(
            {
                "feature": feat,
                "value": val,
                "percentile": pct,
                "min": float(info["min"]),
                "p01": float(info["p01"]),
                "p50": float(info["p50"]),
                "p99": float(info["p99"]),
                "max": float(info["max"]),
            }
        )

    return pd.DataFrame(records)


def input_warnings(user_inputs: dict[str, Any], defaults: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Check input values against training ranges:
    - Outside [min, max] -> level="out_of_range"
    - Outside [p01, p99] (but within [min, max]) -> level="range"
    - Inside [p01, p99] -> no warning
    """
    base_df = build_full_row(user_inputs, defaults)
    base_row = base_df.iloc[0].to_dict()

    warnings_list: list[dict[str, Any]] = []
    for feat, info in defaults["inputs"].items():
        val = float(base_row[feat])
        f_min = float(info["min"])
        f_p01 = float(info["p01"])
        f_p99 = float(info["p99"])
        f_max = float(info["max"])

        if val < f_min or val > f_max:
            warnings_list.append(
                {
                    "feature": feat,
                    "value": val,
                    "level": "out_of_range",
                    "min": f_min,
                    "p01": f_p01,
                    "p99": f_p99,
                    "max": f_max,
                    "message": f"{feat} 超出訓練資料全距 [{f_min:.6g}, {f_max:.6g}]，推論結果可能不可靠。",
                }
            )
        elif val < f_p01 or val > f_p99:
            warnings_list.append(
                {
                    "feature": feat,
                    "value": val,
                    "level": "range",
                    "min": f_min,
                    "p01": f_p01,
                    "p99": f_p99,
                    "max": f_max,
                    "message": f"{feat} 超出訓練資料常見範圍 (p01~p99: [{f_p01:.6g}, {f_p99:.6g}])。",
                }
            )

    return warnings_list


def operating_point_table(
    components: dict[str, Any],
    cv_table_path: Path | str = DEFAULT_CV_TABLE_PATH,
    flags: dict[str, Any] | pd.Series | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge E_A thresholds from components['thresholds']['E_A'] with CV metrics from
    operating_points_for_report.csv.
    Asserts that threshold and fixed_threshold match across both sources; raises ValueError otherwise.
    Optionally attaches a 'flagged' boolean column if flags is provided.
    """
    if "thresholds" not in components or "E_A" not in components["thresholds"]:
        raise KeyError("Missing 'E_A' thresholds in components['thresholds'].")

    ea_thresh = components["thresholds"]["E_A"]
    cv_path = Path(cv_table_path)
    if not cv_path.exists():
        raise FileNotFoundError(f"Operating points CSV not found at {cv_path}")

    cv_df = pd.read_csv(cv_path)
    required_csv_cols = {
        "group",
        "target_recall",
        "fixed_threshold",
        "actual_recall_mean",
        "alert_ratio_mean",
        "alert_proportion_mean",
        "precision_mean",
        "attainment_rate",
    }
    missing_cols = required_csv_cols - set(cv_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {cv_path}: {sorted(missing_cols)}")

    ea_cv = cv_df[cv_df["group"] == "E_A"].copy()
    if ea_cv.empty:
        raise ValueError("No rows found for group 'E_A' in operating_points_for_report.csv.")

    flag_lookup: dict[str, bool] | None = None
    if flags is not None:
        flag_lookup = {}
        if isinstance(flags, pd.DataFrame):
            if len(flags) != 1:
                raise ValueError("When flags is a DataFrame, it must have exactly 1 row.")
            flag_source = flags.iloc[0].to_dict()
        elif isinstance(flags, pd.Series):
            flag_source = flags.to_dict()
        elif isinstance(flags, dict):
            flag_source = flags
        else:
            raise TypeError("flags must be a dict, Series, 1-row DataFrame, or None.")

        for op_key in ea_thresh.keys():
            if op_key in flag_source:
                flag_lookup[op_key] = bool(flag_source[op_key])
            elif f"flag_{op_key}" in flag_source:
                flag_lookup[op_key] = bool(flag_source[f"flag_{op_key}"])
            else:
                raise KeyError(f"Flag for operating point {op_key!r} not found in provided flags.")

    rows: list[dict[str, Any]] = []
    for op_key, t_info in ea_thresh.items():
        target_r = float(t_info["target_recall"])
        t_json = float(t_info["threshold"])

        matched = ea_cv[np.isclose(ea_cv["target_recall"].astype(float), target_r, atol=1e-6)]
        if len(matched) != 1:
            raise ValueError(
                f"Expected exactly 1 CV row for E_A target_recall={target_r} ({op_key}), found {len(matched)}."
            )

        cv_row = matched.iloc[0]
        t_csv = float(cv_row["fixed_threshold"])

        if not np.isclose(t_json, t_csv, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Threshold mismatch for {op_key}: thresholds.json has {t_json}, "
                f"operating_points_for_report.csv has {t_csv}."
            )

        rec: dict[str, Any] = {
            "operating_point": op_key,
            "target_recall": target_r,
            "recommended_margin": float(t_info.get("recommended_margin", cv_row.get("recommended_margin", 0.0))),
            "threshold": t_json,
            "fixed_threshold": t_csv,
            "actual_recall_mean": float(cv_row["actual_recall_mean"]),
            "alert_ratio_mean": float(cv_row["alert_ratio_mean"]),
            "alert_proportion_mean": float(cv_row["alert_proportion_mean"]),
            "precision_mean": float(cv_row["precision_mean"]),
            "attainment_rate": float(cv_row["attainment_rate"]),
        }
        if flag_lookup is not None:
            rec["flagged"] = flag_lookup[op_key]

        rows.append(rec)

    return pd.DataFrame(rows)


def describe_score(score: float) -> str:
    """
    Return a plain-language description of risk percentile score `score` (in [0, 1]).
    Explains that it is the average of percentiles from the three models.
    """
    val = _validate_numeric_value("score", score)
    return (
        f"平均百分位分數 {val:.2f}"
        "（三個模型各自把這筆輸入排在訓練資料參考分布的百分位，再取平均）"
    )


def describe_percentile(p: float) -> str:
    """Backward compatibility alias for describe_score."""
    return describe_score(p)


def _get_ensemble_reference_scores(components: dict[str, Any]) -> np.ndarray:
    """Compute and cache reference ensemble score distribution across training samples."""
    if "_ensemble_reference_scores" in components:
        return components["_ensemble_reference_scores"]

    ref_m3 = components["references"]["M3"]
    ref_m4 = components["references"]["M4"]
    ref_m5 = components["references"]["M5"]

    n_total = len(ref_m3)
    if n_total % 2 == 0 and n_total > 5455:
        n = n_total // 2
        m3_score = (ref_m3[:n] + ref_m3[n:]) / 2.0
        m4_score = (ref_m4[:n] + ref_m4[n:]) / 2.0
        m5_score = (ref_m5[:n] + ref_m5[n:]) / 2.0
    else:
        m3_score = ref_m3
        m4_score = ref_m4
        m5_score = ref_m5

    pct_3 = scoring.to_percentile(m3_score, ref_m3)
    pct_4 = scoring.to_percentile(m4_score, ref_m4)
    pct_5 = scoring.to_percentile(m5_score, ref_m5)
    ea_scores = (pct_3 + pct_4 + pct_5) / 3.0
    components["_ensemble_reference_scores"] = ea_scores
    return ea_scores


def training_share_at_or_above(score: float, components: dict[str, Any]) -> float:
    """Return proportion of training reference samples with ensemble score >= score."""
    val = _validate_numeric_value("score", score)
    ref_scores = _get_ensemble_reference_scores(components)
    return float(np.mean(ref_scores >= val))

