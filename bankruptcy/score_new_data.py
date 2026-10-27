"""
score_new_data.py: Standalone command-line scoring tool for new company data.

Supports:
- Models: E_A (default, 3-model rank ensemble) or M3 (ExtraTrees control)
- Operating points: R80, R90 (default), R95, R100
- Robust column handling: Whitespace stripping, missing feature detection, extra column pass-through

Usage:
    python score_new_data.py --input <path_to_input_csv> --output <path_to_output_csv> [--operating-point R90] [--model E_A]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring import load_final_components, load_mvp_components, score_dataset, score_mvp_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("score_new_data")

OPERATING_POINTS = ["R80", "R90", "R95", "R100"]
MODELS = ["E_A", "M3", "M4", "M1"]


def get_required_feature_names(components: dict[str, Any]) -> list[str]:
    """Retrieves expected feature names from the frozen pipeline."""
    # From model_M3 feature selector or scaler
    pipe = components["models"]["M3"]
    # The first step after FeatureSelector is DropConstantAndDuplicate,
    # which takes the 95 raw features. We can get them from version_metadata or config.
    meta = components.get("metadata", {})
    if "feature_names" in meta:
        return meta["feature_names"]
    
    # Alternatively from data/data.csv or config
    import config
    df_sample = pd.read_csv(config.DATA_PATH, nrows=1)
    df_sample.columns = df_sample.columns.str.strip()
    return [c for c in df_sample.columns if c != config.TARGET_COL]


def score_csv(
    input_path: str | Path,
    output_path: str | Path,
    operating_point: str = "R90",
    model: str = "E_A",
    components_dir: str | Path | None = None,
    allow_missing: bool = False,
    sort_by_risk: bool = False,
    high_risk_only: bool = False,
) -> pd.DataFrame:
    """
    Reads new input CSV, performs integrity checks, scores data, and writes output.
    Supports --allow-missing (default False), --sort-by-risk, and --high-risk-only filtering.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    if operating_point not in OPERATING_POINTS:
        raise ValueError(f"Invalid operating point '{operating_point}'. Supported: {OPERATING_POINTS}")

    if model not in MODELS:
        raise ValueError(f"Invalid model '{model}'. Supported: {MODELS}")

    if components_dir is None:
        components_dir = PROJECT_ROOT / "outputs" / "models" / "final"

    components = load_final_components(components_dir)
    expected_features = get_required_feature_names(components)

    logger.info(f"Loading input data from {in_path} ...")
    df = pd.read_csv(in_path)
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Identify missing required features (columns not present in input or containing NaN)
    missing_cols = [f for f in expected_features if f not in df.columns]
    nan_cols = [f for f in expected_features if f in df.columns and df[f].isna().any()]
    missing_features = sorted(list(set(missing_cols + nan_cols)))

    if not allow_missing and missing_features:
        err_msg = (
            f"Input data is missing {len(missing_features)} required feature(s) for scoring: "
            f"{missing_features}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Identify extra / identifier columns
    extra_cols = [c for c in df.columns if c not in expected_features and c != "Bankrupt?"]
    if extra_cols:
        logger.warning(
            f"Found {len(extra_cols)} extra / non-feature column(s) which will be preserved as identifiers: {extra_cols}"
        )

    # Score dataset using unified inference
    logger.info(f"Scoring {len(df)} records using model '{model}' at operating point '{operating_point}' ...")

    if allow_missing or model in ["M4", "M1"]:
        from src.scoring import load_mvp_components, score_mvp_batch
        mvp_comp = load_mvp_components()
        scored_mvp = score_mvp_batch(df, model_name=model, operating_point=operating_point, allow_missing=allow_missing, components=mvp_comp)
        risk_score = scored_mvp["risk_score"].values
        threshold_info = mvp_comp["thresholds"][model][operating_point]
        threshold_val = float(threshold_info["threshold"])
        alert_flag = scored_mvp["alert"].values
        platt_prob = scored_mvp["calibrated_score"].values
    else:
        X_features = df[expected_features].copy()
        scored_df = score_dataset(X_features, components)

        threshold_info = components["thresholds"][model][operating_point]
        threshold_val = float(threshold_info["threshold"])

        if model == "E_A":
            risk_score = scored_df["score_E_A"].values
            alert_flag = scored_df[f"flag_{operating_point}"].values
        else:  # M3
            risk_score = scored_df["score_M3"].values
            alert_flag = (risk_score >= threshold_val)

        platt_prob = scored_df.get("calibrated_prob_platt", pd.Series(np.nan, index=df.index)).values

    # Assemble output dataframe
    output_dict: dict[str, Any] = {}
    for col in extra_cols:
        output_dict[col] = df[col].values

    output_dict["risk_percentile_score"] = risk_score
    output_dict["threshold"] = threshold_val
    output_dict["alert"] = alert_flag
    output_dict["calibrated_probability_platt"] = platt_prob

    result_df = pd.DataFrame(output_dict, index=df.index)

    # Optional post-processing: high-risk filtering and risk sorting
    if high_risk_only:
        result_df = result_df[result_df["alert"] == True]
    if sort_by_risk:
        result_df = result_df.sort_values(by="risk_percentile_score", ascending=False)

    # Save pure CSV output without comment header (first line must be column header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    # Save accompanying methodological notes in *.README.txt in the same directory
    readme_content = (
        f"==============================================================================\n"
        f"方法學註記 (Methodological Notes)\n"
        f"關聯輸出檔案: {out_path.name}\n"
        f"模型: {model} | 工作點: {operating_point} | 固化門檻: {threshold_val:.4f}\n"
        f"==============================================================================\n"
        f"1. 分數為相對於訓練資料參考分布的百分位（0 代表最低風險，1 代表最高風險）。\n"
        f"2. 校準機率反映訓練資料約 3.2% 的破產率（若真實環境盛行率不同，需進行先驗勝算比調整）。\n"
        f"3. 凍結門檻僅適用於與訓練資料分布相近的資料，對真實世界新資料使用前需重新驗證。\n"
    )

    readme_path = out_path.parent / f"{out_path.stem}.README.txt"
    readme_path.write_text(readme_content, encoding="utf-8")
    # Also write with full filename extension if different (e.g. predictions.csv.README.txt)
    if out_path.suffix:
        readme_path_alt = out_path.parent / f"{out_path.name}.README.txt"
        readme_path_alt.write_text(readme_content, encoding="utf-8")

    logger.info(f"Successfully generated predictions at {out_path}")
    logger.info(f"Generated methodological README at {readme_path}")
    logger.info(f"Summary: {int(alert_flag.sum())} / {len(df)} records flagged as alert ({alert_flag.mean():.2%}).")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Score new data with frozen bankruptcy prediction models.")
    parser.add_argument("--input", required=True, type=str, help="Path to input CSV containing company features.")
    parser.add_argument("--output", required=True, type=str, help="Path to output CSV for scored results.")
    parser.add_argument(
        "--operating-point",
        default="R90",
        choices=OPERATING_POINTS,
        help="Target operating point threshold (default: R90).",
    )
    parser.add_argument(
        "--model",
        default="E_A",
        choices=MODELS,
        help="Model to use for scoring: 'E_A' (ensemble) or 'M3' (ExtraTrees) (default: E_A).",
    )
    parser.add_argument(
        "--components-dir",
        default=None,
        type=str,
        help="Directory containing frozen model components (default: outputs/models/final).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        default=False,
        help="Allow missing features up to 20% by imputing with training medians (default: False).",
    )
    parser.add_argument(
        "--sort-by-risk",
        action="store_true",
        default=False,
        help="Sort output records by risk score in descending order.",
    )
    parser.add_argument(
        "--high-risk-only",
        action="store_true",
        default=False,
        help="Filter output to only records flagged as high risk (alert == True).",
    )
    args = parser.parse_args()

    score_csv(
        input_path=args.input,
        output_path=args.output,
        operating_point=args.operating_point,
        model=args.model,
        components_dir=args.components_dir,
        allow_missing=args.allow_missing,
        sort_by_risk=args.sort_by_risk,
        high_risk_only=args.high_risk_only,
    )


if __name__ == "__main__":
    main()
