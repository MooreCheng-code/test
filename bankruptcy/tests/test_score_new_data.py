"""
Unit tests for score_new_data.py command-line tool.
Strictly uses training subsamples; never touches test data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from score_new_data import score_csv
from src.cv import get_train_data
from src.scoring import load_final_components, score_dataset


@pytest.fixture(scope="module")
def train_sample_data():
    """Extracts a leak-free 50-row subsample from training data with dummy identifier."""
    X_train, y_train = get_train_data()
    sample = X_train.iloc[:50].copy()
    sample.insert(0, "company_id", [f"COMP_{i:04d}" for i in range(50)])
    sample["extra_metadata_col"] = "test_meta"
    return sample


def test_score_new_data_exact_match_EA(train_sample_data, tmp_path: Path):
    """Verifies that score_new_data produces identical results to direct scoring.py for E_A."""
    in_csv = tmp_path / "sample_input.csv"
    out_csv = tmp_path / "sample_output.csv"
    train_sample_data.to_csv(in_csv, index=False)

    # Run tool
    result_df = score_csv(
        input_path=in_csv,
        output_path=out_csv,
        operating_point="R90",
        model="E_A",
    )

    # Run direct scoring for reference
    components = load_final_components("outputs/models/final")
    pure_features = train_sample_data.drop(columns=["company_id", "extra_metadata_col"])
    direct_df = score_dataset(pure_features, components)

    # Assertions
    assert "company_id" in result_df.columns
    assert "extra_metadata_col" in result_df.columns
    assert (result_df["company_id"] == train_sample_data["company_id"]).all()

    # Scores must be bitwise identical
    diff_scores = np.abs(result_df["risk_percentile_score"].values - direct_df["score_E_A"].values)
    assert np.max(diff_scores) == pytest.approx(0.0, abs=1e-12)

    # Alert flags must be identical
    assert (result_df["alert"].values == direct_df["flag_R90"].values).all()

    # Platt calibrated probabilities must be identical
    diff_platt = np.abs(
        result_df["calibrated_probability_platt"].values - direct_df["calibrated_prob_platt"].values
    )
    assert np.max(diff_platt) == pytest.approx(0.0, abs=1e-12)

    # Output file exists and first line must be pure column headers
    assert out_csv.exists()
    with open(out_csv, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    assert first_line.startswith("company_id,")
    assert not first_line.startswith("#")

    # File can be read cleanly with pd.read_csv directly without comment parameter
    reloaded = pd.read_csv(out_csv)
    assert len(reloaded) == 50
    assert list(reloaded.columns) == list(result_df.columns)
    assert "risk_percentile_score" in reloaded.columns

    # README file exists in same directory with methodological notes
    readme_file = tmp_path / "sample_output.README.txt"
    assert readme_file.exists()
    readme_text = readme_file.read_text(encoding="utf-8")
    assert "方法學註記" in readme_text
    assert "相對於訓練資料參考分布的百分位" in readme_text
    assert "約 3.2% 的破產率" in readme_text
    assert "重新驗證" in readme_text


def test_score_new_data_exact_match_M3(train_sample_data, tmp_path: Path):
    """Verifies that score_new_data produces identical results to direct scoring.py for M3."""
    in_csv = tmp_path / "sample_input_m3.csv"
    out_csv = tmp_path / "sample_output_m3.csv"
    train_sample_data.to_csv(in_csv, index=False)

    result_df = score_csv(
        input_path=in_csv,
        output_path=out_csv,
        operating_point="R80",
        model="M3",
    )

    components = load_final_components("outputs/models/final")
    pure_features = train_sample_data.drop(columns=["company_id", "extra_metadata_col"])
    direct_df = score_dataset(pure_features, components)

    diff_scores = np.abs(result_df["risk_percentile_score"].values - direct_df["score_M3"].values)
    assert np.max(diff_scores) == pytest.approx(0.0, abs=1e-12)

    thresh_val = components["thresholds"]["M3"]["R80"]["threshold"]
    expected_flags = direct_df["score_M3"].values >= thresh_val
    assert (result_df["alert"].values == expected_flags).all()


def test_score_new_data_missing_column_raises_error(train_sample_data, tmp_path: Path):
    """Verifies that missing features cause an explicit, descriptive ValueError."""
    in_csv = tmp_path / "broken_input.csv"
    out_csv = tmp_path / "broken_output.csv"

    # Drop a required feature
    broken_df = train_sample_data.drop(columns=["Operating Gross Margin"])
    broken_df.to_csv(in_csv, index=False)

    with pytest.raises(ValueError, match="missing.*required feature.*Operating Gross Margin"):
        score_csv(
            input_path=in_csv,
            output_path=out_csv,
            operating_point="R90",
            model="E_A",
        )


def test_score_new_data_matches_b3ecc2c_golden(tmp_path: Path):
    """Verifies default score_new_data CLI output matches b3ecc2c golden file in names, order, and values."""
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    golden_in = fixture_dir / "golden_input_sample.csv"
    golden_ref = fixture_dir / "golden_cli_default.csv"
    assert golden_in.exists(), f"Missing fixture input: {golden_in}"
    assert golden_ref.exists(), f"Missing golden reference: {golden_ref}"

    golden_df = pd.read_csv(golden_ref)
    out_csv = tmp_path / "cli_default_out.csv"

    res_df = score_csv(
        input_path=golden_in,
        output_path=out_csv,
        operating_point="R90",
        model="E_A",
    )

    # Column names and order must match golden exactly
    assert list(res_df.columns) == list(golden_df.columns)

    # Identifiers must match exactly
    assert (res_df["company_id"] == golden_df["company_id"]).all()
    assert (res_df["extra_note"] == golden_df["extra_note"]).all()

    # Float values must be bitwise identical
    diff_scores = np.abs(res_df["risk_percentile_score"].values - golden_df["risk_percentile_score"].values)
    assert np.max(diff_scores) == pytest.approx(0.0, abs=1e-12)

    diff_thresh = np.abs(res_df["threshold"].values - golden_df["threshold"].values)
    assert np.max(diff_thresh) == pytest.approx(0.0, abs=1e-12)

    assert (res_df["alert"].values == golden_df["alert"].values).all()

    diff_cal = np.abs(res_df["calibrated_probability_platt"].values - golden_df["calibrated_probability_platt"].values)
    assert np.max(diff_cal) == pytest.approx(0.0, abs=1e-12)


def test_score_new_data_cli_flags_sort_and_filter(tmp_path: Path):
    """Verifies --sort-by-risk and --high-risk-only behavior in score_csv."""
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    golden_in = fixture_dir / "golden_input_sample.csv"

    out_sort = tmp_path / "out_sort.csv"
    res_sort = score_csv(
        input_path=golden_in,
        output_path=out_sort,
        operating_point="R90",
        model="E_A",
        sort_by_risk=True,
    )
    # Scores must be strictly non-increasing
    scores = res_sort["risk_percentile_score"].values
    assert np.all(scores[:-1] >= scores[1:])

    out_high = tmp_path / "out_high.csv"
    res_high = score_csv(
        input_path=golden_in,
        output_path=out_high,
        operating_point="R90",
        model="E_A",
        high_risk_only=True,
    )
    assert (res_high["alert"] == True).all()


def test_score_new_data_cli_no_allow_missing_raises(tmp_path: Path):
    """Verifies that allow_missing=False (default) raises exact b3ecc2c error format on NaNs."""
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    golden_in = fixture_dir / "golden_input_sample.csv"
    df = pd.read_csv(golden_in)
    # Inject a NaN
    df.loc[0, "Operating Gross Margin"] = np.nan
    in_nan = tmp_path / "in_nan.csv"
    df.to_csv(in_nan, index=False)

    out_csv = tmp_path / "out_nan.csv"
    # Default is allow_missing=False
    with pytest.raises(ValueError, match=r"Input data is missing \d+ required feature\(s\) for scoring:.*Operating Gross Margin"):
        score_csv(
            input_path=in_nan,
            output_path=out_csv,
            operating_point="R90",
            model="E_A",
        )

    # When allow_missing=True, scoring succeeds cleanly
    out_ok = tmp_path / "out_ok.csv"
    res_ok = score_csv(
        input_path=in_nan,
        output_path=out_ok,
        operating_point="R90",
        model="E_A",
        allow_missing=True,
    )
    assert len(res_ok) == 20
    assert not np.isnan(res_ok.loc[0, "risk_percentile_score"])


def test_score_new_data_column_normalization(tmp_path: Path):
    """
    Verifies robust column normalization:
    1. Leading/trailing whitespace in column names.
    2. Permuted column order.
    3. Extra unknown columns preserved as identifiers.
    4. Duplicate column names deduplicated safely.
    5. Whole column missing behavior under allow_missing=False and True.
    """
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    golden_in = fixture_dir / "golden_input_sample.csv"
    df = pd.read_csv(golden_in)

    # 1. Add whitespace to columns
    df.columns = [f"  {c}  " for c in df.columns]

    # 2. Permute column order (reverse)
    cols = list(df.columns)
    df = df[cols[::-1]]

    # 3. Add extra unknown column
    df["custom_meta_tag"] = "audit_2026"

    # 4. Add duplicate column name
    df.insert(0, "  ROA(C) before interest and depreciation before interest  ", df.iloc[:, 5].values, allow_duplicates=True)

    in_mod = tmp_path / "in_mod.csv"
    df.to_csv(in_mod, index=False)

    out_mod = tmp_path / "out_mod.csv"
    res_mod = score_csv(
        input_path=in_mod,
        output_path=out_mod,
        operating_point="R90",
        model="E_A",
        allow_missing=False,
    )
    assert len(res_mod) == 20
    assert "custom_meta_tag" in res_mod.columns
    assert (res_mod["custom_meta_tag"] == "audit_2026").all()

    # 5. Whole column missing
    df_missing_col = pd.read_csv(golden_in).drop(columns=["Operating Gross Margin"])
    in_missing_col = tmp_path / "in_missing_col.csv"
    df_missing_col.to_csv(in_missing_col, index=False)

    # Under allow_missing=False: raises exact b3ecc2c error format
    with pytest.raises(ValueError, match=r"Input data is missing \d+ required feature\(s\) for scoring:.*Operating Gross Margin"):
        score_csv(
            input_path=in_missing_col,
            output_path=tmp_path / "err.csv",
            operating_point="R90",
            model="E_A",
            allow_missing=False,
        )

    # Under allow_missing=True: succeeds with automatic median imputation
    res_imputed = score_csv(
        input_path=in_missing_col,
        output_path=tmp_path / "imputed.csv",
        operating_point="R90",
        model="E_A",
        allow_missing=True,
    )
    assert len(res_imputed) == 20
