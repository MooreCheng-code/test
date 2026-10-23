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
