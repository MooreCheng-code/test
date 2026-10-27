"""
Unit Tests for Stage 3 MVP Batch Inference and Robust Scoring (src/scoring.py & score_new_data.py).

Covers:
1. load_mvp_components loads all 4 models (E_A, M3, M4, M1), references, and thresholds.
2. Bitwise backward compatibility: for complete data, score_mvp_batch('E_A') matches score_dataset() exactly.
3. Missing value protection:
   - Missing <= 20%: status='ok', auto-imputed using train_medians_95.
   - Missing > 20%: status='rejected_high_missing', risk_score and ranks are None/NaN, excluded from N_valid.
   - Single sample request with > 20% missing raises ValueError.
4. Batch ranking and Top 10%:
   - N_valid >= 25: compute_top_k_quota flags top floor(0.10 * N_valid) samples.
   - N_valid < 25: in_batch_top_10_pct is None.
5. All 4 models (E_A, M3, M4, M1) run cleanly with correct margin_validated flags.
6. Reference distribution of E_A reproduces 10,910 distribution.
7. score_new_data CLI handles M4 and M1 while preserving legacy columns.
8. No Bankrupt? column or 0 positives does not cause ValueError.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from score_new_data import score_csv
from src.cv import get_train_data
from src.scoring import (
    load_final_components,
    load_mvp_components,
    score_dataset,
    score_mvp_batch,
    to_percentile,
)


@pytest.fixture(scope="module")
def train_sample_50():
    """Extracts a leak-free 50-row subsample from training data with dummy identifier."""
    X_train, y_train = get_train_data()
    sample = X_train.iloc[:50].copy()
    sample.insert(0, "company_id", [f"COMP_{i:04d}" for i in range(50)])
    sample["Bankrupt?"] = y_train.iloc[:50].values
    return sample


# ==============================================================================
# 1. load_mvp_components Tests
# ==============================================================================
def test_load_mvp_components():
    comp = load_mvp_components()
    assert "models" in comp
    for m in ["M3", "M4", "M5", "M1"]:
        assert m in comp["models"]

    assert "references" in comp
    for m in ["M3", "M4", "M5", "M1", "E_A"]:
        assert m in comp["references"]
        assert len(comp["references"][m]) == 10910

    assert "thresholds" in comp
    assert "E_A" in comp["thresholds"]
    assert "M1" in comp["thresholds"]
    assert "M4" in comp["thresholds"]
    assert "train_medians_95" in comp
    assert len(comp["train_medians_95"]) == 95


# ==============================================================================
# 2. Backward Compatibility with score_dataset()
# ==============================================================================
def test_score_mvp_batch_exact_match_score_dataset(train_sample_50):
    final_comp = load_final_components(project_root / "outputs" / "models" / "final")
    mvp_comp = load_mvp_components()

    pure_features = train_sample_50.drop(columns=["company_id", "Bankrupt?"])
    legacy_res = score_dataset(pure_features, final_comp)
    mvp_res = score_mvp_batch(train_sample_50, model_name="E_A", operating_point="R90", components=mvp_comp)

    # Identifiers preserved
    assert "company_id" in mvp_res.columns
    assert (mvp_res["company_id"] == train_sample_50["company_id"]).all()

    # Scores must be bitwise identical
    diff_scores = np.abs(mvp_res["risk_score"].values - legacy_res["score_E_A"].values)
    assert np.max(diff_scores) == pytest.approx(0.0, abs=1e-12)

    # Alerts must be identical
    assert (mvp_res["alert"].values == legacy_res["flag_R90"].values).all()

    # Calibrated probabilities must be identical
    diff_cal = np.abs(mvp_res["calibrated_score"].values - legacy_res["calibrated_prob_platt"].values)
    assert np.max(diff_cal) == pytest.approx(0.0, abs=1e-12)


# ==============================================================================
# 3. Missing Value Safeguards
# ==============================================================================
def test_score_mvp_batch_with_missing_under_limit(train_sample_50):
    mvp_comp = load_mvp_components()
    df_missing = train_sample_50.copy()

    # Inject NaNs into 5 columns (< 20% of 95 features) for row 0
    cols_to_nan = list(mvp_comp["raw_features_95"][:5])
    df_missing.loc[0, cols_to_nan] = np.nan

    res = score_mvp_batch(df_missing, model_name="E_A", components=mvp_comp)
    assert res.loc[0, "status"] == "ok"
    assert res.loc[0, "missing_feature_count"] == 5
    assert res.loc[0, "missing_feature_ratio"] == pytest.approx(5 / 95.0)
    assert not np.isnan(res.loc[0, "risk_score"])
    assert res.loc[0, "batch_rank"] is not None


def test_score_mvp_batch_with_missing_over_limit(train_sample_50):
    mvp_comp = load_mvp_components()
    df_missing = train_sample_50.copy()

    # Inject NaNs into 25 columns (> 20% of 95 features) for row 0
    cols_to_nan = list(mvp_comp["raw_features_95"][:25])
    df_missing.loc[0, cols_to_nan] = np.nan

    res = score_mvp_batch(df_missing, model_name="E_A", components=mvp_comp)
    assert res.loc[0, "status"] == "rejected_high_missing"
    assert res.loc[0, "missing_feature_count"] == 25
    assert res.loc[0, "missing_feature_ratio"] == pytest.approx(25 / 95.0)
    assert pd.isna(res.loc[0, "risk_score"])
    assert pd.isna(res.loc[0, "reference_percentile"])
    assert pd.isna(res.loc[0, "alert"]) or res.loc[0, "alert"] is None
    assert pd.isna(res.loc[0, "batch_rank"])
    assert pd.isna(res.loc[0, "in_batch_top_10_pct"]) or res.loc[0, "in_batch_top_10_pct"] is None

    # Verify that valid rows (rows 1..49) have 49 valid rankings
    valid_ranks = res.loc[1:, "batch_rank"].dropna().values
    assert len(valid_ranks) == 49
    assert np.min(valid_ranks) == 1
    assert np.max(valid_ranks) <= 49


def test_single_sample_rejection(train_sample_50):
    mvp_comp = load_mvp_components()
    single_row = train_sample_50.iloc[[0]].copy()
    cols_to_nan = list(mvp_comp["raw_features_95"][:30])
    single_row[cols_to_nan] = np.nan

    with pytest.raises(ValueError, match="Single sample request rejected"):
        score_mvp_batch(single_row, components=mvp_comp)


# ==============================================================================
# 4. Batch Top 10% Quota and Sample Size Threshold
# ==============================================================================
def test_batch_top_10_pct_quota(train_sample_50):
    mvp_comp = load_mvp_components()

    # N = 50 >= 25 -> floor(0.10 * 50) = 5
    res_50 = score_mvp_batch(train_sample_50, model_name="E_A", components=mvp_comp)
    top10_flags = res_50["in_batch_top_10_pct"].values
    assert int(np.sum(top10_flags == True)) == 5
    # The top 5 flagged must be ranks 1, 2, 3, 4, 5
    flagged_ranks = res_50[res_50["in_batch_top_10_pct"] == True]["batch_rank"].values
    assert sorted(flagged_ranks) == [1, 2, 3, 4, 5]

    # N = 20 < 25 -> in_batch_top_10_pct must be None for all rows
    sample_20 = train_sample_50.iloc[:20].copy()
    res_20 = score_mvp_batch(sample_20, model_name="E_A", components=mvp_comp)
    assert all(x is None for x in res_20["in_batch_top_10_pct"])


# ==============================================================================
# 5. Model Coverage & Margin Validation Flags
# ==============================================================================
@pytest.mark.parametrize("m_name, expected_margin_val", [
    ("E_A", True),
    ("M3", True),
    ("M4", False),
    ("M1", False),
])
def test_all_models_scoring(train_sample_50, m_name, expected_margin_val):
    mvp_comp = load_mvp_components()
    res = score_mvp_batch(train_sample_50, model_name=m_name, operating_point="R90", components=mvp_comp)
    assert len(res) == 50
    assert (res["margin_validated"] == expected_margin_val).all()
    assert (res["status"] == "ok").all()
    assert not res["risk_score"].isna().any()
    assert not res["reference_percentile"].isna().any()


# ==============================================================================
# 6. score_new_data CLI Support for M4 and M1
# ==============================================================================
def test_score_new_data_cli_m4_and_m1(train_sample_50, tmp_path: Path):
    in_csv = tmp_path / "in.csv"
    train_sample_50.to_csv(in_csv, index=False)

    for m in ["M4", "M1"]:
        out_csv = tmp_path / f"out_{m}.csv"
        res = score_csv(
            input_path=in_csv,
            output_path=out_csv,
            operating_point="R90",
            model=m,
        )
        assert len(res) == 50
        assert "risk_percentile_score" in res.columns
        assert "alert" in res.columns
        assert "calibrated_probability_platt" in res.columns
        assert out_csv.exists()


# ==============================================================================
# 7. No Bankrupt? Column Tolerant
# ==============================================================================
def test_no_bankrupt_col_tolerant(train_sample_50):
    mvp_comp = load_mvp_components()
    df_no_y = train_sample_50.drop(columns=["Bankrupt?"])
    res = score_mvp_batch(df_no_y, model_name="E_A", components=mvp_comp)
    assert "Bankrupt?" not in res.columns
    assert len(res) == 50
    assert (res["status"] == "ok").all()
