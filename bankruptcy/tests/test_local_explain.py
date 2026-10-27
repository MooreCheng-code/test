"""
Unit Tests for Local Feature Attribution (src/local_explain.py).

Covers:
1. M4 TreeSHAP explainability in both 'final' and 'demo_oof' modes.
2. M1 Linear standardized coefficient explainability in both 'final' and 'demo_oof' modes.
3. Additivity verification: base_value + sum(attributions) == prediction_margin.
4. Input types: DataFrame, Series, dict, and rejection of multi-row DataFrames.
5. Missing value safeguards: < 20% auto-imputed, > 20% raises ValueError.
6. Attribution integrity: positive drivers increase risk, negative drivers decrease risk.
7. Disclaimers and financial dimension tags.
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

from src.cv import get_train_data
from src.local_explain import LocalExplainer, explain_case, get_local_explainer


@pytest.fixture(scope="module")
def sample_case():
    """Extracts a leak-free single record from training data."""
    X_train, y_train = get_train_data()
    row = X_train.iloc[[0]].copy()
    return row


# ==============================================================================
# 1. M4 TreeSHAP Tests (Final & Demo Modes)
# ==============================================================================
def test_explain_case_m4_final(sample_case):
    """Verifies M4 TreeSHAP feature attribution in final mode."""
    res = explain_case(sample_case, model_name="M4", mode="final", top_k=5)

    assert res["model_name"] == "M4"
    assert res["mode"] == "final"
    assert "base_value" in res
    assert "prediction_margin" in res
    assert len(res["all_attributions"]) == 72

    # Verify additivity: base_value + sum(shap) == prediction_margin
    total_shap = sum(res["all_attributions"].values())
    assert (res["base_value"] + total_shap) == pytest.approx(res["prediction_margin"], abs=1e-4)

    # Top drivers
    assert len(res["top_positive_drivers"]) <= 5
    for d in res["top_positive_drivers"]:
        assert d["attribution"] > 0
        assert "feature" in d
        assert "financial_dimension" in d
        assert d["transformed_value"] is not None

    assert len(res["top_negative_drivers"]) <= 5
    for d in res["top_negative_drivers"]:
        assert d["attribution"] < 0
        assert "feature" in d
        assert "financial_dimension" in d

    # Disclaimer check
    assert "log-odds" in res["disclaimer"]
    assert "E_A" in res["disclaimer"]
    assert "共線性" in res["disclaimer"]


def test_explain_case_m4_demo_oof(sample_case):
    """Verifies M4 TreeSHAP feature attribution in demo_oof mode."""
    res = explain_case(sample_case, model_name="M4", mode="demo_oof", top_k=5)

    assert res["model_name"] == "M4"
    assert res["mode"] == "demo_oof"
    assert len(res["all_attributions"]) == 72
    total_shap = sum(res["all_attributions"].values())
    assert (res["base_value"] + total_shap) == pytest.approx(res["prediction_margin"], abs=1e-4)


# ==============================================================================
# 2. M1 Linear Attribution Tests (Final & Demo Modes)
# ==============================================================================
def test_explain_case_m1_final(sample_case):
    """Verifies M1 standardized coefficient attribution in final mode."""
    res = explain_case(sample_case, model_name="M1", mode="final", top_k=5)

    assert res["model_name"] == "M1"
    assert res["mode"] == "final"
    assert len(res["all_attributions"]) == 72

    total_contrib = sum(res["all_attributions"].values())
    assert (res["base_value"] + total_contrib) == pytest.approx(res["prediction_margin"], abs=1e-4)

    for d in res["top_positive_drivers"]:
        assert d["attribution"] > 0
    for d in res["top_negative_drivers"]:
        assert d["attribution"] < 0


def test_explain_case_m1_demo_oof(sample_case):
    """Verifies M1 standardized coefficient attribution in demo_oof mode."""
    res = explain_case(sample_case, model_name="M1", mode="demo_oof", top_k=5)

    assert res["model_name"] == "M1"
    assert res["mode"] == "demo_oof"
    assert len(res["all_attributions"]) == 72
    total_contrib = sum(res["all_attributions"].values())
    assert (res["base_value"] + total_contrib) == pytest.approx(res["prediction_margin"], abs=1e-4)


# ==============================================================================
# 3. Input Polymorphism & Validation
# ==============================================================================
def test_explain_case_input_formats(sample_case):
    """Verifies that explain_case accepts DataFrame, Series, and dict."""
    # 1. Single-row DataFrame
    res_df = explain_case(sample_case, model_name="M4")
    # 2. Series
    series_case = sample_case.iloc[0]
    res_series = explain_case(series_case, model_name="M4")
    # 3. Dict
    dict_case = sample_case.iloc[0].to_dict()
    res_dict = explain_case(dict_case, model_name="M4")

    assert res_df["prediction_margin"] == pytest.approx(res_series["prediction_margin"], abs=1e-6)
    assert res_df["prediction_margin"] == pytest.approx(res_dict["prediction_margin"], abs=1e-6)


def test_explain_case_rejects_multi_row(sample_case):
    """Verifies that DataFrames with len != 1 raise ValueError."""
    multi_rows = pd.concat([sample_case, sample_case], ignore_index=True)
    with pytest.raises(ValueError, match="expects exactly 1 row"):
        explain_case(multi_rows, model_name="M4")


# ==============================================================================
# 4. Missing Value Safeguards
# ==============================================================================
def test_explain_case_missingness_protection(sample_case):
    """Verifies missingness <= 20% is imputed and > 20% raises ValueError."""
    # Under limit (< 20% missing, e.g. 5 features)
    case_ok = sample_case.copy()
    first_5_cols = list(case_ok.columns[:5])
    case_ok.loc[:, first_5_cols] = np.nan
    res_ok = explain_case(case_ok, model_name="M4")
    assert res_ok is not None
    assert len(res_ok["all_attributions"]) == 72

    # Over limit (> 20% missing, e.g. 25 features)
    case_bad = sample_case.copy()
    first_25_cols = list(case_bad.columns[:25])
    case_bad.loc[:, first_25_cols] = np.nan
    with pytest.raises(ValueError, match="Sample rejected: missing feature ratio"):
        explain_case(case_bad, model_name="M4")
