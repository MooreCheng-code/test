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

from scipy.special import logit
from src.cv import get_train_data
from src.explain import get_underlying_classifier, transform_through_preprocessing
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
    """Verifies M4 TreeSHAP feature attribution in final mode against independent clf.predict(output_margin=True) and logit."""
    res = explain_case(sample_case, model_name="M4", mode="final", top_k=5)

    assert res["model_name"] == "M4"
    assert res["mode"] == "final"
    assert "base_value" in res
    assert "prediction_margin" in res
    assert "risk_score" in res
    assert "reference_percentile" in res
    assert "imputed_features" in res
    assert len(res["all_attributions"]) == 72

    # Independent calculations directly from pipeline and classifier
    explainer = get_local_explainer()
    pipe_m4 = explainer.get_pipeline("M4", mode="final")
    clf = get_underlying_classifier(pipe_m4)
    X_prep, _ = transform_through_preprocessing(pipe_m4, sample_case)

    # 1. Independent margin directly from underlying tree classifier
    independent_margin = float(clf.predict(X_prep, output_margin=True)[0])

    # 2. Independent logit directly from pipeline.predict_proba
    p_val = float(pipe_m4.predict_proba(sample_case)[:, 1][0])
    independent_logit = float(logit(p_val))

    # TreeSHAP sum: base_value + sum(phi)
    total_shap = res["base_value"] + sum(res["all_attributions"].values())

    diff_shap_vs_margin = abs(total_shap - independent_margin)
    diff_shap_vs_logit = abs(total_shap - independent_logit)
    diff_margin_vs_logit = abs(independent_margin - independent_logit)

    assert diff_shap_vs_margin == pytest.approx(0.0, abs=1e-5)
    assert diff_shap_vs_logit == pytest.approx(0.0, abs=1e-5)
    assert diff_margin_vs_logit == pytest.approx(0.0, abs=1e-5)

    # Top drivers: no financial_dimension (unapproved taxonomy removed)
    assert len(res["top_positive_drivers"]) <= 5
    for d in res["top_positive_drivers"]:
        assert d["attribution"] > 0
        assert "feature" in d
        assert "financial_dimension" not in d
        assert d["transformed_value"] is not None

    assert len(res["top_negative_drivers"]) <= 5
    for d in res["top_negative_drivers"]:
        assert d["attribution"] < 0
        assert "feature" in d
        assert "financial_dimension" not in d

    # Disclaimer check
    assert "log-odds" in res["disclaimer"]
    assert "E_A" in res["disclaimer"]


def test_explain_case_m4_demo_oof(sample_case):
    """Verifies M4 TreeSHAP feature attribution in demo_oof mode against independent clf margin and logit."""
    res = explain_case(sample_case, model_name="M4", mode="demo_oof", top_k=5)

    assert res["model_name"] == "M4"
    assert res["mode"] == "demo_oof"
    assert "base_value" in res
    assert "prediction_margin" in res
    assert "risk_score" in res
    assert "reference_percentile" in res
    assert "imputed_features" in res
    assert len(res["all_attributions"]) == 72

    explainer = get_local_explainer()
    pipe_m4 = explainer.get_pipeline("M4", mode="demo_oof")
    clf = get_underlying_classifier(pipe_m4)
    X_prep, _ = transform_through_preprocessing(pipe_m4, sample_case)

    independent_margin = float(clf.predict(X_prep, output_margin=True)[0])
    p_val = float(pipe_m4.predict_proba(sample_case)[:, 1][0])
    independent_logit = float(logit(p_val))

    total_shap = res["base_value"] + sum(res["all_attributions"].values())
    assert abs(total_shap - independent_margin) == pytest.approx(0.0, abs=1e-5)
    assert abs(total_shap - independent_logit) == pytest.approx(0.0, abs=1e-5)
def test_explain_case_m1_final(sample_case):
    """Verifies M1 standardized coefficient attribution in final mode against decision_function."""
    res = explain_case(sample_case, model_name="M1", mode="final", top_k=5)

    assert res["model_name"] == "M1"
    assert res["mode"] == "final"
    assert len(res["all_attributions"]) == 72

    # Independent decision_function calculation
    explainer = get_local_explainer()
    pipe_m1 = explainer.get_pipeline("M1", mode="final")
    clf_m1 = get_underlying_classifier(pipe_m1)
    X_prep, feat_names = transform_through_preprocessing(pipe_m1, sample_case)
    independent_dec = float(clf_m1.decision_function(pd.DataFrame(X_prep, columns=feat_names))[0])

    total_contrib = res["base_value"] + sum(res["all_attributions"].values())
    assert abs(total_contrib - independent_dec) == pytest.approx(0.0, abs=1e-5)
    assert abs(res["prediction_margin"] - independent_dec) == pytest.approx(0.0, abs=1e-5)

    for d in res["top_positive_drivers"]:
        assert d["attribution"] > 0
        assert "financial_dimension" not in d
    for d in res["top_negative_drivers"]:
        assert d["attribution"] < 0
        assert "financial_dimension" not in d


def test_explain_case_m1_demo_oof(sample_case):
    """Verifies M1 standardized coefficient attribution in demo_oof mode against decision_function."""
    res = explain_case(sample_case, model_name="M1", mode="demo_oof", top_k=5)

    assert res["model_name"] == "M1"
    assert res["mode"] == "demo_oof"
    assert len(res["all_attributions"]) == 72

    explainer = get_local_explainer()
    pipe_demo = explainer.get_pipeline("M1", mode="demo_oof")
    clf_demo = get_underlying_classifier(pipe_demo)
    X_prep, feat_names = transform_through_preprocessing(pipe_demo, sample_case)
    independent_dec = float(clf_demo.decision_function(pd.DataFrame(X_prep, columns=feat_names))[0])

    total_contrib = res["base_value"] + sum(res["all_attributions"].values())
    assert abs(total_contrib - independent_dec) == pytest.approx(0.0, abs=1e-5)


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
