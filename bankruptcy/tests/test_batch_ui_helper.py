"""
tests/test_batch_ui_helper.py: Unit tests for Tab 5 UI helper module (src/batch_ui_helper.py).

Verifies:
1. Dynamic Top-K percentage label formatting (no hardcoded decimals/percentages).
2. Dynamic operating point option text formatting (including R100 '接近全抓').
3. Filter and pessimistic sorting behavior.
4. Summary metrics calculations under all edge conditions:
   - Normal batch with labels (N >= 25)
   - Small batch with labels (N < 25)
   - Unlabelled batch
   - Zero positives batch
   - Empty batch
5. BOM-encoded utf-8-sig CSV export byte serialization.
6. Static isolation check: helper file contains zero forbidden test-set references (get_test_data, test_idx, e6_).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.batch_ui_helper import (
    compute_batch_summary_metrics,
    filter_and_sort_batch,
    format_operating_point_option,
    format_top_k_header,
    format_top_k_pct_text,
    prepare_export_csv_bytes,
)

HELPER_PATH = Path(__file__).resolve().parent.parent / "src" / "batch_ui_helper.py"


def test_dynamic_top_k_formatting():
    """Verifies that Top-K text formatting adapts dynamically to arbitrary k_percent."""
    # Default from config
    expected_default_pct = f"{int(round(config.TOP_K_PERCENT * 100))}%"
    assert format_top_k_pct_text() == expected_default_pct
    assert format_top_k_header() == f"前 {expected_default_pct} (Top {expected_default_pct})"

    # Custom k percentages
    assert format_top_k_pct_text(0.05) == "5%"
    assert format_top_k_header(0.05) == "前 5% (Top 5%)"

    assert format_top_k_pct_text(0.15) == "15%"
    assert format_top_k_header(0.15) == "前 15% (Top 15%)"

    assert format_top_k_pct_text(0.20) == "20%"
    assert format_top_k_header(0.20) == "前 20% (Top 20%)"


def test_format_operating_point_option():
    """Verifies dynamic operating point labels with target recall and R100 label."""
    meta = {
        "R80": {"target_recall": 0.80},
        "R90": {"target_recall": 0.90},
        "R95": {"target_recall": 0.95},
        "R100": {"target_recall": 1.00},
    }
    lbl_r90 = format_operating_point_option("R90", meta)
    assert "R90" in lbl_r90
    assert "90%" in lbl_r90
    assert "接近全抓" not in lbl_r90

    lbl_r100 = format_operating_point_option("R100", meta)
    assert "R100" in lbl_r100
    assert "100%" in lbl_r100
    assert "接近全抓" in lbl_r100

    # Fallback when meta missing
    assert format_operating_point_option("CUSTOM", None) == "CUSTOM"


def test_filter_and_sort_batch():
    """Verifies high-risk filtering and risk score sorting."""
    df = pd.DataFrame({
        "company_id": ["C1", "C2", "C3", "C4"],
        "risk_score": [0.2, 0.9, 0.5, 0.7],
        "alert": [False, True, False, True],
    })

    # Sort only
    df_sorted = filter_and_sort_batch(df, high_risk_only=False, sort_by_risk=True)
    assert df_sorted["company_id"].tolist() == ["C2", "C4", "C3", "C1"]

    # Filter and sort
    df_filtered = filter_and_sort_batch(df, high_risk_only=True, sort_by_risk=True)
    assert df_filtered["company_id"].tolist() == ["C2", "C4"]

    # Empty handling
    assert len(filter_and_sort_batch(pd.DataFrame())) == 0
    assert len(filter_and_sort_batch(None)) == 0


def test_compute_batch_summary_metrics_normal_batch():
    """Verifies summary metrics for a valid batch (N >= 25) with labels."""
    # 30 records: 5 positives, 6 alerts
    rows = []
    for i in range(30):
        rows.append({
            "company_id": f"C_{i:02d}",
            "risk_score": float(30 - i) / 30.0,
            "status": "ok",
            "alert": bool(i < 6),
            "in_batch_top_10_pct": bool(i < 3),  # 10% of 30 = 3
            "Bankrupt?": 1 if i < 5 else 0,
        })
    df = pd.DataFrame(rows)

    metrics = compute_batch_summary_metrics(df, target_col="Bankrupt?")
    assert metrics["total_records"] == 30
    assert metrics["valid_records"] == 30
    assert metrics["rejected_records"] == 0
    assert metrics["alert_count"] == 6
    assert metrics["alert_proportion"] == pytest.approx(6 / 30.0)
    assert metrics["target_available"] is True
    assert metrics["total_positives"] == 5
    assert metrics["actual_hits_alert"] == 5
    assert metrics["actual_hits_top_k"] == 3
    assert metrics["recall_at_top_k"] == pytest.approx(3 / 5.0)
    assert metrics["top_k_enabled"] is True


def test_compute_batch_summary_metrics_small_batch():
    """Verifies that N_valid < 25 suppresses Top-K flag and recall."""
    rows = []
    for i in range(15):  # 15 < 25
        rows.append({
            "company_id": f"C_{i:02d}",
            "risk_score": float(15 - i) / 15.0,
            "status": "ok",
            "alert": bool(i < 2),
            "in_batch_top_10_pct": None,
            "Bankrupt?": 1 if i < 2 else 0,
        })
    df = pd.DataFrame(rows)

    metrics = compute_batch_summary_metrics(df, target_col="Bankrupt?")
    assert metrics["valid_records"] == 15
    assert metrics["top_k_enabled"] is False
    assert metrics["recall_at_top_k"] is None
    assert "未達前" in metrics["note"]


def test_compute_batch_summary_metrics_unlabelled_and_zero_positives():
    """Verifies graceful handling when target column is missing or has zero positives."""
    df_no_label = pd.DataFrame({
        "risk_score": [0.5, 0.8, 0.2],
        "status": ["ok", "ok", "ok"],
        "alert": [False, True, False],
    })
    m_no_label = compute_batch_summary_metrics(df_no_label, target_col="Bankrupt?")
    assert m_no_label["target_available"] is False
    assert m_no_label["recall_at_top_k"] is None
    assert "未含標籤" in m_no_label["note"]

    # Zero positives in large batch
    df_zero_pos = pd.DataFrame({
        "risk_score": [0.1] * 30,
        "status": ["ok"] * 30,
        "alert": [False] * 30,
        "Bankrupt?": [0] * 30,
    })
    m_zero_pos = compute_batch_summary_metrics(df_zero_pos, target_col="Bankrupt?")
    assert m_zero_pos["target_available"] is True
    assert m_zero_pos["total_positives"] == 0
    assert m_zero_pos["recall_at_top_k"] is None
    assert "無破產正樣本" in m_zero_pos["note"]

    # Empty batch
    m_empty = compute_batch_summary_metrics(pd.DataFrame())
    assert m_empty["total_records"] == 0
    assert m_empty["recall_at_top_k"] is None


def test_prepare_export_csv_bytes():
    """Verifies that CSV export produces valid utf-8-sig byte stream with BOM header."""
    df = pd.DataFrame({"company": ["公司A", "公司B"], "risk": [0.95, 0.12]})
    csv_bytes = prepare_export_csv_bytes(df)

    assert isinstance(csv_bytes, bytes)
    # Starts with UTF-8 BOM
    assert csv_bytes.startswith(b"\xef\xbb\xbf")

    # Roundtrip read with pandas
    df_roundtrip = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig")
    assert list(df_roundtrip.columns) == ["company", "risk"]
    assert df_roundtrip["company"].tolist() == ["公司A", "公司B"]


def test_batch_ui_helper_isolation_and_no_test_set_references():
    """Verifies batch_ui_helper.py strictly satisfies isolation from test set."""
    assert HELPER_PATH.exists()
    content = HELPER_PATH.read_text(encoding="utf-8")

    forbidden_tokens = ["get_test_data", "test_idx", "e6_", "outputs/e6"]
    for token in forbidden_tokens:
        assert token not in content, f"Forbidden token '{token}' found in {HELPER_PATH}"
