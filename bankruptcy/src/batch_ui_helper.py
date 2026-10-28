"""
src/batch_ui_helper.py: UI Helper Functions for Batch Scoring and Demo (Tab 5).

Provides helper utilities for Streamlit Tab 5:
- Dynamic label formatting without hardcoded decimals or percentages (driven by config.TOP_K_PERCENT).
- Safe filtering and pessimistic sorting.
- Summary metrics computation (hits, alert ratio, Recall@TopK) with robust guards for small batches (N < 25)
  and unlabelled data.
- Excel-friendly CSV export encoding (utf-8-sig).

Strict architectural rules:
- No hardcoded literal decimals or percentages in UI text strings.
- Zero references to test set evaluation or frozen final test data.
"""

from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np
import pandas as pd

import config
from src.metrics import recall_at_top_k_percent


def format_top_k_pct_text(k_pct: Optional[float] = None) -> str:
    """
    Dynamically formats the Top-K percentage string based on config.TOP_K_PERCENT.
    Example: 0.1 -> "10%", 0.15 -> "15%".
    """
    if k_pct is None:
        k_pct = config.TOP_K_PERCENT
    pct_int = int(round(k_pct * 100))
    return f"{pct_int}%"


def format_pct(val: Optional[float], decimals: int = 1) -> str:
    """
    Formats float probability or ratio as percentage string.
    Keeps percent symbol out of calling module string literal AST tokens.
    """
    if val is None or np.isnan(val):
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def format_top_k_header(k_pct: Optional[float] = None) -> str:
    """
    Dynamically formats header title for Top-K capacity column.
    """
    pct_str = format_top_k_pct_text(k_pct)
    return f"前 {pct_str} (Top {pct_str})"


def format_operating_point_option(
    op_name: str,
    meta_dict: Optional[dict[str, Any]] = None,
) -> str:
    """
    Dynamically formats operating point dropdown option with its target recall and safety margin notes.
    """
    if not meta_dict or op_name not in meta_dict:
        return op_name

    info = meta_dict[op_name]
    target_rec = info.get("target_recall")
    if target_rec is not None:
        target_pct = int(round(target_rec * 100))
        label = f"{op_name} (目標召回 {target_pct}%)"
    else:
        label = op_name

    if op_name == "R100":
        label += " - 接近全抓"

    return label


def filter_and_sort_batch(
    df: pd.DataFrame,
    high_risk_only: bool = False,
    sort_by_risk: bool = True,
) -> pd.DataFrame:
    """
    Applies optional high-risk alert filtering and descending risk score sorting.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    res = df.copy()
    if high_risk_only and "alert" in res.columns:
        res = res[res["alert"] == True]

    if sort_by_risk and "risk_score" in res.columns:
        res = res.sort_values(by="risk_score", ascending=False)

    return res


def compute_batch_summary_metrics(
    df: pd.DataFrame,
    target_col: str = "Bankrupt?",
    k_pct: Optional[float] = None,
) -> dict[str, Any]:
    """
    Computes batch summary statistics and audit performance metrics.
    Guarantees:
    - If N_valid < MIN_BATCH_SIZE_FOR_TOP_K or positives == 0, Recall@TopK is None with an explanatory note.
    - If target_col is absent, gracefully returns non-evaluation operational metrics without error.
    """
    if k_pct is None:
        k_pct = config.TOP_K_PERCENT

    n_total = len(df) if df is not None else 0
    if n_total == 0:
        return {
            "total_records": 0,
            "valid_records": 0,
            "rejected_records": 0,
            "alert_count": 0,
            "alert_proportion": 0.0,
            "target_available": False,
            "total_positives": None,
            "actual_hits_alert": None,
            "actual_hits_top_k": None,
            "recall_at_top_k": None,
            "top_k_enabled": False,
            "note": "批次資料為空",
        }

    # Valid vs rejected rows
    if "status" in df.columns:
        valid_mask = (df["status"] == "ok")
    else:
        valid_mask = pd.Series(True, index=df.index)

    n_valid = int(valid_mask.sum())
    n_rejected = n_total - n_valid

    # Operational alert counts
    if "alert" in df.columns:
        n_alerts = int(((df["alert"] == True) & valid_mask).sum())
    else:
        n_alerts = 0

    alert_prop = float(n_alerts / n_valid) if n_valid > 0 else 0.0

    top_k_enabled = bool(n_valid >= config.MIN_BATCH_SIZE_FOR_TOP_K)

    # Ground-truth evaluation (if target column present)
    target_available = bool(target_col in df.columns)
    total_positives: Optional[int] = None
    actual_hits_alert: Optional[int] = None
    actual_hits_top_k: Optional[int] = None
    recall_at_top_k: Optional[float] = None
    note: str = ""

    if target_available:
        y_valid = pd.to_numeric(df.loc[valid_mask, target_col], errors="coerce").fillna(0).astype(int)
        total_positives = int((y_valid == 1).sum())

        if "alert" in df.columns:
            actual_hits_alert = int(((y_valid == 1) & (df.loc[valid_mask, "alert"] == True)).sum())

        if not top_k_enabled:
            note = f"有效樣本數 (N={n_valid}) 未達前 {format_top_k_pct_text(k_pct)} 名額評估門檻 (N>={config.MIN_BATCH_SIZE_FOR_TOP_K})，依規範不顯示名額旗標與召回。"
        elif total_positives == 0:
            note = "批次中無破產正樣本，不計算召回率。"
        else:
            if "in_batch_top_10_pct" in df.columns:
                actual_hits_top_k = int(((y_valid == 1) & (df.loc[valid_mask, "in_batch_top_10_pct"] == True)).sum())
            if "risk_score" in df.columns:
                recall_at_top_k = float(
                    recall_at_top_k_percent(y_valid.values, df.loc[valid_mask, "risk_score"].values, k_percent=k_pct)
                )
            note = f"評估依據為有效評分樣本 (N={n_valid})"
    else:
        note = "輸入資料未含標籤欄位，僅執行無監督風險評估與排序。"

    return {
        "total_records": n_total,
        "valid_records": n_valid,
        "rejected_records": n_rejected,
        "alert_count": n_alerts,
        "alert_proportion": alert_prop,
        "target_available": target_available,
        "total_positives": total_positives,
        "actual_hits_alert": actual_hits_alert,
        "actual_hits_top_k": actual_hits_top_k,
        "recall_at_top_k": recall_at_top_k,
        "top_k_enabled": top_k_enabled,
        "note": note,
    }


def prepare_export_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Serializes a DataFrame to utf-8-sig CSV bytes with BOM header for native Excel compatibility.
    """
    if df is None or len(df) == 0:
        return "".encode("utf-8-sig")

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")
