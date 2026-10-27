"""
FastAPI Bankruptcy Prediction & Local Attribution Service (api/main.py).

Provides production-ready REST API endpoints:
- GET /health: Health check, version, loaded model components.
- GET /model-info: Operating points, thresholds, margins, alert ratios, alerts per hit, M5 limitations.
- POST /score/json: JSON batch / single-case scoring with row-level status and NaN -> null handling.
- POST /score/batch-csv: CSV batch upload scoring with in-memory streaming and 10MB upload limit.
- POST /explain/case: Single-case feature attribution (mode='final' strictly) with M4/E_A and multicollinearity disclaimers.

Guarantees:
- Binds strictly to localhost (127.0.0.1).
- In-memory processing: zero disk writing for uploaded payloads.
- JSON responses strictly sanitize NaN / Infinity to null.
- Single-case exceeding 20% missingness returns HTTP 422 with the list of missing features.
- All scoring responses include standard business and legal disclaimers.
"""

from __future__ import annotations

import io
import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from src.local_explain import explain_case, get_local_explainer
from src.scoring import load_mvp_components, score_mvp_batch

# Maximum upload limit: 10 MB
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024

DISCLAIMER_TEXT = (
    "本系統模型預測、評分與歸因結果僅供內部信貸審查與風險稽核流程參考，"
    "不構成任何授信准駁、投資決策或法律效力保證。"
)

EXPLAIN_DISCLAIMER_TEXT = (
    "本局部歸因基於 M4 (XGBoost TreeSHAP) 或 M1 (正規化回歸係數) 計算，單位為對數勝算比（log-odds）。"
    "當前主要排序與評分模型為 E_A (三模型等權集成)，"
    "單一模型歸因僅供業務審查時掌握風險驅動方向之參考；"
    "由於財務變數間存在高度共線性，相近指標間的權重分配具有替代性，"
    "不代表因果關係或單一變數變動對破產機率的邊際效應。"
)

app = FastAPI(
    title="企業破產早期預警評分與解釋服務 API",
    description="提供企業破產風險評分、分位數排序、Top 10% 集中審查標記與 TreeSHAP 局部特徵歸因 REST 服務。",
    version="1.0.0",
)

# Global cached components
_COMPONENTS = None


def get_components():
    global _COMPONENTS
    if _COMPONENTS is None:
        _COMPONENTS = load_mvp_components()
    return _COMPONENTS


def sanitize_nan(val: Any) -> Any:
    """Recursively replaces NaN, Inf, and -Inf with None (null in JSON)."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    elif isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    elif isinstance(val, (int, np.integer)):
        return int(val)
    elif isinstance(val, dict):
        return {k: sanitize_nan(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple, np.ndarray)):
        return [sanitize_nan(v) for v in val]
    elif pd.isna(val):
        return None
    return val


# ==============================================================================
# Pydantic Schemas
# ==============================================================================
class ScoreJsonRequest(BaseModel):
    model_name: str = Field(default="E_A", description="Model name: 'E_A', 'M3', 'M4', 'M1'")
    operating_point: str = Field(default="R90", description="Operating point: 'R80', 'R90', 'R95', 'R100'")
    allow_missing: bool = Field(default=False, description="Allow missing features up to 20% (default: False)")
    records: list[dict[str, Any]] = Field(..., description="List of company feature records (up to 95 features)")


class ExplainCaseRequest(BaseModel):
    record: dict[str, Any] = Field(..., description="Single company feature record")
    model_name: str = Field(default="M4", description="Model for explanation: 'M4' (TreeSHAP) or 'M1' (Linear)")
    top_k: int = Field(default=5, description="Number of top positive and negative drivers to return")


# ==============================================================================
# 1. Health & Model Information Endpoints
# ==============================================================================
@app.get("/health", summary="服務健康狀態檢核")
def health():
    comp = get_components()
    return {
        "status": "ok",
        "service": "bankruptcy_scoring_api",
        "version": "1.0.0",
        "models_available": list(comp["models"].keys()) + ["E_A"],
        "thresholds_available": list(comp["thresholds"].keys()),
    }


@app.get("/model-info", summary="查詢模型與工作點門檻資訊")
def model_info():
    comp = get_components()
    thresholds_meta = comp.get("thresholds", {})

    m5_limitation = (
        "CV 評估的 M5 啟用 subsample_freq=1，最終凍結 M5 為 subsample_freq=0；"
        "CV 指標僅供選模參考，最終效能以 E6 測試集結果為準；"
        "E5 的 M5 全域 SHAP 亦使用未含該參數的重建設定。"
    )

    return sanitize_nan({
        "models": ["E_A", "M3", "M4", "M1"],
        "operating_points": ["R80", "R90", "R95", "R100"],
        "m5_limitation": m5_limitation,
        "thresholds": thresholds_meta,
        "disclaimer": DISCLAIMER_TEXT,
    })


# ==============================================================================
# 2. JSON Scoring Endpoint
# ==============================================================================
@app.post("/score/json", summary="JSON 格式單筆或批次財務資料評分")
def score_json(payload: ScoreJsonRequest):
    comp = get_components()
    raw_features_95 = comp["raw_features_95"]

    if not payload.records:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The 'records' list cannot be empty.",
        )

    df_input = pd.DataFrame(payload.records)

    # For single sample: strictly check missingness > 20% and return 422 with missing columns
    if len(df_input) == 1:
        row_dict = payload.records[0]
        missing_cols = [c for c in raw_features_95 if c not in row_dict or pd.isna(row_dict[c])]
        missing_count = len(missing_cols)
        missing_ratio = missing_count / 95.0

        if not payload.allow_missing and missing_count > 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": f"Input data is missing {missing_count} required feature(s) for scoring (allow_missing=False).",
                    "missing_feature_count": missing_count,
                    "missing_features": missing_cols,
                },
            )

        if missing_ratio > 0.20:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": f"Single sample rejected: missing feature ratio {missing_ratio:.1%} ({missing_count}/95) exceeds allowed limit (20%).",
                    "missing_feature_count": missing_count,
                    "missing_feature_ratio": missing_ratio,
                    "missing_features": missing_cols,
                },
            )

    try:
        scored_df = score_mvp_batch(
            df_input,
            model_name=payload.model_name,
            operating_point=payload.operating_point,
            allow_missing=payload.allow_missing,
            components=comp,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    # Convert DataFrame to records, replacing NaN with null
    records_out = scored_df.to_dict(orient="records")
    sanitized_records = [sanitize_nan(r) for r in records_out]

    n_total = len(df_input)
    n_rejected = sum(1 for r in sanitized_records if r.get("status") == "rejected_high_missing")
    n_valid = n_total - n_rejected

    response_data = {
        "model_name": payload.model_name,
        "operating_point": payload.operating_point,
        "total_records": n_total,
        "valid_records": n_valid,
        "rejected_records": n_rejected,
        "top_10_pct_quota_enabled": bool(n_valid >= 25),
        "results": sanitized_records,
        "disclaimer": DISCLAIMER_TEXT,
    }

    return JSONResponse(content=response_data)


# ==============================================================================
# 3. CSV Batch Upload Scoring Endpoint
# ==============================================================================
@app.post("/score/batch-csv", summary="CSV 檔案上傳批次評分（記憶體串流不落地）")
async def score_batch_csv(
    file: UploadFile = File(..., description="Input CSV file containing company records"),
    model_name: str = Form(default="E_A"),
    operating_point: str = Form(default="R90"),
    allow_missing: bool = Form(default=False),
    output_format: str = Query(default="csv", regex="^(csv|json)$"),
):
    # Enforce 10MB memory upload limit
    content_bytes = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if len(content_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB.",
        )

    # Read CSV in-memory via StringIO
    try:
        csv_text = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            csv_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            csv_text = content_bytes.decode("cp950", errors="replace")

    try:
        df_input = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse uploaded CSV: {e}",
        )

    comp = get_components()
    try:
        scored_df = score_mvp_batch(
            df_input,
            model_name=model_name,
            operating_point=operating_point,
            allow_missing=allow_missing,
            components=comp,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    if output_format == "json":
        records_out = scored_df.to_dict(orient="records")
        sanitized_records = [sanitize_nan(r) for r in records_out]
        return JSONResponse(
            content={
                "model_name": model_name,
                "operating_point": operating_point,
                "total_records": len(df_input),
                "results": sanitized_records,
                "disclaimer": DISCLAIMER_TEXT,
            }
        )
    else:
        # Export as Excel-friendly utf-8-sig CSV
        out_buf = io.StringIO()
        scored_df.to_csv(out_buf, index=False)
        csv_output_bytes = out_buf.getvalue().encode("utf-8-sig")

        return Response(
            content=csv_output_bytes,
            media_type="text/csv; charset=utf-8-sig",
            headers={
                "Content-Disposition": f"attachment; filename=scored_{model_name}_{operating_point}.csv",
            },
        )


# ==============================================================================
# 4. Local Feature Attribution Endpoint
# ==============================================================================
@app.post("/explain/case", summary="單筆企業財務資料特徵歸因（嚴格僅 final 模式）")
def explain_case_endpoint(payload: ExplainCaseRequest):
    comp = get_components()
    raw_features_95 = comp["raw_features_95"]

    if payload.model_name not in ["M4", "M1"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Local explanation only supports 'M4' (TreeSHAP) or 'M1' (Linear), got '{payload.model_name}'.",
        )

    # Check missingness
    missing_cols = [c for c in raw_features_95 if c not in payload.record or pd.isna(payload.record[c])]
    missing_count = len(missing_cols)
    missing_ratio = missing_count / 95.0

    if missing_ratio > 0.20:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": f"Sample rejected: missing feature ratio {missing_ratio:.1%} ({missing_count}/95) exceeds allowed limit (20%).",
                "missing_feature_count": missing_count,
                "missing_feature_ratio": missing_ratio,
                "missing_features": missing_cols,
            },
        )

    try:
        # Strictly enforce mode='final' as mandated by specification
        result = explain_case(
            payload.record,
            model_name=payload.model_name,
            mode="final",
            top_k=payload.top_k,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    sanitized_res = sanitize_nan(result)
    sanitized_res["disclaimer"] = EXPLAIN_DISCLAIMER_TEXT
    return JSONResponse(content=sanitized_res)
