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
- Upload stream limits: aborts during 1MB chunked streaming when payload exceeds 10MB.
- JSON responses strictly sanitize NaN / Infinity to null.
- Single-case exceeding 20% missingness returns HTTP 422 with the list of missing features.
- All scoring responses include standard business and legal disclaimers.
"""

from __future__ import annotations

import io
import math
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from src.local_explain import explain_case, get_local_explainer
from src.scoring import load_mvp_components, score_mvp_batch

# Maximum upload limit: 10 MB
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MB stream chunk
MAX_JSON_RECORDS = 1000

HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY)
HTTP_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

DISCLAIMER_TEXT = (
    "本系統模型預測與評估結果僅供內部審查與風險控管流程參考，"
    "非授信核准或法律效力保證。"
)

EXPLAIN_DISCLAIMER_TEXT = (
    "特徵貢獻由 M4 (XGBoost TreeSHAP) 與 M1 (正規化迴歸係數) 計算，尺度為對數勝算（log-odds）。"
    "前端主要排序與集成模型為 E_A (三模型百分位均值)，"
    "單一模型歸因僅供業務審查時掌握風險特徵驅動方向參考；"
    "由於財務變數間存在高度共線性，各特徵貢獻度分配具有相對性，"
    "不代表改變單一變數即可線性改變破產風險。"
)

# Global cached components
_COMPONENTS = None


def get_components():
    global _COMPONENTS
    if _COMPONENTS is None:
        _COMPONENTS = load_mvp_components()
    return _COMPONENTS


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load MVP components and explainer on startup
    get_components()
    get_local_explainer()
    yield


app = FastAPI(
    title="企業破產預警評分服務 API",
    description="提供企業破產風險評分、百分位排序、Top 10% 審查標記與 TreeSHAP 特徵歸因之 REST 服務。",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS restricted strictly to localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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
@app.get("/health", summary="服務健全狀態檢查")
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
        "CV 評估 M5 啟用 subsample_freq=1，最終凍結 M5 為 subsample_freq=0；"
        "CV 指標僅供選模參考，最終效能以 E6 測試結果為準；"
        "E5 之 M5 全域 SHAP 亦使用未含該參數的重建設定。"
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
@app.post("/score/json", summary="JSON 格式單筆或批次財務數據評分")
def score_json(payload: ScoreJsonRequest):
    comp = get_components()
    raw_features_95 = comp["raw_features_95"]

    if not payload.records:
        raise HTTPException(
            status_code=HTTP_422,
            detail="The 'records' list cannot be empty.",
        )

    if len(payload.records) > MAX_JSON_RECORDS:
        raise HTTPException(
            status_code=HTTP_422,
            detail=f"The 'records' list exceeds maximum limit of {MAX_JSON_RECORDS} records. Please use /score/batch-csv for larger batches.",
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
                status_code=HTTP_422,
                detail={
                    "message": f"Input data is missing {missing_count} required feature(s) for scoring while allow_missing=False.",
                    "missing_feature_count": missing_count,
                    "missing_feature_ratio": missing_ratio,
                    "missing_features": missing_cols,
                },
            )

        if missing_ratio > 0.20:
            raise HTTPException(
                status_code=HTTP_422,
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
            status_code=HTTP_422,
            detail=str(e),
        )

    # Convert DataFrame to records, replacing NaN with null
    records_out = scored_df.to_dict(orient="records")
    sanitized_records = [sanitize_nan(r) for r in records_out]

    n_total = len(df_input)
    n_rejected = sum(1 for r in sanitized_records if r.get("status") == "rejected_high_missing")
    n_valid = n_total - n_rejected

    # Optional Bankrupt? evaluation metrics
    metrics_data = None
    if "Bankrupt?" in df_input.columns:
        valid_mask = (scored_df["status"] == "ok")
        y_true = pd.to_numeric(df_input.loc[valid_mask, "Bankrupt?"], errors="coerce").fillna(0).astype(int)
        total_pos = int((y_true == 1).sum())
        hits_alert = int(((y_true == 1) & (scored_df.loc[valid_mask, "alert"] == True)).sum())

        if n_valid < 25 or total_pos == 0:
            recall_top10 = None
            hits_top10 = None
            metrics_note = "N_valid < 25 或總破產樣本數為 0，依規範不計算 Recall@Top10%"
        else:
            hits_top10 = int(((y_true == 1) & (scored_df.loc[valid_mask, "in_batch_top_10_pct"] == True)).sum())
            recall_top10 = float(hits_top10 / total_pos)
            metrics_note = "Recall@Top10% 依有效評分樣本計算"

        metrics_data = {
            "total_positives": total_pos,
            "actual_hits_alert": hits_alert,
            "actual_hits_top_10_pct": hits_top10,
            "recall_at_top_10_pct": recall_top10,
            "note": metrics_note,
        }

    response_data: dict[str, Any] = {
        "model_name": payload.model_name,
        "operating_point": payload.operating_point,
        "total_records": n_total,
        "valid_records": n_valid,
        "rejected_records": n_rejected,
        "top_10_pct_quota_enabled": bool(n_valid >= 25),
        "metrics": metrics_data,
        "results": sanitized_records,
        "disclaimer": DISCLAIMER_TEXT,
    }

    return JSONResponse(content=response_data)


# ==============================================================================
# 3. CSV Batch Upload Scoring Endpoint
# ==============================================================================
@app.post("/score/batch-csv", summary="CSV 檔案上傳批次評分（記憶體串流）")
async def score_batch_csv(
    file: UploadFile = File(..., description="Input CSV file containing company records"),
    model_name: str = Form(default="E_A"),
    operating_point: str = Form(default="R90"),
    allow_missing: bool = Form(default=False),
    high_risk_only: bool = Form(default=False),
    sort_by_risk: bool = Form(default=False),
    output_format: str = Query(default="csv", regex="^(csv|json)$"),
):
    # Stream in 1MB chunks and abort immediately if payload exceeds 10MB
    chunks = []
    total_bytes = 0
    while True:
        chunk = await file.read(CHUNK_SIZE_BYTES)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=HTTP_413,
                detail=f"Uploaded file exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB. Upload aborted during stream.",
            )
        chunks.append(chunk)

    content_bytes = b"".join(chunks)

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
            status_code=HTTP_422,
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
            status_code=HTTP_422,
            detail=str(e),
        )

    # Calculate metrics if Bankrupt? is present
    metrics_data = None
    n_total = len(df_input)
    n_rejected = int((scored_df["status"] == "rejected_high_missing").sum())
    n_valid = n_total - n_rejected

    if "Bankrupt?" in df_input.columns:
        valid_mask = (scored_df["status"] == "ok")
        y_true = pd.to_numeric(df_input.loc[valid_mask, "Bankrupt?"], errors="coerce").fillna(0).astype(int)
        total_pos = int((y_true == 1).sum())
        hits_alert = int(((y_true == 1) & (scored_df.loc[valid_mask, "alert"] == True)).sum())

        if n_valid < 25 or total_pos == 0:
            recall_top10 = None
            hits_top10 = None
            metrics_note = "N_valid < 25 或總破產樣本數為 0，依規範不計算 Recall@Top10%"
        else:
            hits_top10 = int(((y_true == 1) & (scored_df.loc[valid_mask, "in_batch_top_10_pct"] == True)).sum())
            recall_top10 = float(hits_top10 / total_pos)
            metrics_note = "Recall@Top10% 依有效評分樣本計算"

        metrics_data = {
            "total_positives": total_pos,
            "actual_hits_alert": hits_alert,
            "actual_hits_top_10_pct": hits_top10,
            "recall_at_top_10_pct": recall_top10,
            "note": metrics_note,
        }

    # Apply optional filtering and sorting
    if high_risk_only:
        scored_df = scored_df[scored_df["alert"] == True]
    if sort_by_risk:
        scored_df = scored_df.sort_values(by="risk_score", ascending=False)

    if output_format == "json":
        records_out = scored_df.to_dict(orient="records")
        sanitized_records = [sanitize_nan(r) for r in records_out]
        return JSONResponse(
            content={
                "model_name": model_name,
                "operating_point": operating_point,
                "total_records": len(df_input),
                "valid_records": n_valid,
                "rejected_records": n_rejected,
                "metrics": metrics_data,
                "results": sanitized_records,
                "disclaimer": DISCLAIMER_TEXT,
            }
        )
    else:
        # Export as Excel-friendly utf-8-sig CSV
        out_buf = io.StringIO()
        scored_df.to_csv(out_buf, index=False)
        csv_output_bytes = out_buf.getvalue().encode("utf-8-sig")

        resp_headers = {
            "Content-Disposition": f"attachment; filename=scored_{model_name}_{operating_point}.csv",
        }
        if metrics_data:
            resp_headers["X-Total-Positives"] = str(metrics_data["total_positives"])
            resp_headers["X-Actual-Hits-Alert"] = str(metrics_data["actual_hits_alert"])
            resp_headers["X-Recall-Top-10-Pct"] = str(metrics_data["recall_at_top_10_pct"]) if metrics_data["recall_at_top_10_pct"] is not None else "null"

        return Response(
            content=csv_output_bytes,
            media_type="text/csv; charset=utf-8-sig",
            headers=resp_headers,
        )


# ==============================================================================
# 4. Local Feature Attribution Endpoint
# ==============================================================================
@app.post("/explain/case", summary="單筆企業財務數據特徵歸因（限 final 模式）")
def explain_case_endpoint(payload: ExplainCaseRequest):
    comp = get_components()
    raw_features_95 = comp["raw_features_95"]

    if payload.model_name not in ["M4", "M1"]:
        raise HTTPException(
            status_code=HTTP_422,
            detail=f"Local explanation only supports 'M4' (TreeSHAP) or 'M1' (Linear), got '{payload.model_name}'.",
        )

    # Check missingness
    missing_cols = [c for c in raw_features_95 if c not in payload.record or pd.isna(payload.record[c])]
    missing_count = len(missing_cols)
    missing_ratio = missing_count / 95.0

    if missing_ratio > 0.20:
        raise HTTPException(
            status_code=HTTP_422,
            detail={
                "message": f"Single sample rejected: missing feature ratio {missing_ratio:.1%} ({missing_count}/95) exceeds allowed limit (20%).",
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
            status_code=HTTP_422,
            detail=str(e),
        )

    sanitized_res = sanitize_nan(result)
    sanitized_res["disclaimer"] = EXPLAIN_DISCLAIMER_TEXT
    return JSONResponse(content=sanitized_res)
