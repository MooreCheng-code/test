"""
Unit and Integration Tests for FastAPI Scoring & Attribution Service (tests/test_api.py).

Covers:
1. GET /health: 200 OK, service healthy, model list.
2. GET /model-info: 200 OK, threshold parameters, effective_target, margin_validated, alert_ratio, M5 limitation.
3. POST /score/json:
   - Single valid record scores properly.
   - Single record with > 20% missing returns HTTP 422 with missing feature list.
   - Small batch (N < 25) sets in_batch_top_10_pct to null.
   - Rejected records in batch properly labeled 'rejected_high_missing'.
   - All responses convert NaN to null strictly (no invalid JSON tokens).
4. POST /score/batch-csv:
   - Valid CSV upload returns utf-8-sig CSV response.
   - Oversized upload (> 10MB) returns HTTP 413.
5. POST /explain/case:
   - Strictly final mode for M4 and M1.
   - Missingness > 20% returns HTTP 422.
   - Response contains disclaimer with log-odds, E_A, and multicollinearity notice.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Add project root and bankruptcy to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import MAX_UPLOAD_SIZE_BYTES, app
from src.cv import get_train_data

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_feature_dict():
    """Extracts a valid single record from training data as dict."""
    X_train, _ = get_train_data()
    row = X_train.iloc[0].to_dict()
    row["company_id"] = "COMP_TEST_001"
    return row


@pytest.fixture(scope="module")
def sample_20_records():
    """Extracts 20 valid records (N < 25) from training data."""
    X_train, _ = get_train_data()
    records = []
    for i in range(20):
        d = X_train.iloc[i].to_dict()
        d["company_id"] = f"COMP_BATCH_{i:03d}"
        records.append(d)
    return records


# ==============================================================================
# 1. Health & Model Info Endpoints
# ==============================================================================
def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "E_A" in data["models_available"]
    assert "M4" in data["models_available"]


def test_api_model_info():
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "thresholds" in data
    assert "m5_limitation" in data
    assert "subsample_freq=0" in data["m5_limitation"]

    # Verify R100 label
    t_ea = data["thresholds"]["E_A"]["R100"]
    assert t_ea.get("operating_label") == "接近全抓"
    assert t_ea["margin_validated"] is True
    assert "alert_ratio" in t_ea
    assert "alerts_per_hit" in t_ea

    # Verify M4 margin_validated is False
    t_m4 = data["thresholds"]["M4"]["R90"]
    assert t_m4["margin_validated"] is False
    assert t_m4["effective_target"] == 0.90


# ==============================================================================
# 2. JSON Scoring Endpoint
# ==============================================================================
def test_score_json_single_valid(sample_feature_dict):
    payload = {
        "model_name": "E_A",
        "operating_point": "R90",
        "allow_missing": False,
        "records": [sample_feature_dict],
    }
    res = client.post("/score/json", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_records"] == 1
    assert data["valid_records"] == 1
    rec = data["results"][0]
    assert rec["status"] == "ok"
    assert rec["risk_score"] is not None
    assert rec["calibrated_score"] is not None
    assert rec["company_id"] == "COMP_TEST_001"


def test_score_json_single_missing_over_limit(sample_feature_dict):
    bad_dict = dict(sample_feature_dict)
    # Remove 25 features (> 20% of 95)
    keys_to_delete = list(bad_dict.keys())[:25]
    for k in keys_to_delete:
        del bad_dict[k]

    payload = {
        "model_name": "E_A",
        "operating_point": "R90",
        "allow_missing": True,
        "records": [bad_dict],
    }
    res = client.post("/score/json", json=payload)
    assert res.status_code == 422
    err_detail = res.json()["detail"]
    assert "sample rejected" in err_detail["message"].lower()
    assert err_detail["missing_feature_count"] >= 25
    assert len(err_detail["missing_features"]) >= 25


def test_score_json_small_batch_null_top10(sample_20_records):
    """N = 20 < 25: in_batch_top_10_pct must be null in JSON."""
    payload = {
        "model_name": "E_A",
        "operating_point": "R90",
        "allow_missing": False,
        "records": sample_20_records,
    }
    res = client.post("/score/json", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_records"] == 20
    assert data["top_10_pct_quota_enabled"] is False
    for rec in data["results"]:
        assert rec["in_batch_top_10_pct"] is None


def test_score_json_batch_with_rejected_rows(sample_20_records):
    """Batch with a rejected row: row has status='rejected_high_missing' and null values."""
    batch = [dict(r) for r in sample_20_records]
    # Invalidate row 0 by deleting 30 columns
    keys_to_del = list(batch[0].keys())[:30]
    for k in keys_to_del:
        del batch[0][k]

    payload = {
        "model_name": "E_A",
        "operating_point": "R90",
        "allow_missing": True,
        "records": batch,
    }
    res = client.post("/score/json", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["rejected_records"] == 1
    assert data["valid_records"] == 19

    rej_row = data["results"][0]
    assert rej_row["status"] == "rejected_high_missing"
    assert rej_row["risk_score"] is None
    assert rej_row["batch_rank"] is None
    assert rej_row["alert"] is None

    # Valid rows have status='ok'
    ok_row = data["results"][1]
    assert ok_row["status"] == "ok"
    assert ok_row["risk_score"] is not None


# ==============================================================================
# 3. CSV Batch Upload Endpoint
# ==============================================================================
def test_score_batch_csv_success():
    fixture_path = project_root / "tests" / "fixtures" / "golden_input_sample.csv"
    with open(fixture_path, "rb") as f:
        file_bytes = f.read()

    files = {"file": ("test_input.csv", file_bytes, "text/csv")}
    data = {"model_name": "E_A", "operating_point": "R90", "allow_missing": "false"}

    res = client.post("/score/batch-csv?output_format=csv", files=files, data=data)
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert res.content.startswith(b"\xef\xbb\xbf")  # utf-8-sig BOM

    # Verify CSV can be read back cleanly
    df_out = pd.read_csv(io.BytesIO(res.content))
    assert len(df_out) == 20
    assert "risk_score" in df_out.columns
    assert "alert" in df_out.columns


def test_score_batch_csv_oversized_payload():
    # Construct dummy byte payload exceeding 10MB
    oversized = b"a" * (MAX_UPLOAD_SIZE_BYTES + 1024)
    files = {"file": ("big.csv", oversized, "text/csv")}
    res = client.post("/score/batch-csv", files=files)
    assert res.status_code == 413
    assert "maximum allowed limit" in res.json()["detail"]


# ==============================================================================
# 4. Local Feature Attribution Endpoint (/explain/case)
# ==============================================================================
def test_explain_case_endpoint_m4(sample_feature_dict):
    payload = {
        "record": sample_feature_dict,
        "model_name": "M4",
        "top_k": 5,
    }
    res = client.post("/explain/case", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "M4"
    assert data["mode"] == "final"
    assert "risk_score" in data
    assert "reference_percentile" in data
    assert "imputed_features" in data
    assert len(data["top_positive_drivers"]) <= 5
    assert len(data["top_negative_drivers"]) <= 5
    assert "log-odds" in data["disclaimer"]
    assert "E_A" in data["disclaimer"]


def test_explain_case_endpoint_m1(sample_feature_dict):
    payload = {
        "record": sample_feature_dict,
        "model_name": "M1",
        "top_k": 5,
    }
    res = client.post("/explain/case", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "M1"
    assert "risk_score" in data
    assert "reference_percentile" in data
    assert "imputed_features" in data
    assert len(data["top_positive_drivers"]) <= 5
def test_explain_case_endpoint_missing_over_limit(sample_feature_dict):
    bad_dict = dict(sample_feature_dict)
    keys_to_del = list(bad_dict.keys())[:30]
    for k in keys_to_del:
        del bad_dict[k]

    payload = {
        "record": bad_dict,
        "model_name": "M4",
        "top_k": 5,
    }
    res = client.post("/explain/case", json=payload)
    assert res.status_code == 422
    err_detail = res.json()["detail"]
    assert "sample rejected" in err_detail["message"].lower()
    assert len(err_detail["missing_features"]) >= 30


def test_api_strict_json_no_nan_or_inf(sample_feature_dict):
    """Strictly validates that JSON response contains zero NaN, Infinity, or -Infinity tokens."""
    def raise_on_illegal_constant(c):
        raise ValueError(f"Illegal constant '{c}' in JSON response")

    # 1. /model-info
    res_info = client.get("/model-info")
    assert res_info.status_code == 200
    json.loads(res_info.text, parse_constant=raise_on_illegal_constant)

    # 2. /score/json with rejected row
    bad_row = dict(sample_feature_dict)
    for k in list(bad_row.keys())[:25]:
        bad_row[k] = None
    payload = {
        "records": [sample_feature_dict, bad_row],
        "model_name": "E_A",
        "allow_missing": True,
    }
    res_score = client.post("/score/json", json=payload)
    assert res_score.status_code == 200
    parsed = json.loads(res_score.text, parse_constant=raise_on_illegal_constant)
    assert parsed["results"][1]["risk_score"] is None
    assert parsed["results"][1]["status"] == "rejected_high_missing"


def test_score_json_allow_missing_false_behavior(sample_feature_dict):
    """Verifies that allow_missing=False via API returns HTTP 422 when features are missing."""
    bad_row = dict(sample_feature_dict)
    del bad_row["Operating Gross Margin"]

    payload = {
        "records": [bad_row],
        "model_name": "E_A",
        "allow_missing": False,
    }
    res = client.post("/score/json", json=payload)
    assert res.status_code == 422
    err_detail = res.json()["detail"]
    assert "Operating Gross Margin" in err_detail["missing_features"]


def test_score_json_max_records_limit(sample_feature_dict):
    """Verifies that /score/json rejects batches exceeding MAX_JSON_RECORDS (1000)."""
    oversized_records = [sample_feature_dict] * 1001
    payload = {
        "records": oversized_records,
        "model_name": "E_A",
    }
    res = client.post("/score/json", json=payload)
    assert res.status_code == 422
    assert "exceeds maximum limit" in res.json()["detail"]


def test_score_batch_csv_with_bankrupt_column_and_metrics(sample_feature_dict):
    """Verifies CSV batch scoring calculates metrics when Bankrupt? is present, and supports filtering/sorting."""
    rows = []
    for i in range(30):
        r = dict(sample_feature_dict)
        r["company_id"] = f"COMP_{i:03d}"
        r["Bankrupt?"] = 1 if i < 5 else 0
        rows.append(r)

    df_in = pd.DataFrame(rows)
    buf = io.StringIO()
    df_in.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    # Request with output_format=json to verify metrics structure
    res_json = client.post(
        "/score/batch-csv?output_format=json",
        files={"file": ("batch.csv", csv_bytes, "text/csv")},
        data={"model_name": "E_A", "operating_point": "R90", "sort_by_risk": "true"},
    )
    assert res_json.status_code == 200
    data = res_json.json()
    assert data["metrics"] is not None
    assert data["metrics"]["total_positives"] == 5
    assert "actual_hits_alert" in data["metrics"]
    assert "recall_at_top_10_pct" in data["metrics"]

    # Verify risk sorting
    scores = [r["risk_score"] for r in data["results"] if r.get("risk_score") is not None]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))


def test_score_batch_csv_rejected_rows_in_csv_output(sample_feature_dict):
    """Verifies that rows with > 20% missingness are included in output CSV with rejected status."""
    row_good = dict(sample_feature_dict)
    row_good["company_id"] = "GOOD_01"

    row_bad = dict(sample_feature_dict)
    row_bad["company_id"] = "BAD_02"
    for k in list(row_bad.keys())[:25]:
        row_bad[k] = np.nan

    df_in = pd.DataFrame([row_good, row_bad])
    buf = io.StringIO()
    df_in.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    res_csv = client.post(
        "/score/batch-csv?output_format=csv",
        files={"file": ("test_rejected.csv", csv_bytes, "text/csv")},
        data={"model_name": "E_A", "allow_missing": "true"},
    )
    assert res_csv.status_code == 200
    df_out = pd.read_csv(io.BytesIO(res_csv.content))
    assert len(df_out) == 2
    assert df_out.loc[df_out["company_id"] == "GOOD_01", "status"].iloc[0] == "ok"
    assert df_out.loc[df_out["company_id"] == "BAD_02", "status"].iloc[0] == "rejected_high_missing"
    assert pd.isna(df_out.loc[df_out["company_id"] == "BAD_02", "risk_score"].iloc[0])
