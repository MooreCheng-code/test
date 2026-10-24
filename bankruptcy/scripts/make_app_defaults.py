"""
scripts/make_app_defaults.py

Generates app/assets/defaults.json strictly from training data.
Ensures zero exposure to out-of-fold holdout test data.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cv import get_train_data


def generate_app_defaults() -> Path:
    final_models_dir = PROJECT_ROOT / "outputs" / "models" / "final"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    data_path = PROJECT_ROOT / "data" / "data.csv"
    splits_path = PROJECT_ROOT / "outputs" / "models" / "splits.joblib"
    out_path = PROJECT_ROOT / "app" / "assets" / "defaults.json"

    # 1. Load frozen model pipeline components
    m3_path = final_models_dir / "model_M3.joblib"
    if not m3_path.exists():
        raise FileNotFoundError(f"Frozen pipeline not found at {m3_path}")
    pipe_m3 = joblib.load(m3_path)

    raw_columns = list(pipe_m3.named_steps["feature_selector"].selected_features_)
    model_columns = list(pipe_m3.named_steps["corr_filter"].selected_features_)
    if len(raw_columns) != 95:
        raise ValueError(f"Expected 95 raw columns, got {len(raw_columns)}")
    if len(model_columns) != 72:
        raise ValueError(f"Expected 72 model columns, got {len(model_columns)}")

    # 2. Load 18 core features
    core_txt_path = final_models_dir / "core_features_extended.txt"
    if not core_txt_path.exists():
        raise FileNotFoundError(f"Core features file not found at {core_txt_path}")
    with open(core_txt_path, "r", encoding="utf-8") as f:
        core_features = [line.strip() for line in f if line.strip()]

    if len(core_features) != 18:
        raise ValueError(f"Expected 18 core features, got {len(core_features)}")
    for feat in core_features:
        if feat not in model_columns:
            raise ValueError(f"Core feature '{feat}' is not in model_columns!")

    fill_columns = [c for c in model_columns if c not in core_features]
    unused_columns = [c for c in raw_columns if c not in model_columns]
    if len(fill_columns) != 54:
        raise ValueError(f"Expected 54 fill columns, got {len(fill_columns)}")
    if len(unused_columns) != 23:
        raise ValueError(f"Expected 23 unused columns, got {len(unused_columns)}")

    # 3. Retrieve training data only
    X_train, y_train = get_train_data(data_path)
    if list(X_train.columns) != raw_columns:
        raise ValueError("Training data columns do not match raw_columns order!")

    # Verify train index hash integrity
    splits = joblib.load(splits_path)
    tr_hash = hashlib.sha256(np.ascontiguousarray(splits["train_idx"]).tobytes()).hexdigest()
    hashes_path = final_models_dir / "index_hashes.json"
    with open(hashes_path, "r", encoding="utf-8") as f:
        saved_hashes = json.load(f)
    if tr_hash != saved_hashes["train_indices_sha256"]:
        raise ValueError("Train index hash mismatch against index_hashes.json!")

    # 4. Compute 95 raw medians
    medians = {col: float(X_train[col].median()) for col in raw_columns}

    # 5. Compute feature tiers from permutation importance
    perm_csv_path = tables_dir / "e5a_permutation_importance.csv"
    if not perm_csv_path.exists():
        raise FileNotFoundError(f"Permutation importance table not found at {perm_csv_path}")
    perm_df = pd.read_csv(perm_csv_path)

    sub_perm = perm_df[
        perm_df["feature"].isin(core_features) & perm_df["model"].isin(["M3", "M4", "M5"])
    ]
    importance_mean = (
        sub_perm.groupby("feature")["pr_auc_drop_mean"]
        .mean()
        .sort_values(ascending=False)
    )

    inputs = {}
    for rank, (feat, imp) in enumerate(importance_mean.items(), start=1):
        tier = "main" if rank <= 8 else "advanced"
        s = X_train[feat]
        inputs[feat] = {
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "tier": tier,
            "importance_rank": int(rank),
            "importance_mean": float(imp),
            "importance_metric": "pr_auc_drop_mean",
        }

    # 6. Compute virtual preset profiles
    X_train_core = X_train[core_features]
    normal_subset = X_train_core[y_train == 0]
    bankrupt_subset = X_train_core[y_train == 1]

    normal_profile_vals = {col: float(normal_subset[col].median()) for col in core_features}
    bankrupt_profile_vals = {col: float(bankrupt_subset[col].median()) for col in core_features}

    profile_note = "訓練資料彙總出來的虛擬輪廓，不是任何一家真實公司"

    # Assert virtual profiles do not match any row in training data
    match_norm = (X_train_core == pd.Series(normal_profile_vals)).all(axis=1).any()
    match_bank = (X_train_core == pd.Series(bankrupt_profile_vals)).all(axis=1).any()
    if match_norm:
        raise ValueError("Normal profile accidentally matches a real training row!")
    if match_bank:
        raise ValueError("Bankrupt profile accidentally matches a real training row!")

    presets = {
        "description": profile_note,
        "normal_profile": {
            "label": "典型正常公司",
            "description": profile_note,
            "values": normal_profile_vals,
            **normal_profile_vals,
        },
        "bankrupt_profile": {
            "label": "訓練中破產公司的典型輪廓",
            "description": profile_note,
            "values": bankrupt_profile_vals,
            **bankrupt_profile_vals,
        },
    }

    # 7. Metadata
    meta = {
        "n_train": int(len(X_train)),
        "n_raw_columns": int(len(raw_columns)),
        "n_model_columns": int(len(model_columns)),
        "n_input_features": int(len(core_features)),
        "train_bankruptcy_count": int((y_train == 1).sum()),
        "train_bankruptcy_rate": float((y_train == 1).mean()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_indices_sha256": tr_hash,
    }

    payload = {
        "meta": meta,
        "raw_columns": raw_columns,
        "model_columns": model_columns,
        "fill_columns": fill_columns,
        "unused_columns": unused_columns,
        "medians": medians,
        "inputs": inputs,
        "presets": presets,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated defaults at: {out_path}")
    print(f"Meta: {json.dumps(meta, indent=2)}")
    print(f"raw_columns: {len(raw_columns)}, model_columns: {len(model_columns)}")
    print(f"fill_columns: {len(fill_columns)}, unused_columns: {len(unused_columns)}")
    print(f"inputs: {len(inputs)}")
    print("\nMain inputs (top 8 by PR-AUC drop):")
    for feat, info in inputs.items():
        if info["tier"] == "main":
            print(f"  {info['importance_rank']}. {feat}: {info['importance_mean']:.6f}")
    print("\nAdvanced inputs (ranks 9-18):")
    for feat, info in inputs.items():
        if info["tier"] == "advanced":
            print(f"  {info['importance_rank']}. {feat}: {info['importance_mean']:.6f}")

    return out_path


if __name__ == "__main__":
    generate_app_defaults()
