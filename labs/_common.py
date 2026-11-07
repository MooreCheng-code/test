"""labs/_common.py - Common utilities, safety guards, CV splitters, and reporting for labs.

SAFETY RULES:
1. ONLY load training data via load_train_data() which calls bankruptcy.src.cv.get_train_data().
   NEVER call get_test_data(), load_splits(), save_splits(), or touch splits.joblib / E6_LOCK.
2. Lab CV splits are constructed on the training set using StratifiedKFold or RepeatedStratifiedKFold.
   Never reuse project cv_splits file.
3. Outputs are written exclusively to labs/outputs/<lab_id>/, never to bankruptcy/outputs/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

# Pure computation imports from bankruptcy package
from bankruptcy import config
from bankruptcy.src.cv import get_train_data
from bankruptcy.src.metrics import alert_ratio_at_recall, brier, recall_at_alert_ratio
from bankruptcy.src.pipelines import build_pipeline, get_score

# Repository root and key directory paths
REPO_ROOT = Path(__file__).resolve().parent.parent
LABS_DIR = REPO_ROOT / "labs"
LABS_OUTPUTS_DIR = LABS_DIR / "outputs"
LAB_SNAPSHOTS_DIR = REPO_ROOT / "docs" / "code_learning" / "lab_snapshots"
SNAPSHOT_DIR = LAB_SNAPSHOTS_DIR
BEST_PARAMS_DIR = REPO_ROOT / "bankruptcy" / "outputs" / "models" / "best_params"
TABLES_DIR = REPO_ROOT / "bankruptcy" / "outputs" / "tables"


def get_project_pipeline_flags() -> dict[str, Any]:
    """Extract project pipeline flags from tune.py AST.

    Reads tune.py AST and extracts kwargs passed to build_pipeline.
    Returns:
        dict: {'scaler': 'quantile', 'use_winsorizer': False, 'use_corr_filter': True, 'use_large_indicator': False}
    """
    import ast
    tune_file = REPO_ROOT / "bankruptcy" / "src" / "tune.py"
    if not tune_file.is_file():
        raise FileNotFoundError(f"tune.py not found at {tune_file}")

    with open(tune_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if func_name == "build_pipeline":
                kw_map = {}
                for kw in node.keywords:
                    if kw.arg in ("scaler", "use_winsorizer", "use_corr_filter", "use_large_indicator"):
                        kw_map[kw.arg] = ast.literal_eval(kw.value)
                if len(kw_map) == 4:
                    return kw_map
    raise RuntimeError("Failed to extract project pipeline flags from tune.py AST")


def get_pipeline_flags(pipeline_mode: str = "project") -> dict[str, Any]:
    """Get preprocessing flags for pipeline construction.

    Args:
        pipeline_mode: 'project' (matches final M3/M4/M5 without winsorizer)
                       or 'teaching' (build_pipeline default with winsorizer).
    """
    if pipeline_mode == "project":
        return get_project_pipeline_flags()
    elif pipeline_mode == "teaching":
        return {
            "scaler": "quantile",
            "use_winsorizer": True,
            "use_corr_filter": True,
            "use_large_indicator": False,
        }
    else:
        raise ValueError(f"Unknown pipeline_mode: '{pipeline_mode}'. Expected 'project' or 'teaching'.")



def assert_train_data_guard(X: pd.DataFrame, y: Sequence | pd.Series | np.ndarray) -> None:
    """Strict guard assertion to guarantee train-only data is accessed.

    Raises:
        RuntimeError: if full dataset, wrong row count, or wrong positive count is detected.
    """
    n_samples = len(X)
    n_y = len(y)
    if n_samples == 6819 or n_y == 6819:
        raise RuntimeError(
            "GUARD FAILURE: Detected full dataset (6819 rows) instead of train-only data! "
            "Access to holdout test data is strictly prohibited."
        )
    if n_samples != 5455 or n_y != 5455:
        raise RuntimeError(
            f"GUARD FAILURE: Expected exactly 5455 training rows, got len(X)={n_samples}, len(y)={n_y}! Aborting."
        )

    pos_count = int(np.sum(y))
    if pos_count != 176:
        raise RuntimeError(
            f"GUARD FAILURE: Expected exactly 176 positive samples in training set, got {pos_count}! Aborting."
        )


def load_train_data(check_guard: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Safely load training data with guard assertions.

    Only accesses get_train_data(). Never accesses holdout test data.
    """
    X_train, y_train = get_train_data()
    if check_guard:
        assert_train_data_guard(X_train, y_train)
    return X_train, y_train


def get_cv_splitter(
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = config.SEED,
) -> StratifiedKFold | RepeatedStratifiedKFold:
    """Create a stratified CV splitter independently on the training data.

    Note: This is created on the 5,455 training records and does not touch
    or reuse the precomputed splits.joblib file.
    """
    if n_repeats == 1:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def evaluate_fold_predictions(
    y_true: np.ndarray | Sequence | pd.Series,
    score_pred: np.ndarray,
    proba_pred: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate ranking metrics and calibration metrics for a validation fold."""
    y_true_arr = np.asarray(y_true, dtype=int)
    score_pred_arr = np.asarray(score_pred, dtype=float)

    # Ranking metrics
    pr_auc = float(average_precision_score(y_true_arr, score_pred_arr))
    roc_auc = float(roc_auc_score(y_true_arr, score_pred_arr))
    ar_80 = float(alert_ratio_at_recall(y_true_arr, score_pred_arr, target_recall=0.80))
    ar_90 = float(alert_ratio_at_recall(y_true_arr, score_pred_arr, target_recall=0.90))
    ar_95 = float(alert_ratio_at_recall(y_true_arr, score_pred_arr, target_recall=0.95))

    # Recall at alert ratio
    r_at_ar15 = float(recall_at_alert_ratio(y_true_arr, score_pred_arr, ratio=1.5))
    r_at_ar30 = float(recall_at_alert_ratio(y_true_arr, score_pred_arr, ratio=3.0))

    # Calibration / probability metrics
    eval_proba = proba_pred if proba_pred is not None else score_pred_arr
    mean_proba = float(np.mean(eval_proba))
    brier_score = float(brier(y_true_arr, eval_proba))
    n_unique = int(len(np.unique(score_pred_arr)))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "ar_80": ar_80,
        "ar_90": ar_90,
        "ar_95": ar_95,
        "recall_at_ar1.5": r_at_ar15,
        "recall_at_ar3.0": r_at_ar30,
        "mean_proba": mean_proba,
        "brier": brier_score,
        "n_unique_scores": n_unique,
        "n_positives": int(np.sum(y_true_arr)),
        "n_samples": int(len(y_true_arr)),
    }


def load_best_params(model_name: str) -> dict[str, Any]:
    """Read-only loader for frozen best_params/<model_name>.json."""
    path = BEST_PARAMS_DIR / f"{model_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Best params file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_chinese_font() -> str:
    """Configure matplotlib font to handle Traditional / Simplified Chinese.

    Falls back gracefully to English without errors if Chinese font is missing.
    """
    candidate_fonts = [
        "Microsoft JhengHei",  # Windows Traditional Chinese
        "SimHei",              # Windows Simplified Chinese
        "Arial Unicode MS",
        "PingFang SC",
        "DejaVu Sans",
    ]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    chosen = "sans-serif"
    for font in candidate_fonts:
        if font in available:
            chosen = font
            break

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [chosen] + candidate_fonts + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def get_git_head() -> str:
    """Safely obtain the current Git HEAD commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_git_status_info() -> tuple[bool, int]:
    """Check git status: returns (git_dirty_labs, git_dirty_other_count).

    git_dirty_labs: True if any changes in labs/ or tools/
    git_dirty_other_count: count of modified/untracked files outside labs/ and tools/
    """
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        labs_dirty = False
        other_count = 0
        for line in lines:
            if line.startswith("??"):
                other_count += 1
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                path = parts[1].strip('"')
            else:
                path = line[2:].strip().strip('"')
            if path.startswith("labs/") or path.startswith("tools/"):
                labs_dirty = True
            else:
                other_count += 1
        return labs_dirty, other_count
    except Exception:
        return False, 0


def get_git_dirty(target_path: str = "labs") -> bool:
    """Check if the specified path has uncommitted changes in Git."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain", target_path],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(res.stdout.strip())
    except Exception:
        return False


def save_manifest(
    output_dir: str | Path,
    lab_id: str,
    params: dict[str, Any],
    elapsed_seconds: float,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save run_manifest.json with execution metadata."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    package_names = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "lightgbm",
        "xgboost",
        "optuna",
        "matplotlib",
    ]
    packages_info: dict[str, str] = {}
    for pkg in package_names:
        try:
            mod = __import__(pkg)
            packages_info[pkg] = getattr(mod, "__version__", "installed")
        except Exception:
            packages_info[pkg] = "not_available"

    git_dirty_labs, git_dirty_other_count = get_git_status_info()

    manifest = {
        "lab_id": lab_id,
        "git_head": get_git_head(),
        "git_dirty": git_dirty_labs,
        "git_dirty_labs": git_dirty_labs,
        "git_dirty_other_count": git_dirty_other_count,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "seed": seed,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "packages": packages_info,
        "params": params,
    }
    if extra:
        manifest["extra"] = extra

    manifest_file = out_path / "run_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_file


def save_snapshot(df: pd.DataFrame, snapshot_name: str) -> Path:
    """Save small snapshot CSV into docs/code_learning/lab_snapshots/ for versioning."""
    LAB_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    target = LAB_SNAPSHOTS_DIR / snapshot_name
    df.to_csv(target, index=False, encoding="utf-8")
    return target


# Future hooks for subsequent labs (lab_06, lab_07, lab_08)
def generate_inner_oof(
    model_name: str,
    imbalance: str,
    model_params: dict[str, Any] | None,
    X_train_fold: pd.DataFrame,
    y_train_fold: pd.Series,
    n_inner_splits: int = 3,
    inner_seed: int = 42,
    pipeline_mode: str = "project",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate inner out-of-fold probability predictions on a training fold (for lab_06)."""
    flags = get_pipeline_flags(pipeline_mode)
    skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=inner_seed)
    oof_scores = np.zeros(len(y_train_fold))
    for inner_train_idx, inner_val_idx in skf.split(X_train_fold, y_train_fold):
        X_it = X_train_fold.iloc[inner_train_idx]
        y_it = y_train_fold.iloc[inner_train_idx]
        X_iv = X_train_fold.iloc[inner_val_idx]

        pipe = build_pipeline(
            model_name=model_name,
            imbalance=imbalance,
            model_params=model_params,
            seed=inner_seed,
            **flags,
        )
        pipe.fit(X_it, y_it)
        oof_scores[inner_val_idx] = get_score(pipe, X_iv)
    return oof_scores, y_train_fold.to_numpy()
