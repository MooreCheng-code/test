"""
E2: Hyperparameter Tuning Module (Nested Cross-Validation & Final Tuning).

Executes:
1. Phase E2a (Nested CV):
   - Outer Loop: Repeated Stratified K-Fold (Phase 1: repeat 0; Phase 2: repeats 0, 1, 2).
   - Inner Loop: 3-fold CV via cv.make_inner_splits.
   - Objective: Minimize mean(log(alert_ratio@r) for r in [0.80, 0.90, 0.95]),
     evaluated directly on POOLED inner OOF predictions (~141 positives).
   - Optuna TPESampler with SQLite storage per fold study.
   - Categorical hyperparameter: imbalance in {S0, S3}.
   - Preprocessing: Winsorizer OFF, CorrFilter ON, LargeIndicator OFF, Scaler quantile.
   - Outer evaluation: refit on full outer training fold, evaluate on outer validation fold.
   - Diagnostics: fold score distributions (pos/neg median, p90, p99),
     and Percentile Rank Pooled OOF to eliminate cross-fold scale distortion.
   - Paired tests vs strictly matched untuned baseline (e2_default_matched_per_fold.csv).
2. Phase E2b (Final Training on Full Training Set for E6):
   - CV: Repeated Stratified 5-fold x 2 repeats (10 folds).
   - Same objective & search space.
   - n_trials = 1.5x E2a (M1: 38 trials; M2-M5: 60 trials).
   - Saves final parameters to outputs/models/best_params/<model>.json.
   - Saves landscape flatness summary to outputs/tables/e2b_final_params_summary.csv.

Usage:
    python src/tune.py --outer-repeats 0             # Phase 1: 5 outer folds (Repeat 0)
    python src/tune.py --outer-repeats 0 1 2 --run-e2b # Phase 2 + E2b final tuning
    python src/tune.py --quick                       # Quick sanity check
"""

import argparse
import datetime
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import (
    CV_FOLDS,
    CV_REPEATS,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    SEED,
    TABLES_DIR,
)
from src.cv import (
    get_train_data,
    load_splits,
    make_inner_splits,
)
from src.metrics import (
    alert_ratio_at_recall,
    evaluate_scores,
    recall_at_alert_ratio,
)
from src.pipelines import build_pipeline
from src.stats_utils import (
    holm_bonferroni,
    nadeau_bengio_paired_test,
    wilcoxon_test,
)

# Directories setup
CACHE_DIR = LOGS_DIR / "e2_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STUDIES_DIR = LOGS_DIR / "optuna_studies"
STUDIES_DIR.mkdir(parents=True, exist_ok=True)

OOF_DIR = MODELS_DIR / "oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS_DIR = MODELS_DIR / "best_params"
BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)

# Silence noisy Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "e2_tuning.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("tune")


# ==============================================================================
# Objective Function & Search Space Helpers
# ==============================================================================
def compute_three_point_log_loss(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Computes mean(log(alert_ratio@recall_r) for r in [0.80, 0.90, 0.95])."""
    ar80 = alert_ratio_at_recall(y_true, scores, 0.80)
    ar90 = alert_ratio_at_recall(y_true, scores, 0.90)
    ar95 = alert_ratio_at_recall(y_true, scores, 0.95)
    return float(np.mean([np.log(ar80), np.log(ar90), np.log(ar95)]))


def sample_hyperparameters(trial: optuna.Trial, model_name: str) -> tuple[str, dict[str, Any]]:
    """Sample imbalance strategy and model-specific hyperparameters."""
    imbalance = trial.suggest_categorical("imbalance", ["S0", "S3"])
    params: dict[str, Any] = {}

    if model_name == "M1":
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l2", "l1"])
        if penalty == "l1":
            solver = "saga"
            l1_ratio = 1.0
        else:
            solver = "lbfgs"
            l1_ratio = None
        params = {
            "C": C,
            "penalty": penalty,
            "solver": solver,
            "max_iter": 5000,
            "l1_ratio": l1_ratio,
        }

    elif model_name in ["M2", "M3"]:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8, 12, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
        }

    elif model_name == "M4":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        }

    elif model_name == "M5":
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "subsample": subsample,
            "subsample_freq": 1 if subsample < 1.0 else 0,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return imbalance, params


def create_inner_objective(
    model_name: str,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    X_outer_tr: pd.DataFrame,
    y_outer_tr: pd.Series,
    outer_train_idx: np.ndarray,
):
    """
    Creates an Optuna objective evaluating hyperparameters on pooled inner OOF.
    """
    n_outer = len(outer_train_idx)
    y_outer_arr = y_outer_tr.values
    idx_to_pos = {idx: i for i, idx in enumerate(outer_train_idx)}

    # Precompute relative positions once
    inner_split_positions = []
    for in_tr_idx, in_val_idx in inner_splits:
        tr_pos = np.array([idx_to_pos[i] for i in in_tr_idx])
        val_pos = np.array([idx_to_pos[i] for i in in_val_idx])
        inner_split_positions.append((tr_pos, val_pos))

    def objective(trial: optuna.Trial) -> float:
        imbalance, params = sample_hyperparameters(trial, model_name)

        def pipe_factory():
            return build_pipeline(
                model_name=model_name,
                imbalance=imbalance,
                scaler="quantile",
                use_winsorizer=False,
                use_corr_filter=True,
                use_large_indicator=False,
                model_params=params,
            )

        inner_oof_scores = np.zeros(n_outer, dtype=float)
        fold_losses = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr_positions, val_positions in inner_split_positions:
                pipe = pipe_factory()
                pipe.fit(X_outer_tr.iloc[tr_positions], y_outer_tr.iloc[tr_positions])
                scores_val = pipe.predict_proba(X_outer_tr.iloc[val_positions])[:, 1]
                inner_oof_scores[val_positions] = scores_val

                # Evaluate individual inner fold loss
                f_y = y_outer_arr[val_positions]
                f_loss = compute_three_point_log_loss(f_y, scores_val)
                fold_losses.append(f_loss)

        # Pooled inner OOF metric calculation
        pooled_loss = compute_three_point_log_loss(y_outer_arr, inner_oof_scores)
        ar80 = alert_ratio_at_recall(y_outer_arr, inner_oof_scores, 0.80)
        ar90 = alert_ratio_at_recall(y_outer_arr, inner_oof_scores, 0.90)
        ar95 = alert_ratio_at_recall(y_outer_arr, inner_oof_scores, 0.95)
        pr_auc = float(average_precision_score(y_outer_arr, inner_oof_scores))
        roc_auc = float(roc_auc_score(y_outer_arr, inner_oof_scores))

        fold_loss_std = float(np.std(fold_losses, ddof=1)) if len(fold_losses) > 1 else 0.0

        trial.set_user_attr("pooled_loss", pooled_loss)
        trial.set_user_attr("ar80", ar80)
        trial.set_user_attr("ar90", ar90)
        trial.set_user_attr("ar95", ar95)
        trial.set_user_attr("pr_auc", pr_auc)
        trial.set_user_attr("roc_auc", roc_auc)
        trial.set_user_attr("fold_losses", fold_losses)
        trial.set_user_attr("fold_loss_std", fold_loss_std)

        return pooled_loss

    return objective


# ==============================================================================
# Single Outer Fold Execution with Checkpoint Caching
# ==============================================================================
def run_single_outer_fold(
    model_name: str,
    repeat: int,
    fold: int,
    outer_train_idx: np.ndarray,
    outer_val_idx: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
) -> dict[str, Any]:
    """Executes inner tuning and outer evaluation for a single outer fold."""
    setting_id = f"{model_name}_r{repeat}_f{fold}"
    cache_file = CACHE_DIR / f"{setting_id}.joblib"

    if cache_file.exists():
        try:
            cached_data = joblib.load(cache_file)
            logger.info(f"Loaded {setting_id} from cache (Outer Loss: {cached_data['outer_loss']:.4f}).")
            return cached_data
        except Exception:
            logger.warning(f"Corrupt cache for {setting_id}, re-running.")

    start_iso = datetime.datetime.now().isoformat()
    t0 = time.time()
    logger.info(f"[{model_name}] Starting Repeat {repeat} Fold {fold} ({n_trials} trials) at {start_iso} ...")

    X_outer_tr = X_train.iloc[outer_train_idx]
    y_outer_tr = y_train.iloc[outer_train_idx]
    X_outer_val = X_train.iloc[outer_val_idx]
    y_outer_val = y_train.iloc[outer_val_idx]

    # Inner 3-fold CV splits
    outer_id = f"r{repeat}_f{fold}"
    inner_splits = make_inner_splits(
        outer_train_idx=outer_train_idx,
        y=y_train.values,
        outer_id=outer_id,
        n_splits=3,
    )

    study_db_path = STUDIES_DIR / f"study_{setting_id}.db"
    storage_url = f"sqlite:///{study_db_path}"
    study_name = f"optuna_{setting_id}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )

    remaining_trials = max(0, n_trials - len(study.trials))
    if remaining_trials > 0:
        objective = create_inner_objective(
            model_name=model_name,
            inner_splits=inner_splits,
            X_outer_tr=X_outer_tr,
            y_outer_tr=y_outer_tr,
            outer_train_idx=outer_train_idx,
        )
        study.optimize(objective, n_trials=remaining_trials)

    best_trial = study.best_trial
    inner_best_loss = float(best_trial.value)
    best_params = best_trial.params.copy()
    best_imbalance = best_params.pop("imbalance")

    # Format params for pipeline reconstruction
    model_params = best_params.copy()
    if model_name == "M1":
        penalty = model_params.get("penalty", "l2")
        model_params["solver"] = "saga" if penalty == "l1" else "lbfgs"
        model_params["max_iter"] = 5000
        model_params["l1_ratio"] = 1.0 if penalty == "l1" else None
    elif model_name == "M5":
        subsample = model_params.get("subsample", 1.0)
        model_params["subsample_freq"] = 1 if subsample < 1.0 else 0

    # Refit best pipeline on entire outer training fold
    outer_pipe = build_pipeline(
        model_name=model_name,
        imbalance=best_imbalance,
        scaler="quantile",
        use_winsorizer=False,
        use_corr_filter=True,
        use_large_indicator=False,
        model_params=model_params,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outer_pipe.fit(X_outer_tr, y_outer_tr)
        outer_val_scores = outer_pipe.predict_proba(X_outer_val)[:, 1]

    outer_metrics = evaluate_scores(y_outer_val.values, outer_val_scores)
    outer_loss = compute_three_point_log_loss(y_outer_val.values, outer_val_scores)
    optimistic_bias = outer_loss - inner_best_loss

    # Score distribution diagnostics on outer validation fold
    y_val_arr = y_outer_val.values
    pos_scores = outer_val_scores[y_val_arr == 1]
    neg_scores = outer_val_scores[y_val_arr == 0]

    pos_median = float(np.median(pos_scores)) if len(pos_scores) > 0 else 0.0
    pos_p90 = float(np.percentile(pos_scores, 90)) if len(pos_scores) > 0 else 0.0
    pos_p99 = float(np.percentile(pos_scores, 99)) if len(pos_scores) > 0 else 0.0

    neg_median = float(np.median(neg_scores)) if len(neg_scores) > 0 else 0.0
    neg_p90 = float(np.percentile(neg_scores, 90)) if len(neg_scores) > 0 else 0.0
    neg_p99 = float(np.percentile(neg_scores, 99)) if len(neg_scores) > 0 else 0.0

    # Inner trial variation vs inner fold variation
    trial_losses = [t.value for t in study.trials if t.value is not None]
    trial_loss_std = float(np.std(trial_losses, ddof=1)) if len(trial_losses) > 1 else 0.0
    best_inner_fold_std = best_trial.user_attrs.get("fold_loss_std", 0.0)

    # Compare with matched default baseline on this fold
    matched_df = pd.read_csv(TABLES_DIR / "e2_default_matched_per_fold.csv")
    matched_row = matched_df[
        (matched_df["model"] == model_name) &
        (matched_df["repeat"] == repeat) &
        (matched_df["fold"] == fold)
    ].iloc[0]
    matched_roc = float(matched_row["roc_auc"])
    auc_jump = float(outer_metrics["roc_auc"] - matched_roc)

    # Single feature (EPS) AUC on this validation fold
    top_feat = "Persistent EPS in the Last Four Seasons"
    if top_feat in X_outer_val.columns:
        single_feat_auc = float(roc_auc_score(y_val_arr, -X_outer_val[top_feat].values))
    else:
        single_feat_auc = np.nan

    elapsed = time.time() - t0
    end_iso = datetime.datetime.now().isoformat()
    logger.info(
        f"[{model_name}] Finished r{repeat}f{fold} in {elapsed:.1f}s | "
        f"Inner Loss: {inner_best_loss:.4f} | Outer Loss: {outer_loss:.4f} | "
        f"Bias: {optimistic_bias:+.4f} | ROC-AUC: {outer_metrics['roc_auc']:.4f} (Matched: {matched_roc:.4f}, Jump: {auc_jump:+.4f}) | "
        f"PR-AUC: {outer_metrics['pr_auc']:.4f} | AR@95: {outer_metrics['alert_ratio@recall_0.95']:.2f}x"
    )

    # Stopping condition verification (User Rule):
    # Rule 2: Jump > 0.03 on the same fold triggers stop
    if auc_jump > 0.03:
        logger.error(
            f"STOPPING CONDITION TRIGGERED: ROC-AUC jump = {auc_jump:+.4f} > 0.03 on {setting_id} "
            f"(Tuned: {outer_metrics['roc_auc']:.4f}, Default: {matched_roc:.4f})"
        )
        raise AssertionError(f"Stopping condition: ROC-AUC jump = {auc_jump:+.4f} > 0.03 on {setting_id}")

    # Rule 3: Single fold ROC-AUC > 0.95 logs for reporting table, does NOT stop
    if outer_metrics["roc_auc"] > 0.95:
        logger.info(
            f"NOTE: Single fold ROC-AUC = {outer_metrics['roc_auc']:.4f} > 0.95 on {setting_id} "
            f"(Matched Default: {matched_roc:.4f}, Single Feat: {single_feat_auc:.4f}, Jump: {auc_jump:+.4f}). Continuing."
        )

    res = {
        "setting_id": setting_id,
        "model": model_name,
        "repeat": repeat,
        "fold": fold,
        "best_imbalance": best_imbalance,
        "best_params": best_params,
        "model_params": model_params,
        "inner_best_loss": inner_best_loss,
        "outer_loss": outer_loss,
        "optimistic_bias": optimistic_bias,
        "matched_roc": matched_roc,
        "auc_jump": auc_jump,
        "single_feat_auc": single_feat_auc,
        "outer_val_scores": outer_val_scores,
        "outer_val_idx": outer_val_idx,
        "outer_metrics": outer_metrics,
        "pos_median": pos_median,
        "pos_p90": pos_p90,
        "pos_p99": pos_p99,
        "neg_median": neg_median,
        "neg_p90": neg_p90,
        "neg_p99": neg_p99,
        "trial_loss_std": trial_loss_std,
        "best_inner_fold_std": best_inner_fold_std,
        "elapsed_seconds": elapsed,
        "start_iso": start_iso,
        "end_iso": end_iso,
    }

    joblib.dump(res, cache_file)
    return res


# ==============================================================================
# Untuned Matched Baseline Loader
# ==============================================================================
def get_untuned_baseline_results(
    models: list[str],
    selected_splits: list[dict],
) -> dict[str, pd.DataFrame]:
    """
    Loads strictly matched untuned baseline (S0, quantile, Winsorizer OFF, CorrFilter ON)
    from outputs/tables/e2_default_matched_per_fold.csv.
    """
    matched_file = TABLES_DIR / "e2_default_matched_per_fold.csv"
    if not matched_file.exists():
        raise FileNotFoundError(f"Missing {matched_file}. Run baseline generation first.")

    matched_df = pd.read_csv(matched_file)
    req_tuples = set((int(s["repeat"]), int(s["fold"])) for s in selected_splits)

    baseline_dfs = {}
    for m in models:
        m_sub = matched_df[
            (matched_df["model"] == m) &
            (matched_df.apply(lambda r: (int(r["repeat"]), int(r["fold"])) in req_tuples, axis=1))
        ].copy()
        baseline_dfs[m] = m_sub.sort_values(["repeat", "fold"]).reset_index(drop=True)

    return baseline_dfs


# ==============================================================================
# Main Orchestrator for Phase E2a
# ==============================================================================
def run_e2_tuning(outer_repeats: list[int], quick: bool = False):
    print("=" * 80)
    print(f"       E2: NESTED CV HYPERPARAMETER TUNING (Outer Repeats: {outer_repeats})")
    print("=" * 80)

    X_train, y_train = get_train_data()
    splits_payload = load_splits()
    all_splits = splits_payload["cv_splits"]
    train_orig_idx = splits_payload["train_idx"]
    orig_to_rel = {orig: i for i, orig in enumerate(train_orig_idx)}

    selected_splits = [s for s in all_splits if s["repeat"] in outer_repeats]
    if quick:
        selected_splits = selected_splits[:2]
    n_outer_folds = len(selected_splits)
    print(f"Dataset Size: {len(X_train)} training samples. Total Outer Folds to process: {n_outer_folds}")

    models = ["M1", "M2", "M3", "M4", "M5"]
    n_trials_map = {
        "M1": 3 if quick else 25,
        "M2": 3 if quick else 40,
        "M3": 3 if quick else 40,
        "M4": 3 if quick else 40,
        "M5": 3 if quick else 40,
    }

    # 1. Run Nested CV across models and folds
    model_fold_results: dict[str, list[dict[str, Any]]] = {m: [] for m in models}

    for m in models:
        print(f"\n>>> Starting Nested CV for Model {m} (Target Trials: {n_trials_map[m]}) <<<")
        for s in selected_splits:
            rep = s["repeat"]
            f = s["fold"]
            out_tr_idx = np.array([orig_to_rel[i] for i in s["train_idx"]])
            out_val_idx = np.array([orig_to_rel[i] for i in s["val_idx"]])

            res = run_single_outer_fold(
                model_name=m,
                repeat=rep,
                fold=f,
                outer_train_idx=out_tr_idx,
                outer_val_idx=out_val_idx,
                X_train=X_train,
                y_train=y_train,
                n_trials=n_trials_map[m],
            )
            model_fold_results[m].append(res)

    # 2. Compile Per-Fold Table
    per_fold_rows = []
    matched_df = pd.read_csv(TABLES_DIR / "e2_default_matched_per_fold.csv")

    for m in models:
        for r in model_fold_results[m]:
            if "pos_median" in r:
                pos_median = r["pos_median"]
                pos_p90 = r["pos_p90"]
                pos_p99 = r["pos_p99"]
                neg_median = r["neg_median"]
                neg_p90 = r["neg_p90"]
                neg_p99 = r["neg_p99"]
                matched_roc = r.get("matched_roc", np.nan)
                auc_jump = r.get("auc_jump", np.nan)
                single_feat_auc = r.get("single_feat_auc", np.nan)
            else:
                y_v = y_train.iloc[r["outer_val_idx"]].values
                sc_v = r["outer_val_scores"]
                pos_sc = sc_v[y_v == 1]
                neg_sc = sc_v[y_v == 0]
                pos_median = float(np.median(pos_sc)) if len(pos_sc) > 0 else 0.0
                pos_p90 = float(np.percentile(pos_sc, 90)) if len(pos_sc) > 0 else 0.0
                pos_p99 = float(np.percentile(pos_sc, 99)) if len(pos_sc) > 0 else 0.0
                neg_median = float(np.median(neg_sc)) if len(neg_sc) > 0 else 0.0
                neg_p90 = float(np.percentile(neg_sc, 90)) if len(neg_sc) > 0 else 0.0
                neg_p99 = float(np.percentile(neg_sc, 99)) if len(neg_sc) > 0 else 0.0

                m_row = matched_df[
                    (matched_df["model"] == m) &
                    (matched_df["repeat"] == r["repeat"]) &
                    (matched_df["fold"] == r["fold"])
                ].iloc[0]
                matched_roc = float(m_row["roc_auc"])
                auc_jump = float(r["outer_metrics"]["roc_auc"] - matched_roc)

                X_v = X_train.iloc[r["outer_val_idx"]]
                top_feat = "Persistent EPS in the Last Four Seasons"
                single_feat_auc = float(roc_auc_score(y_v, -X_v[top_feat].values)) if top_feat in X_v.columns else np.nan

            row = {
                "model": m,
                "repeat": r["repeat"],
                "fold": r["fold"],
                "best_imbalance": r["best_imbalance"],
                "best_params": json.dumps(r["best_params"]),
                "inner_best_loss": r["inner_best_loss"],
                "outer_loss": r["outer_loss"],
                "optimistic_bias": r["optimistic_bias"],
                "matched_roc": matched_roc,
                "auc_jump": auc_jump,
                "single_feat_auc": single_feat_auc,
                "pos_median": pos_median,
                "pos_p90": pos_p90,
                "pos_p99": pos_p99,
                "neg_median": neg_median,
                "neg_p90": neg_p90,
                "neg_p99": neg_p99,
                "trial_loss_std": r["trial_loss_std"],
                "best_inner_fold_std": r["best_inner_fold_std"],
                "elapsed_seconds": r["elapsed_seconds"],
                "start_time": r["start_iso"],
                "end_time": r["end_iso"],
            }
            for k, v in r["outer_metrics"].items():
                row[k] = v
            per_fold_rows.append(row)

    per_fold_df = pd.DataFrame(per_fold_rows).sort_values(["model", "repeat", "fold"]).reset_index(drop=True)

    # Multi-phase merge with existing table
    per_fold_file = TABLES_DIR / "e2a_nested_per_fold.csv"
    if per_fold_file.exists():
        try:
            old_df = pd.read_csv(per_fold_file)
            merged_df = pd.concat([old_df, per_fold_df], ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=["model", "repeat", "fold"], keep="last")
            per_fold_df = merged_df.sort_values(["model", "repeat", "fold"]).reset_index(drop=True)
        except Exception:
            pass

    per_fold_df.to_csv(per_fold_file, index=False)
    print(f"\nSaved {per_fold_file} (Total records: {len(per_fold_df)})")

    # Stopping condition: Check if any model outer fold mean ROC-AUC > 0.95
    for m in models:
        m_mean_roc = per_fold_df[per_fold_df["model"] == m]["roc_auc"].mean()
        if m_mean_roc > 0.95:
            logger.error(f"STOPPING CONDITION TRIGGERED: Mean outer ROC-AUC = {m_mean_roc:.4f} > 0.95 on {m}")
            raise AssertionError(f"Stopping condition: Mean outer ROC-AUC = {m_mean_roc:.4f} > 0.95 on {m}")

    # 3. Compile Summary Table (Fold Mean, Pooled OOF, and Percentile Rank Pooled OOF per repeat)
    summary_rows = []
    oof_predictions_dict = {}
    oof_pct_ranks_dict = {}

    y_train_arr = y_train.values
    n_reps = len(outer_repeats)
    rep_to_idx = {rep: i for i, rep in enumerate(outer_repeats)}

    for m in models:
        m_folds = per_fold_df[per_fold_df["model"] == m].copy()
        k_folds = len(m_folds)

        m_oof_scores = np.full((n_reps, len(X_train)), np.nan, dtype=float)
        m_pct_ranks = np.full((n_reps, len(X_train)), np.nan, dtype=float)

        for _, r in m_folds.iterrows():
            rep_i = int(r["repeat"])
            fold_i = int(r["fold"])
            c_file = CACHE_DIR / f"{m}_r{rep_i}_f{fold_i}.joblib"
            c_data = joblib.load(c_file)
            scores_f = c_data["outer_val_scores"]
            val_idx_f = c_data["outer_val_idx"]

            if rep_i in rep_to_idx:
                r_idx = rep_to_idx[rep_i]
                m_oof_scores[r_idx, val_idx_f] = scores_f
                ranks_f = stats.rankdata(scores_f, method="average") / len(scores_f)
                m_pct_ranks[r_idx, val_idx_f] = ranks_f

        oof_predictions_dict[m] = m_oof_scores
        oof_pct_ranks_dict[m] = m_pct_ranks
        np.save(OOF_DIR / f"e2_{m}.npy", m_oof_scores)

        # Compute Pooled OOF metrics per repeat and average across repeats
        rep_p_losses, rep_p_ar95, rep_p_ar90, rep_p_ar80, rep_p_pr, rep_p_roc, rep_p_rec3 = [], [], [], [], [], [], []
        rep_pct_losses, rep_pct_ar95, rep_pct_ar90, rep_pct_ar80, rep_pct_pr, rep_pct_roc, rep_pct_rec3 = [], [], [], [], [], [], []

        for r_idx in range(n_reps):
            sc_r = m_oof_scores[r_idx]
            rk_r = m_pct_ranks[r_idx]
            mask_r = ~np.isnan(sc_r)
            if not np.any(mask_r):
                continue
            y_r = y_train_arr[mask_r]
            sc_valid = sc_r[mask_r]
            rk_valid = rk_r[mask_r]

            ar80 = alert_ratio_at_recall(y_r, sc_valid, 0.80)
            ar90 = alert_ratio_at_recall(y_r, sc_valid, 0.90)
            ar95 = alert_ratio_at_recall(y_r, sc_valid, 0.95)
            rep_p_ar80.append(ar80)
            rep_p_ar90.append(ar90)
            rep_p_ar95.append(ar95)
            rep_p_losses.append(float(np.mean([np.log(ar80), np.log(ar90), np.log(ar95)])))
            rep_p_pr.append(float(average_precision_score(y_r, sc_valid)))
            rep_p_roc.append(float(roc_auc_score(y_r, sc_valid)))
            rep_p_rec3.append(float(recall_at_alert_ratio(y_r, sc_valid, 3.0)))

            pct_ar80 = alert_ratio_at_recall(y_r, rk_valid, 0.80)
            pct_ar90 = alert_ratio_at_recall(y_r, rk_valid, 0.90)
            pct_ar95 = alert_ratio_at_recall(y_r, rk_valid, 0.95)
            rep_pct_ar80.append(pct_ar80)
            rep_pct_ar90.append(pct_ar90)
            rep_pct_ar95.append(pct_ar95)
            rep_pct_losses.append(float(np.mean([np.log(pct_ar80), np.log(pct_ar90), np.log(pct_ar95)])))
            rep_pct_pr.append(float(average_precision_score(y_r, rk_valid)))
            rep_pct_roc.append(float(roc_auc_score(y_r, rk_valid)))
            rep_pct_rec3.append(float(recall_at_alert_ratio(y_r, rk_valid, 3.0)))

        summary_rows.append({
            "model": m,
            "n_outer_folds": k_folds,
            "three_point_loss_fold_mean": float(m_folds["outer_loss"].mean()),
            "three_point_loss_fold_std": float(m_folds["outer_loss"].std()),
            "three_point_loss_pooled_oof": float(np.mean(rep_p_losses)) if rep_p_losses else np.nan,
            "three_point_loss_pct_rank_oof": float(np.mean(rep_pct_losses)) if rep_pct_losses else np.nan,
            "ar95_fold_mean": float(m_folds["alert_ratio@recall_0.95"].mean()),
            "ar95_fold_std": float(m_folds["alert_ratio@recall_0.95"].std()),
            "ar95_pooled_oof": float(np.mean(rep_p_ar95)) if rep_p_ar95 else np.nan,
            "ar95_pct_rank_oof": float(np.mean(rep_pct_ar95)) if rep_pct_ar95 else np.nan,
            "ar90_fold_mean": float(m_folds["alert_ratio@recall_0.90"].mean()),
            "ar90_fold_std": float(m_folds["alert_ratio@recall_0.90"].std()),
            "ar90_pooled_oof": float(np.mean(rep_p_ar90)) if rep_p_ar90 else np.nan,
            "ar90_pct_rank_oof": float(np.mean(rep_pct_ar90)) if rep_pct_ar90 else np.nan,
            "ar80_fold_mean": float(m_folds["alert_ratio@recall_0.80"].mean()),
            "ar80_fold_std": float(m_folds["alert_ratio@recall_0.80"].std()),
            "ar80_pooled_oof": float(np.mean(rep_p_ar80)) if rep_p_ar80 else np.nan,
            "ar80_pct_rank_oof": float(np.mean(rep_pct_ar80)) if rep_pct_ar80 else np.nan,
            "pr_auc_fold_mean": float(m_folds["pr_auc"].mean()),
            "pr_auc_fold_std": float(m_folds["pr_auc"].std()),
            "pr_auc_pooled_oof": float(np.mean(rep_p_pr)) if rep_p_pr else np.nan,
            "pr_auc_pct_rank_oof": float(np.mean(rep_pct_pr)) if rep_pct_pr else np.nan,
            "roc_auc_fold_mean": float(m_folds["roc_auc"].mean()),
            "roc_auc_fold_std": float(m_folds["roc_auc"].std()),
            "roc_auc_pooled_oof": float(np.mean(rep_p_roc)) if rep_p_roc else np.nan,
            "roc_auc_pct_rank_oof": float(np.mean(rep_pct_roc)) if rep_pct_roc else np.nan,
            "rec@ar3_fold_mean": float(m_folds["recall@alert_ratio_3"].mean()),
            "rec@ar3_fold_std": float(m_folds["recall@alert_ratio_3"].std()),
            "rec@ar3_pooled_oof": float(np.mean(rep_p_rec3)) if rep_p_rec3 else np.nan,
            "rec@ar3_pct_rank_oof": float(np.mean(rep_pct_rec3)) if rep_pct_rec3 else np.nan,
            "inner_best_loss_mean": float(m_folds["inner_best_loss"].mean()),
            "optimistic_bias_mean": float(m_folds["optimistic_bias"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = TABLES_DIR / "e2a_nested_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved {summary_file}")

    # 4. Parameter Stability Analysis
    stability_rows = []
    for m in models:
        m_folds = per_fold_df[per_fold_df["model"] == m]
        s0_count = int((m_folds["best_imbalance"] == "S0").sum())
        s3_count = int((m_folds["best_imbalance"] == "S3").sum())

        row = {
            "model": m,
            "n_folds": len(m_folds),
            "imbalance_S0_count": s0_count,
            "imbalance_S3_count": s3_count,
            "imbalance_S0_pct": s0_count / len(m_folds) * 100,
            "imbalance_S3_pct": s3_count / len(m_folds) * 100,
            "inner_best_loss_mean": m_folds["inner_best_loss"].mean(),
            "outer_actual_loss_mean": m_folds["outer_loss"].mean(),
            "optimistic_bias_mean": m_folds["optimistic_bias"].mean(),
            "optimistic_bias_std": m_folds["optimistic_bias"].std(),
            "mean_inner_fold_noise_std": m_folds["best_inner_fold_std"].mean(),
            "mean_trial_diff_std": m_folds["trial_loss_std"].mean(),
        }
        stability_rows.append(row)

    stability_df = pd.DataFrame(stability_rows)
    stability_file = TABLES_DIR / "e2a_best_param_stability.csv"
    stability_df.to_csv(stability_file, index=False)
    print(f"Saved {stability_file}")

    # 5. Paired Comparisons vs Strictly Matched Default Baseline
    baseline_dfs = get_untuned_baseline_results(
        models=models,
        selected_splits=selected_splits,
    )

    paired_rows = []
    for m in models:
        m_tuned = per_fold_df[per_fold_df["model"] == m].sort_values(["repeat", "fold"]).reset_index(drop=True)
        m_base = baseline_dfs[m].sort_values(["repeat", "fold"]).reset_index(drop=True)

        k_comp = len(m_tuned)
        diff_loss = m_tuned["outer_loss"].values - m_base["three_point_loss"].values
        diff_ar95 = m_tuned["alert_ratio@recall_0.95"].values - m_base["alert_ratio@recall_0.95"].values
        diff_pr = m_tuned["pr_auc"].values - m_base["pr_auc"].values
        diff_rec3 = m_tuned["recall@alert_ratio_3"].values - m_base["recall@alert_ratio_3"].values

        m_loss, ci_l_loss, ci_u_loss, p_nb_loss = nadeau_bengio_paired_test(
            diff_loss, n_splits=CV_FOLDS, n_repeats=len(outer_repeats)
        )
        p_wilc_loss = wilcoxon_test(diff_loss)

        m_ar95, ci_l_ar95, ci_u_ar95, p_nb_ar95 = nadeau_bengio_paired_test(
            diff_ar95, n_splits=CV_FOLDS, n_repeats=len(outer_repeats)
        )
        m_pr, ci_l_pr, ci_u_pr, p_nb_pr = nadeau_bengio_paired_test(
            diff_pr, n_splits=CV_FOLDS, n_repeats=len(outer_repeats)
        )
        m_rec3, ci_l_rec3, ci_u_rec3, p_nb_rec3 = nadeau_bengio_paired_test(
            diff_rec3, n_splits=CV_FOLDS, n_repeats=len(outer_repeats)
        )

        paired_rows.append({
            "model": m,
            "n_outer_folds": k_comp,
            "tuned_loss_mean": m_tuned["outer_loss"].mean(),
            "default_loss_mean": m_base["three_point_loss"].mean(),
            "diff_loss_mean": m_loss,
            "diff_loss_ci_lower": ci_l_loss,
            "diff_loss_ci_upper": ci_u_loss,
            "diff_loss_p_nb": p_nb_loss,
            "diff_loss_p_wilcoxon": p_wilc_loss,
            "diff_ar95_mean": m_ar95,
            "diff_ar95_ci_lower": ci_l_ar95,
            "diff_ar95_ci_upper": ci_u_ar95,
            "diff_ar95_p_nb": p_nb_ar95,
            "diff_pr_auc_mean": m_pr,
            "diff_pr_auc_ci_lower": ci_l_pr,
            "diff_pr_auc_ci_upper": ci_u_pr,
            "diff_pr_auc_p_nb": p_nb_pr,
            "diff_rec@ar3_mean": m_rec3,
            "diff_rec@ar3_ci_lower": ci_l_rec3,
            "diff_rec@ar3_ci_upper": ci_u_rec3,
            "diff_rec@ar3_p_nb": p_nb_rec3,
        })

    raw_p_primary = [r["diff_loss_p_nb"] for r in paired_rows]
    holm_p = holm_bonferroni(raw_p_primary)
    for r, hp in zip(paired_rows, holm_p):
        r["diff_loss_p_holm"] = hp
        r["statistically_better"] = bool(r["diff_loss_ci_upper"] < 0 and hp < 0.05)
        r["statistically_worse"] = bool(r["diff_loss_ci_lower"] > 0 and hp < 0.05)

    paired_df = pd.DataFrame(paired_rows)
    paired_file = TABLES_DIR / "e2a_paired_vs_default.csv"
    paired_df.to_csv(paired_file, index=False)
    print(f"Saved {paired_file}")

    # 6. Generate Figures
    generate_e2_figures(
        per_fold_df=per_fold_df,
        baseline_dfs=baseline_dfs,
        oof_predictions_dict=oof_predictions_dict,
        y_train=y_train,
        models=models,
        outer_repeats=outer_repeats,
    )

    print("\nPhase E2a completed successfully.")
    return per_fold_df, summary_df, stability_df, paired_df


# ==============================================================================
# Phase E2b: Final Hyperparameter Tuning on Full Training Set
# ==============================================================================
def run_e2b_final_tuning(quick: bool = False):
    print("\n" + "=" * 80)
    print("       E2b: FINAL HYPERPARAMETER TUNING ON FULL TRAINING SET (FOR E6)")
    print("=" * 80)

    X_train, y_train = get_train_data()
    y_train_arr = y_train.values
    n_samples = len(X_train)
    print(f"Full Training Dataset: {n_samples} samples, {np.sum(y_train_arr == 1)} bankrupt firms.")

    # Repeated Stratified 5-Fold x 2 Repeats (10 inner folds)
    skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)

    splits_r0 = list(skf1.split(np.zeros(n_samples), y_train_arr))
    splits_r1 = list(skf2.split(np.zeros(n_samples), y_train_arr))
    all_10_splits = splits_r0 + splits_r1

    models = ["M1", "M2", "M3", "M4", "M5"]
    n_trials_map = {
        "M1": 5 if quick else 38,
        "M2": 5 if quick else 60,
        "M3": 5 if quick else 60,
        "M4": 5 if quick else 60,
        "M5": 5 if quick else 60,
    }

    final_params_summary = []

    for m in models:
        n_trials = n_trials_map[m]
        study_db = STUDIES_DIR / f"study_e2b_{m}.db"
        storage_url = f"sqlite:///{study_db}"
        study_name = f"optuna_e2b_{m}"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            load_if_exists=True,
        )

        remaining_trials = max(0, n_trials - len(study.trials))
        logger.info(f"[E2b - {m}] Running {remaining_trials} trials (target: {n_trials}) with 10 inner folds ...")

        if remaining_trials > 0:
            def e2b_objective(trial: optuna.Trial) -> float:
                imbalance, params = sample_hyperparameters(trial, m)

                def factory():
                    return build_pipeline(
                        model_name=m,
                        imbalance=imbalance,
                        scaler="quantile",
                        use_winsorizer=False,
                        use_corr_filter=True,
                        use_large_indicator=False,
                        model_params=params,
                    )

                scores_r0 = np.zeros(n_samples, dtype=float)
                scores_r1 = np.zeros(n_samples, dtype=float)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for tr_i, val_i in splits_r0:
                        pipe = factory()
                        pipe.fit(X_train.iloc[tr_i], y_train.iloc[tr_i])
                        scores_r0[val_i] = pipe.predict_proba(X_train.iloc[val_i])[:, 1]

                    for tr_i, val_i in splits_r1:
                        pipe = factory()
                        pipe.fit(X_train.iloc[tr_i], y_train.iloc[tr_i])
                        scores_r1[val_i] = pipe.predict_proba(X_train.iloc[val_i])[:, 1]

                loss_r0 = compute_three_point_log_loss(y_train_arr, scores_r0)
                loss_r1 = compute_three_point_log_loss(y_train_arr, scores_r1)
                mean_loss = float((loss_r0 + loss_r1) / 2.0)
                return mean_loss

            study.optimize(e2b_objective, n_trials=remaining_trials)

        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_imbalance = best_params.pop("imbalance")
        inner_best_loss = float(best_trial.value)

        # Save model JSON
        best_json_path = BEST_PARAMS_DIR / f"{m}.json"
        payload = {
            "model": m,
            "imbalance": best_imbalance,
            "best_params": best_params,
            "inner_best_loss": inner_best_loss,
            "n_trials": len(study.trials),
            "study_name": study_name,
            "created_at": datetime.datetime.now().isoformat(),
        }
        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"[E2b - {m}] Saved final params to {best_json_path}")

        # Top 5 trials landscape flatness analysis
        valid_trials = [t for t in study.trials if t.value is not None]
        sorted_trials = sorted(valid_trials, key=lambda t: t.value)
        top5_values = [float(t.value) for t in sorted_trials[:5]]
        all_values = [float(t.value) for t in valid_trials]

        top5_diff = top5_values[-1] - top5_values[0] if len(top5_values) > 1 else 0.0
        trial_loss_std = float(np.std(all_values, ddof=1)) if len(all_values) > 1 else 0.0

        summary_row = {
            "model": m,
            "best_imbalance": best_imbalance,
            "best_params": json.dumps(best_params),
            "inner_best_loss": inner_best_loss,
            "trial_1_loss": top5_values[0] if len(top5_values) > 0 else np.nan,
            "trial_2_loss": top5_values[1] if len(top5_values) > 1 else np.nan,
            "trial_3_loss": top5_values[2] if len(top5_values) > 2 else np.nan,
            "trial_4_loss": top5_values[3] if len(top5_values) > 3 else np.nan,
            "trial_5_loss": top5_values[4] if len(top5_values) > 4 else np.nan,
            "top5_range": top5_diff,
            "trial_loss_std": trial_loss_std,
        }
        final_params_summary.append(summary_row)

    e2b_summary_df = pd.DataFrame(final_params_summary)
    e2b_summary_path = TABLES_DIR / "e2b_final_params_summary.csv"
    e2b_summary_df.to_csv(e2b_summary_path, index=False)
    print(f"Saved {e2b_summary_path}")

    print("\nPhase E2b completed successfully.")
    return e2b_summary_df


# ==============================================================================
# Visualization Functions
# ==============================================================================
def generate_e2_figures(
    per_fold_df: pd.DataFrame,
    baseline_dfs: dict[str, pd.DataFrame],
    oof_predictions_dict: dict[str, np.ndarray],
    y_train: pd.Series,
    models: list[str],
    outer_repeats: list[int],
):
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # Figure 1: e2_tuned_vs_default.png
    plot_rows = []
    for m in models:
        m_tuned = per_fold_df[per_fold_df["model"] == m]
        m_base = baseline_dfs[m]
        for _, r in m_tuned.iterrows():
            plot_rows.append({
                "model": m,
                "variant": "Tuned (E2a)",
                "loss": float(r["outer_loss"]),
            })
        for _, r in m_base.iterrows():
            plot_rows.append({
                "model": m,
                "variant": "Matched Default (Untuned)",
                "loss": float(r["three_point_loss"]),
            })

    plot_df = pd.DataFrame(plot_rows)

    plt.figure(figsize=(10, 5.5))
    ax = sns.boxplot(
        data=plot_df,
        x="model",
        y="loss",
        hue="variant",
        palette={"Matched Default (Untuned)": "#95a5a6", "Tuned (E2a)": "#2ecc71"},
        width=0.55,
        fliersize=3,
    )
    sns.stripplot(
        data=plot_df,
        x="model",
        y="loss",
        hue="variant",
        dodge=True,
        palette={"Matched Default (Untuned)": "#34495e", "Tuned (E2a)": "#27ae60"},
        alpha=0.6,
        size=5,
        legend=False,
    )

    plt.title(f"E2: Three-Point Log Alert Ratio Loss (Tuned vs Matched Default, Repeats {outer_repeats})", fontsize=12, pad=12)
    plt.xlabel("Model", fontsize=11)
    plt.ylabel("Loss: Mean(log(AR@80, AR@90, AR@95)) [Lower is Better]", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Configuration", loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e2_tuned_vs_default.png", dpi=300)
    plt.close()
    print(f"Saved figure: {FIGURES_DIR / 'e2_tuned_vs_default.png'}")

    # Figure 2: e2_recall_vs_alert_ratio_tuned.png
    plt.figure(figsize=(9, 6))
    alert_ratios_eval = np.linspace(1.0, 30.0, 60)
    colors = {
        "M1": "#e74c3c",
        "M2": "#2980b9",
        "M3": "#16a085",
        "M4": "#f39c12",
        "M5": "#8e44ad",
    }

    y_arr = y_train.values

    for m in models:
        sc_all = oof_predictions_dict[m]
        recalls_reps = []
        for r_i in range(len(sc_all)):
            sc_r = sc_all[r_i] if sc_all.ndim == 2 else sc_all
            mask_r = ~np.isnan(sc_r)
            if not np.any(mask_r):
                continue
            y_sub = y_arr[mask_r]
            sc_sub = sc_r[mask_r]
            rec_r = [recall_at_alert_ratio(y_sub, sc_sub, r_val) for r_val in alert_ratios_eval]
            recalls_reps.append(rec_r)
            if sc_all.ndim == 1:
                break

        if len(recalls_reps) > 0:
            recalls = np.mean(recalls_reps, axis=0)
            plt.plot(
                alert_ratios_eval,
                recalls,
                label=f"{m} Tuned (Pooled OOF)",
                color=colors.get(m, "black"),
                linewidth=2.0,
            )

    plt.axhline(0.95, color="red", linestyle=":", label="Target Recall = 0.95")
    plt.axhline(0.90, color="gray", linestyle="--", alpha=0.7, label="Target Recall = 0.90")
    plt.title(f"E2: Recall vs Alert Ratio Curves (Tuned Models, Repeats {outer_repeats})", fontsize=12, pad=12)
    plt.xlabel("Alert Ratio (Alerted Firms / Actual Bankruptcies)", fontsize=11)
    plt.ylabel("Recall (True Positive Rate)", fontsize=11)
    plt.xlim(1.0, 30.0)
    plt.ylim(0.4, 1.02)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e2_recall_vs_alert_ratio_tuned.png", dpi=300)
    plt.close()
    print(f"Saved figure: {FIGURES_DIR / 'e2_recall_vs_alert_ratio_tuned.png'}")


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="E2: Nested CV Tuning & E2b Module")
    parser.add_argument(
        "--outer-repeats",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="List of outer repeats to execute (default: [0, 1, 2] for all 15 folds)",
    )
    parser.add_argument(
        "--run-e2b",
        action="store_true",
        help="Execute E2b final tuning on full training set after E2a",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: run 2 outer folds and minimal trials for sanity check",
    )
    args = parser.parse_args()

    run_e2_tuning(outer_repeats=args.outer_repeats, quick=args.quick)

    if args.run_e2b:
        run_e2b_final_tuning(quick=args.quick)


if __name__ == "__main__":
    main()
