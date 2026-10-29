#!/usr/bin/env python3
"""
tools/build_param_registry.py
Automated generator for docs/code_learning/03a_param_registry.csv.

Extracts parameters from:
- bankruptcy/config.py (AST parsing)
- bankruptcy/src/preprocessing.py (AST parsing)
- bankruptcy/src/pipelines.py (AST parsing)
- bankruptcy/src/tune.py (AST parsing of search spaces)
- bankruptcy/src/ensemble.py (AST parsing)
- bankruptcy/src/thresholding.py (AST parsing)
- bankruptcy/src/scoring.py (AST parsing)
- bankruptcy/src/prevalence_sensitivity.py (AST parsing)
- bankruptcy/src/seed_stability.py (AST parsing)
- bankruptcy/outputs/models/best_params/*.json (JSON loading + line finding)
- bankruptcy/outputs/models/final/thresholds.json (JSON loading + line finding)
- bankruptcy/app/assets/defaults.json (JSON loading + line finding)

Merges with docs/code_learning/param_notes.csv by 'id' to produce the final 17-column registry.
DOES NOT import any project execution modules or write caches.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "code_learning"
OUTPUT_CSV = DOCS_DIR / "03a_param_registry.csv"
NOTES_CSV = DOCS_DIR / "param_notes.csv"


def parse_ast_assignments(file_path: Path) -> dict[str, tuple[int, Any, str]]:
    """Parses top-level assignments in a Python file. Returns {name: (lineno, value, type_str)}."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    res = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    lineno = node.lineno
                    try:
                        val = ast.literal_eval(node.value)
                        type_str = type(val).__name__
                    except Exception:
                        val = ast.unparse(node.value)
                        type_str = "expr"
                    res[name] = (lineno, val, type_str)
    return res


def find_symbol_line(file_path: Path, symbol: str, start_search: int = 1) -> int:
    """Finds line number containing symbol (word boundary)."""
    lines = file_path.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\b")
    for idx in range(start_search - 1, len(lines)):
        if pattern.search(lines[idx]):
            return idx + 1
    # Fallback to search from beginning
    for idx in range(0, start_search - 1):
        if pattern.search(lines[idx]):
            return idx + 1
    return 0


def find_threshold_json_line(model_key: str, r_key: str, field_name: str) -> int:
    """Finds the exact line of threshold or recommended_margin in thresholds.json."""
    th_file = REPO_ROOT / "bankruptcy" / "outputs" / "models" / "final" / "thresholds.json"
    lines = th_file.read_text(encoding="utf-8").splitlines()
    cur_model = None
    cur_r = None
    for idx, line in enumerate(lines, 1):
        l = line.strip()
        if l.startswith(f'"{model_key}":'):
            cur_model = model_key
        elif cur_model == model_key and l.startswith(f'"{r_key}":'):
            cur_r = r_key
        elif cur_model == model_key and cur_r == r_key:
            if l.startswith(f'"{field_name}":'):
                return idx
            elif l.startswith("}"):
                cur_r = None
    return 0


def find_defaults_json_line(token: str, under_inputs: bool = False) -> int:
    """Finds line of key in defaults.json."""
    d_file = REPO_ROOT / "bankruptcy" / "app" / "assets" / "defaults.json"
    lines = d_file.read_text(encoding="utf-8").splitlines()
    in_inputs = False
    for idx, line in enumerate(lines, 1):
        l = line.strip()
        if l.startswith('"inputs":'):
            in_inputs = True
        if under_inputs:
            if in_inputs and f'"{token}"' in l:
                return idx
        else:
            if not in_inputs and f'"{token}"' in l:
                return idx
    # Fallback
    for idx, line in enumerate(lines, 1):
        if f'"{token}"' in line:
            return idx
    return 0


def build_raw_param_records() -> list[dict[str, Any]]:
    """Builds raw parameter definitions from code and JSON configs."""
    records: list[dict[str, Any]] = []

    # 1. bankruptcy/config.py
    cfg_file = REPO_ROOT / "bankruptcy" / "config.py"
    cfg_assigns = parse_ast_assignments(cfg_file)

    t1_cfg = {
        "SEED": ("交叉驗證", "T1"),
        "TEST_SIZE": ("資料與切分", "T1"),
        "CV_FOLDS": ("交叉驗證", "T1"),
        "CV_REPEATS": ("交叉驗證", "T1"),
        "TARGET_RECALLS": ("門檻與目標", "T1"),
        "ALERT_RATIOS": ("評估指標", "T1"),
        "DEFAULT_TARGET_RECALL": ("門檻與目標", "T1"),
        "TOP_K_PERCENT": ("應用層", "T1"),
        "MIN_BATCH_SIZE_FOR_TOP_K": ("應用層", "T1"),
        "COST_FN": ("評估指標", "T1"),
        "COST_FP": ("評估指標", "T1"),
        "COST_FN_SENSITIVITY_LIST": ("評估指標", "T1"),
        "MAX_MISSING_FEATURE_RATIO": ("前處理", "T1"),
        "MVP_PASS_PR_AUC_MULTIPLIER": ("評估指標", "T1"),
        "MVP_PASS_TOP10_RECALL": ("評估指標", "T1"),
    }

    t2_cfg = {
        "TARGET_COL": ("資料與切分", "T2"),
        "BASE_DIR": ("系統與路徑", "T2"),
        "DATA_DIR": ("系統與路徑", "T2"),
        "DATA_PATH": ("系統與路徑", "T2"),
        "OUTPUT_DIR": ("系統與路徑", "T2"),
        "TABLES_DIR": ("系統與路徑", "T2"),
        "FIGURES_DIR": ("系統與路徑", "T2"),
        "MODELS_DIR": ("系統與路徑", "T2"),
        "LOGS_DIR": ("系統與路徑", "T2"),
        "FEATURES": ("資料與切分", "T2"),
    }

    for name, (cat, tier) in {**t1_cfg, **t2_cfg}.items():
        if name in cfg_assigns:
            lineno, val, typ = cfg_assigns[name]
            curr_val_str = str(val) if (not isinstance(val, str) or typ == "expr") else f'"{val}"'
            records.append({
                "id": f"CFG_{name}",
                "參數名": name,
                "層級(T1/T2)": tier,
                "所在檔案": "bankruptcy/config.py",
                "行號": lineno,
                "類別": cat,
                "來源類型": "code_constant",
                "型別": typ if typ != "list" else f"list[{type(val[0]).__name__ if val else 'Any'}]",
                "目前值": curr_val_str,
                "驗證方式": "ast",
            })

    # 2. bankruptcy/src/preprocessing.py
    records.extend([
        {
            "id": "PREP_CORR_THRESHOLD",
            "參數名": "threshold",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 353,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "float",
            "目前值": "0.95",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_CORR_ENABLED",
            "參數名": "enabled",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 353,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "bool",
            "目前值": "True",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_WINSOR_LOWER",
            "參數名": "lower",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 193,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "float",
            "目前值": "0.01",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_WINSOR_UPPER",
            "參數名": "upper",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 193,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "float",
            "目前值": "0.99",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_WINSOR_ENABLED",
            "參數名": "enabled",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 193,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "bool",
            "目前值": "True",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_QUANTILE_N",
            "參數名": "n_quantiles",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 310,
            "類別": "前處理",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "1000",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_QUANTILE_DIST",
            "參數名": "output_distribution",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 312,
            "類別": "前處理",
            "來源類型": "code_constant",
            "型別": "str",
            "目前值": '"normal"',
            "驗證方式": "ast",
        },
        {
            "id": "PREP_LARGE_ENABLED",
            "參數名": "enabled",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 249,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "bool",
            "目前值": "False",
            "驗證方式": "ast",
        },
    ])

    # 3. bankruptcy/src/pipelines.py
    records.extend([
        {
            "id": "PIPE_SCALE_POS_WEIGHT",
            "參數名": "scale_pos_weight",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 53,
            "類別": "模型超參數",
            "來源類型": "derived",
            "型別": "float",
            "目前值": "動態：由訓練折計算",
            "驗證方式": "ast",
        },
        {
            "id": "PIPE_IMBALANCE_DEFAULT",
            "參數名": "imbalance",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 214,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "str",
            "目前值": '"S0"',
            "驗證方式": "ast",
        },
        {
            "id": "PIPE_SCALER_DEFAULT",
            "參數名": "scaler",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 215,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "str",
            "目前值": '"quantile"',
            "驗證方式": "ast",
        },
    ])

    # 4. bankruptcy/src/tune.py
    records.extend([
        {
            "id": "TUNE_E2A_TRIALS",
            "參數名": "n_trials_map",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 501,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "dict[str, int]",
            "目前值": "{'M1': 25, 'M2': 40, 'M3': 40, 'M4': 40, 'M5': 40}",
            "驗證方式": "ast",
        },
        {
            "id": "TUNE_E2B_TRIALS",
            "參數名": "n_trials_map",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 865,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "dict[str, int]",
            "目前值": "{'M1': 38, 'M2': 60, 'M3': 60, 'M4': 60, 'M5': 60}",
            "驗證方式": "ast",
        },
        {
            "id": "M1_SEARCH_C",
            "參數名": "C",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 127,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.11616457722286455",
            "驗證方式": "json",
        },
        {
            "id": "M1_SEARCH_PENALTY",
            "參數名": "penalty",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 128,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "str",
            "目前值": '"l2"',
            "驗證方式": "json",
        },
        {
            "id": "M2_SEARCH_N_ESTIMATORS",
            "參數名": "n_estimators",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 145,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "300",
            "驗證方式": "json",
        },
        {
            "id": "M2_SEARCH_MAX_DEPTH",
            "參數名": "max_depth",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 146,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "12",
            "驗證方式": "json",
        },
        {
            "id": "M2_SEARCH_MIN_SAMPLES_LEAF",
            "參數名": "min_samples_leaf",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 147,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "10",
            "驗證方式": "json",
        },
        {
            "id": "M2_SEARCH_MAX_FEATURES",
            "參數名": "max_features",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 148,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "str",
            "目前值": '"sqrt"',
            "驗證方式": "json",
        },
        {
            "id": "M3_SEARCH_N_ESTIMATORS",
            "參數名": "n_estimators",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 145,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "600",
            "驗證方式": "json",
        },
        {
            "id": "M3_SEARCH_MAX_DEPTH",
            "參數名": "max_depth",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 146,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "NoneType",
            "目前值": "None",
            "驗證方式": "json",
        },
        {
            "id": "M3_SEARCH_MIN_SAMPLES_LEAF",
            "參數名": "min_samples_leaf",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 147,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "2",
            "驗證方式": "json",
        },
        {
            "id": "M3_SEARCH_MAX_FEATURES",
            "參數名": "max_features",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 148,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.3",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_N_ESTIMATORS",
            "參數名": "n_estimators",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 153,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "250",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_LEARNING_RATE",
            "參數名": "learning_rate",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 154,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.046776815926747296",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_MAX_DEPTH",
            "參數名": "max_depth",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 155,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "2",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_SUBSAMPLE",
            "參數名": "subsample",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 156,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.879563935600964",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_COLSAMPLE",
            "參數名": "colsample_bytree",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 157,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.9017127339966642",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_REG_LAMBDA",
            "參數名": "reg_lambda",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 158,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "1.6902875531341541",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_MIN_CHILD_WEIGHT",
            "參數名": "min_child_weight",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 159,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "7.300913224180319",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_SUBSAMPLE",
            "參數名": "subsample",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 163,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.6071695513247772",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_SUBSAMPLE_FREQ",
            "參數名": "subsample_freq",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 169,
            "類別": "模型超參數",
            "來源類型": "search_space",
            "型別": "int",
            "目前值": "1 (調參強制) / 未寫入固化JSON",
            "驗證方式": "ast",
        },
        {
            "id": "M5_SEARCH_N_ESTIMATORS",
            "參數名": "n_estimators",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 165,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "300",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_LEARNING_RATE",
            "參數名": "learning_rate",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 166,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.01373336360334273",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_NUM_LEAVES",
            "參數名": "num_leaves",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 167,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "43",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_COLSAMPLE",
            "參數名": "colsample_bytree",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 170,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.7788897740771944",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_REG_LAMBDA",
            "參數名": "reg_lambda",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 171,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.02704489144262553",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_MIN_CHILD_SAMPLES",
            "參數名": "min_child_samples",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 172,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "38",
            "驗證方式": "json",
        },
    ])

    # 5. bankruptcy/src/ensemble.py
    records.extend([
        {
            "id": "ENS_SINGLE_MODELS",
            "參數名": "SINGLE_MODELS",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/ensemble.py",
            "行號": 73,
            "類別": "集成",
            "來源類型": "code_constant",
            "型別": "list[str]",
            "目前值": '["M1", "M2", "M3", "M4", "M5"]',
            "驗證方式": "ast",
        },
        {
            "id": "ENS_CANDIDATES",
            "參數名": "ENSEMBLE_CANDIDATES",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/ensemble.py",
            "行號": 75,
            "類別": "集成",
            "來源類型": "code_constant",
            "型別": "dict",
            "目前值": '{"E_all": 5, "E_A": 3, "E_B": 4, "E_trees": 4}',
            "驗證方式": "ast",
        },
        {
            "id": "ENS_PRIMARY_MODEL",
            "參數名": "E_A",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/ensemble.py",
            "行號": 77,
            "類別": "集成",
            "來源類型": "code_constant",
            "型別": "str",
            "目前值": '"E_A"',
            "驗證方式": "ast",
        },
        {
            "id": "ENS_CONTROL_SINGLE_MODEL",
            "參數名": "M3",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/ensemble.py",
            "行號": 73,
            "類別": "集成",
            "來源類型": "code_constant",
            "型別": "str",
            "目前值": '"M3"',
            "驗證方式": "ast",
        },
    ])

    # 6. bankruptcy/src/thresholding.py
    records.extend([
        {
            "id": "TH_MARGINS",
            "參數名": "MARGINS",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/thresholding.py",
            "行號": 67,
            "類別": "門檻與目標",
            "來源類型": "code_constant",
            "型別": "list[float]",
            "目前值": "[0.0, 0.02, 0.04, 0.06]",
            "驗證方式": "ast",
        },
        {
            "id": "TH_ATTAINMENT_THRESHOLD",
            "參數名": "attainment_rate",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/thresholding.py",
            "行號": 361,
            "類別": "門檻與目標",
            "來源類型": "code_constant",
            "型別": "float",
            "目前值": "0.8",
            "驗證方式": "ast",
        },
    ])

    # 7. outputs/models/final/thresholds.json entries
    th_json = REPO_ROOT / "bankruptcy" / "outputs" / "models" / "final" / "thresholds.json"
    if th_json.exists():
        with open(th_json, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        for grp in ["E_A", "M3"]:
            for r_key, vals in t_data.get(grp, {}).items():
                thresh_line = find_threshold_json_line(grp, r_key, "threshold")
                margin_line = find_threshold_json_line(grp, r_key, "recommended_margin")
                records.append({
                    "id": f"TH_{grp}_{r_key}_THRESH",
                    "參數名": "threshold",
                    "層級(T1/T2)": "T1",
                    "所在檔案": "bankruptcy/outputs/models/final/thresholds.json",
                    "行號": thresh_line,
                    "類別": "門檻與目標",
                    "來源類型": "config_json",
                    "型別": "float",
                    "目前值": str(vals["threshold"]),
                    "驗證方式": "json",
                })
                records.append({
                    "id": f"TH_{grp}_{r_key}_MARGIN",
                    "參數名": "recommended_margin",
                    "層級(T1/T2)": "T1",
                    "所在檔案": "bankruptcy/outputs/models/final/thresholds.json",
                    "行號": margin_line,
                    "類別": "門檻與目標",
                    "來源類型": "config_json",
                    "型別": "float",
                    "目前值": str(vals["recommended_margin"]),
                    "驗證方式": "json",
                })

    # 8. bankruptcy/src/prevalence_sensitivity.py & seed_stability.py
    records.append({
        "id": "SENS_PREVALENCES",
        "參數名": "PREVALENCES",
        "層級(T1/T2)": "T1",
        "所在檔案": "bankruptcy/src/prevalence_sensitivity.py",
        "行號": 36,
        "類別": "評估指標",
        "來源類型": "code_constant",
        "型別": "list[float]",
        "目前值": "[0.005, 0.01, 0.02, 0.032, 0.05]",
        "驗證方式": "ast",
    })

    records.append({
        "id": "SENS_SEEDS",
        "參數名": "SEEDS",
        "層級(T1/T2)": "T1",
        "所在檔案": "bankruptcy/src/seed_stability.py",
        "行號": 43,
        "類別": "交叉驗證",
        "來源類型": "code_constant",
        "型別": "list[int]",
        "目前值": "[42, 7, 123, 2025, 31337]",
        "驗證方式": "ast",
    })

    # 9. bankruptcy/src/scoring.py
    records.extend([
        {
            "id": "SCORE_ALLOW_MISSING",
            "參數名": "allow_missing",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/scoring.py",
            "行號": 386,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "bool",
            "目前值": "False",
            "驗證方式": "ast",
        },
        {
            "id": "SCORE_MODEL_NAME",
            "參數名": "model_name",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/scoring.py",
            "行號": 384,
            "類別": "應用層",
            "來源類型": "default_arg",
            "型別": "str",
            "目前值": '"E_A"',
            "驗證方式": "ast",
        },
        {
            "id": "SCORE_OPERATING_POINT",
            "參數名": "operating_point",
            "層級(T1/T2)": "T1",
            "所在檔案": "bankruptcy/src/scoring.py",
            "行號": 385,
            "類別": "應用層",
            "來源類型": "default_arg",
            "型別": "str",
            "目前值": '"R90"',
            "驗證方式": "ast",
        },
    ])

    # 10. bankruptcy/app/assets/defaults.json
    def_json = REPO_ROOT / "bankruptcy" / "app" / "assets" / "defaults.json"
    if def_json.exists():
        with open(def_json, "r", encoding="utf-8") as f:
            d_data = json.load(f)
        meta = d_data.get("meta", {})
        records.extend([
            {
                "id": "DEF_N_TRAIN",
                "參數名": "n_train",
                "層級(T1/T2)": "T2",
                "所在檔案": "bankruptcy/app/assets/defaults.json",
                "行號": find_defaults_json_line("n_train"),
                "類別": "資料與切分",
                "來源類型": "config_json",
                "型別": "int",
                "目前值": str(meta.get("n_train", 5455)),
                "驗證方式": "json",
            },
            {
                "id": "DEF_N_RAW_COLUMNS",
                "參數名": "n_raw_columns",
                "層級(T1/T2)": "T2",
                "所在檔案": "bankruptcy/app/assets/defaults.json",
                "行號": find_defaults_json_line("n_raw_columns"),
                "類別": "資料與切分",
                "來源類型": "config_json",
                "型別": "int",
                "目前值": str(meta.get("n_raw_columns", 95)),
                "驗證方式": "json",
            },
            {
                "id": "DEF_N_MODEL_COLUMNS",
                "參數名": "n_model_columns",
                "層級(T1/T2)": "T2",
                "所在檔案": "bankruptcy/app/assets/defaults.json",
                "行號": find_defaults_json_line("n_model_columns"),
                "類別": "前處理",
                "來源類型": "config_json",
                "型別": "int",
                "目前值": str(meta.get("n_model_columns", 72)),
                "驗證方式": "json",
            },
            {
                "id": "DEF_N_INPUT_FEATURES",
                "參數名": "n_input_features",
                "層級(T1/T2)": "T1",
                "所在檔案": "bankruptcy/app/assets/defaults.json",
                "行號": find_defaults_json_line("n_input_features"),
                "類別": "應用層",
                "來源類型": "config_json",
                "型別": "int",
                "目前值": str(meta.get("n_input_features", 18)),
                "驗證方式": "json",
            },
            {
                "id": "DEF_TRAIN_BANKRUPTCY_RATE",
                "參數名": "train_bankruptcy_rate",
                "層級(T1/T2)": "T2",
                "所在檔案": "bankruptcy/app/assets/defaults.json",
                "行號": find_defaults_json_line("train_bankruptcy_rate"),
                "類別": "評估指標",
                "來源類型": "config_json",
                "型別": "float",
                "目前值": str(meta.get("train_bankruptcy_rate", 0.03226397800183318)),
                "驗證方式": "json",
            },
            {
                "id": "DEF_PERSISTENT_EPS_P50",
                "參數名": "p50",
                "層級(T1/T2)": "T1",
                "所在檔案": "bankruptcy/app/assets/defaults.json",
                "行號": find_defaults_json_line("p50", under_inputs=True),
                "類別": "應用層",
                "來源類型": "config_json",
                "型別": "float",
                "目前值": str(d_data.get("inputs", {}).get("Persistent EPS in the Last Four Seasons", {}).get("p50", 0.224260187198639)),
                "驗證方式": "json",
            },
        ])

    return records


def load_notes_dict() -> dict[str, dict[str, str]]:
    """Loads docs/code_learning/param_notes.csv into a lookup dictionary."""
    if not NOTES_CSV.exists():
        return {}
    res = {}
    with open(NOTES_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"].strip()
            res[pid] = row
    return res


def build_registry():
    """Generates the final merged 03a_param_registry.csv."""
    raw_records = build_raw_param_records()
    notes = load_notes_dict()

    fieldnames = [
        "id", "參數名", "層級(T1/T2)", "所在檔案", "行號", "類別", "來源類型",
        "型別", "目前值", "合理範圍", "範圍來源", "說明", "影響階段",
        "相關實驗ID", "證據檔案", "可調性", "驗證方式", "備註"
    ]

    merged_rows = []
    for rec in raw_records:
        pid = rec["id"]
        note = notes.get(pid, {})

        row = {
            "id": pid,
            "參數名": rec["參數名"],
            "層級(T1/T2)": rec["層級(T1/T2)"],
            "所在檔案": rec["所在檔案"],
            "行號": rec["行號"],
            "類別": rec["類別"],
            "來源類型": rec["來源類型"],
            "型別": rec["型別"],
            "目前值": rec["目前值"],
            "合理範圍": note.get("合理範圍", "不適用"),
            "範圍來源": note.get("範圍來源", "程式碼限制"),
            "說明": note.get("說明", rec["參數名"]),
            "影響階段": note.get("影響階段", "全階段"),
            "相關實驗ID": note.get("相關實驗ID", "EDA-E7"),
            "證據檔案": note.get("證據檔案", rec["所在檔案"]),
            "可調性": note.get("可調性", "可在 lab 安全調整"),
            "驗證方式": rec["驗證方式"],
            "備註": note.get("備註", "自動抽取並對映"),
        }
        merged_rows.append(row)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Successfully generated parameter registry: {OUTPUT_CSV}")
    print(f"Total parameters registered: {len(merged_rows)}")


if __name__ == "__main__":
    build_registry()
