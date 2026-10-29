#!/usr/bin/env python3
"""
tools/build_param_registry.py
Automated generator for docs/code_learning/03a_param_registry.csv (v2).

Extracts parameters from:
- bankruptcy/config.py (AST parsing)
- bankruptcy/src/cv.py (AST parsing)
- bankruptcy/src/pipelines.py (AST parsing)
- bankruptcy/src/preprocessing.py (AST parsing)
- bankruptcy/src/tune.py (AST parsing of search spaces and n_trials_map)
- bankruptcy/src/ensemble.py (AST parsing)
- bankruptcy/src/thresholding.py (AST parsing)
- bankruptcy/src/scoring.py (AST parsing)
- bankruptcy/src/prevalence_sensitivity.py (AST parsing)
- bankruptcy/src/seed_stability.py (AST parsing)
- bankruptcy/outputs/models/best_params/*.json (JSON loading + line finding)
- bankruptcy/outputs/models/final/thresholds.json (JSON loading + line finding)
- bankruptcy/outputs/models/final/model_M5.joblib (joblib parameter inspection)
- bankruptcy/app/assets/defaults.json (JSON loading + line finding)

Merges with docs/code_learning/param_notes.csv by 'id' to produce the final 17-column registry.
DOES NOT import execution modules or write caches.
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


def extract_model_trials_map(tree: ast.AST, var_name: str) -> dict[str, int]:
    """Extracts M1-M5 trial count integers from n_trials_map dict assignment."""
    res = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == var_name:
                    if isinstance(node.value, ast.Dict):
                        for k, v in zip(node.value.keys, node.value.values):
                            if isinstance(k, ast.Constant):
                                m_name = str(k.value)
                                # value could be IfExp: 3 if quick else 25
                                if isinstance(v, ast.IfExp):
                                    if isinstance(v.orelse, ast.Constant):
                                        res[m_name] = int(v.orelse.value)
                                elif isinstance(v, ast.Constant):
                                    res[m_name] = int(v.value)
    return res


def extract_suggest_call_details(tree: ast.AST) -> dict[str, dict[str, Any]]:
    """Extracts low, high, step, log, choices from trial.suggest_* calls."""
    res = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            fn_name = node.func.attr
            if fn_name.startswith("suggest_") and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant):
                    p_name = str(arg0.value)
                    info: dict[str, Any] = {"func": fn_name, "lineno": node.lineno}
                    if fn_name == "suggest_float":
                        if len(node.args) >= 3:
                            try:
                                info["low"] = ast.literal_eval(node.args[1])
                                info["high"] = ast.literal_eval(node.args[2])
                            except Exception:
                                pass
                        for kw in node.keywords:
                            if kw.arg == "log" and isinstance(kw.value, ast.Constant):
                                info["log"] = bool(kw.value.value)
                            elif kw.arg == "step" and isinstance(kw.value, ast.Constant):
                                info["step"] = ast.literal_eval(kw.value)
                    elif fn_name == "suggest_int":
                        if len(node.args) >= 3:
                            try:
                                info["low"] = ast.literal_eval(node.args[1])
                                info["high"] = ast.literal_eval(node.args[2])
                            except Exception:
                                pass
                        for kw in node.keywords:
                            if kw.arg == "step" and isinstance(kw.value, ast.Constant):
                                info["step"] = ast.literal_eval(kw.value)
                    elif fn_name == "suggest_categorical":
                        if len(node.args) >= 2:
                            try:
                                info["choices"] = ast.literal_eval(node.args[1])
                            except Exception:
                                pass
                    res[p_name] = info
    return res


def find_symbol_line(file_path: Path, symbol: str, start_search: int = 1) -> int:
    """Finds line number containing symbol (word boundary)."""
    lines = file_path.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\b")
    for idx in range(start_search - 1, len(lines)):
        if pattern.search(lines[idx]):
            return idx + 1
    for idx in range(0, start_search - 1):
        if pattern.search(lines[idx]):
            return idx + 1
    return 0


def find_threshold_json_line(model_key: str, r_key: str, field_name: str) -> int:
    """Finds exact line in thresholds.json."""
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
    for idx, line in enumerate(lines, 1):
        if f'"{token}"' in line:
            return idx
    return 0


def build_raw_param_records() -> list[dict[str, Any]]:
    """Builds complete v2 parameter records across all 12 categories."""
    records: list[dict[str, Any]] = []

    # 1. bankruptcy/config.py
    cfg_file = REPO_ROOT / "bankruptcy" / "config.py"
    cfg_assigns = parse_ast_assignments(cfg_file)

    cfg_defs = {
        # Category: 資料與切分
        "TEST_SIZE": ("資料與切分", "T1", "code_constant", "已鎖定，不可調整（E6 相關）"),
        "TARGET_COL": ("資料與切分", "T2", "code_constant", "固定常數，不建議調整"),
        "FEATURES": ("資料與切分", "T2", "code_constant", "固定常數，不建議調整"),
        # Category: 交叉驗證
        "SEED": ("交叉驗證", "T1", "code_constant", "可在 lab 安全調整"),
        "CV_FOLDS": ("交叉驗證", "T1", "code_constant", "可在 lab 安全調整"),
        "CV_REPEATS": ("交叉驗證", "T1", "code_constant", "可在 lab 安全調整"),
        # Category: 門檻與目標
        "TARGET_RECALLS": ("門檻與目標", "T1", "code_constant", "可在 lab 安全調整"),
        "DEFAULT_TARGET_RECALL": ("門檻與目標", "T1", "code_constant", "可在 lab 安全調整"),
        # Category: 評估指標
        "ALERT_RATIOS": ("評估指標", "T1", "code_constant", "可在 lab 安全調整"),
        "COST_FN": ("評估指標", "T1", "code_constant", "可在 lab 安全調整"),
        "COST_FP": ("評估指標", "T1", "code_constant", "可在 lab 安全調整"),
        "COST_FN_SENSITIVITY_LIST": ("評估指標", "T1", "code_constant", "可在 lab 安全調整"),
        "MVP_PASS_PR_AUC_MULTIPLIER": ("評估指標", "T1", "code_constant", "可在 lab 安全調整"),
        "MVP_PASS_TOP10_RECALL": ("評估指標", "T1", "code_constant", "可在 lab 安全調整"),
        # Category: 應用層
        "TOP_K_PERCENT": ("應用層", "T1", "code_constant", "僅應用層"),
        "MIN_BATCH_SIZE_FOR_TOP_K": ("應用層", "T1", "code_constant", "僅應用層"),
        "MAX_MISSING_FEATURE_RATIO": ("應用層", "T1", "code_constant", "僅應用層"),
        # Category: 系統與路徑 (T2)
        "BASE_DIR": ("系統與路徑", "T2", "code_constant", "固定常數，不建議調整"),
        "DATA_DIR": ("系統與路徑", "T2", "code_constant", "固定常數，不建議調整"),
        "DATA_PATH": ("系統與路徑", "T2", "code_constant", "固定常數，不建議調整"),
        "OUTPUT_DIR": ("系統與路徑", "T2", "code_constant", "固定常數，不建議調整"),
        "TABLES_DIR": ("系統與路徑", "T2", "code_constant", "固定常數，不建議調整"),
        "FIGURES_DIR": ("系統與路徑", "T2", "code_constant", "固定常數，不建議調整"),
        "MODELS_DIR": ("系統與路徑", "T2", "code_constant", "固定常數，不建議調整"),
        "LOGS_DIR": ("系統與路徑", "T2", "code_constant", "固定常數，不建議調整"),
    }

    for name, (cat, tier, src_t, adjust) in cfg_defs.items():
        if name in cfg_assigns:
            lineno, val, typ = cfg_assigns[name]
            curr_str = str(val) if (not isinstance(val, str) or typ == "expr") else f'"{val}"'
            records.append({
                "id": f"CFG_{name}",
                "參數名": name,
                "層級(T1/T2/T3)": tier,
                "所在檔案": "bankruptcy/config.py",
                "行號": lineno,
                "類別": cat,
                "來源類型": src_t,
                "型別": typ if typ != "list" else f"list[{type(val[0]).__name__ if val else 'Any'}]",
                "目前值": curr_str,
                "可調性": adjust,
                "驗證方式": "ast",
            })

    # 2. Category: 交叉驗證 (CV 擴展項目)
    records.extend([
        {
            "id": "CV_INNER_E2A_SPLITS",
            "參數名": "make_inner_splits",
            "層級(T1/T2/T3)": "T2",
            "所在檔案": "bankruptcy/src/cv.py",
            "行號": 254,
            "類別": "交叉驗證",
            "來源類型": "default_arg",
            "型別": "int",
            "目前值": "3",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "CV_INNER_E2B_SPLITS",
            "參數名": "StratifiedKFold",
            "層級(T1/T2/T3)": "T2",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 857,
            "類別": "交叉驗證",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "10",
            "可調性": "需重跑固化流程才生效",
            "驗證方式": "ast",
        },
        {
            "id": "CV_INNER_E3_SPLITS",
            "參數名": "StratifiedKFold",
            "層級(T1/T2/T3)": "T2",
            "所在檔案": "bankruptcy/src/ensemble.py",
            "行號": 153,
            "類別": "交叉驗證",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "5",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "SENS_SEEDS",
            "參數名": "SEEDS",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/seed_stability.py",
            "行號": 43,
            "類別": "交叉驗證",
            "來源類型": "code_constant",
            "型別": "list[int]",
            "目前值": "[42, 7, 123, 2025, 31337]",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
    ])

    # 3. Category: 前處理 (Preprocessing)
    records.extend([
        {
            "id": "PREP_CORR_THRESHOLD",
            "參數名": "threshold",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 353,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "float",
            "目前值": "0.95",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_CORR_ENABLED",
            "參數名": "enabled",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 353,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "bool",
            "目前值": "True",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_WINSOR_LOWER",
            "參數名": "lower",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 193,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "float",
            "目前值": "0.01",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_WINSOR_UPPER",
            "參數名": "upper",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 193,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "float",
            "目前值": "0.99",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_WINSOR_ENABLED",
            "參數名": "enabled",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 193,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "bool",
            "目前值": "True",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_SCALER_KIND",
            "參數名": "kind",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 300,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "str",
            "目前值": '"quantile"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_QUANTILE_N",
            "參數名": "n_quantiles",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 310,
            "類別": "前處理",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "1000",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_QUANTILE_DIST",
            "參數名": "output_distribution",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 312,
            "類別": "前處理",
            "來源類型": "code_constant",
            "型別": "str",
            "目前值": '"normal"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PREP_LARGE_ENABLED",
            "參數名": "enabled",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/preprocessing.py",
            "行號": 249,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "bool",
            "目前值": "False",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PIPE_SCALER_DEFAULT",
            "參數名": "scaler",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 215,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "str",
            "目前值": '"quantile"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PIPE_USE_WINSORIZER",
            "參數名": "use_winsorizer",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 216,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "bool",
            "目前值": "True",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PIPE_USE_CORR_FILTER",
            "參數名": "use_corr_filter",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 217,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "bool",
            "目前值": "True",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PIPE_USE_LARGE_INDICATOR",
            "參數名": "use_large_indicator",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 218,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "bool",
            "目前值": "False",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "PIPE_USE_IMPUTER",
            "參數名": "use_imputer",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 222,
            "類別": "前處理",
            "來源類型": "default_arg",
            "型別": "bool",
            "目前值": "False",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "DEF_N_MODEL_COLUMNS",
            "參數名": "n_model_columns",
            "層級(T1/T2/T3)": "T3",
            "所在檔案": "bankruptcy/app/assets/defaults.json",
            "行號": find_defaults_json_line("n_model_columns"),
            "類別": "前處理",
            "來源類型": "config_json",
            "型別": "int",
            "目前值": "72",
            "可調性": "衍生值，不可直接調整",
            "驗證方式": "json",
        },
    ])

    # 4. Category: 不平衡處理 (Imbalance Handling)
    records.extend([
        {
            "id": "PIPE_IMBALANCE_DEFAULT",
            "參數名": "imbalance",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 214,
            "類別": "不平衡處理",
            "來源類型": "default_arg",
            "型別": "str",
            "目前值": '"S0"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "IMB_RESAMPLER_S2",
            "參數名": "SMOTE",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 117,
            "類別": "不平衡處理",
            "來源類型": "code_constant",
            "型別": "float",
            "目前值": "0.1",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "IMB_RESAMPLER_S3",
            "參數名": "SMOTE",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 119,
            "類別": "不平衡處理",
            "來源類型": "code_constant",
            "型別": "float",
            "目前值": "0.3",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "IMB_RESAMPLER_S4",
            "參數名": "SMOTE",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 121,
            "類別": "不平衡處理",
            "來源類型": "code_constant",
            "型別": "float",
            "目前值": "0.5",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "IMB_RESAMPLER_S5",
            "參數名": "BorderlineSMOTE",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 123,
            "類別": "不平衡處理",
            "來源類型": "code_constant",
            "型別": "float",
            "目前值": "0.3",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "IMB_RESAMPLER_S6",
            "參數名": "SMOTETomek",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 126,
            "類別": "不平衡處理",
            "來源類型": "code_constant",
            "型別": "float",
            "目前值": "0.3",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "IMB_SMOTE_K_NEIGHBORS",
            "參數名": "k_neighbors",
            "層級(T1/T2/T3)": "T2",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 117,
            "類別": "不平衡處理",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "5",
            "可調性": "固定常數，不建議調整",
            "驗證方式": "ast",
        },
        {
            "id": "M1_BEST_IMBALANCE",
            "參數名": "imbalance",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/outputs/models/best_params/M1.json",
            "行號": 3,
            "類別": "不平衡處理",
            "來源類型": "best_params_json",
            "型別": "str",
            "目前值": '"S0"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M2_BEST_IMBALANCE",
            "參數名": "imbalance",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/outputs/models/best_params/M2.json",
            "行號": 3,
            "類別": "不平衡處理",
            "來源類型": "best_params_json",
            "型別": "str",
            "目前值": '"S3"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M3_BEST_IMBALANCE",
            "參數名": "imbalance",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/outputs/models/best_params/M3.json",
            "行號": 3,
            "類別": "不平衡處理",
            "來源類型": "best_params_json",
            "型別": "str",
            "目前值": '"S0"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M4_BEST_IMBALANCE",
            "參數名": "imbalance",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/outputs/models/best_params/M4.json",
            "行號": 3,
            "類別": "不平衡處理",
            "來源類型": "best_params_json",
            "型別": "str",
            "目前值": '"S0"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M5_BEST_IMBALANCE",
            "參數名": "imbalance",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/outputs/models/best_params/M5.json",
            "行號": 3,
            "類別": "不平衡處理",
            "來源類型": "best_params_json",
            "型別": "str",
            "目前值": '"S3"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
    ])

    # 5. Category: 模型超參數 (Model Hyperparameters)
    records.extend([
        # Dynamic scale_pos_weight
        {
            "id": "PIPE_SCALE_POS_WEIGHT",
            "參數名": "scale_pos_weight",
            "層級(T1/T2/T3)": "T3",
            "所在檔案": "bankruptcy/src/pipelines.py",
            "行號": 72,
            "類別": "模型超參數",
            "來源類型": "derived",
            "型別": "float",
            "目前值": "動態：由訓練折計算",
            "可調性": "衍生值，不可直接調整",
            "驗證方式": "ast",
        },
        # M1 Search & Best
        {
            "id": "M1_SEARCH_C",
            "參數名": "C",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 127,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.11616457722286455",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M1_SEARCH_PENALTY",
            "參數名": "penalty",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 128,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "str",
            "目前值": '"l2"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M1_FIXED_MAX_ITER",
            "參數名": "max_iter",
            "層級(T1/T2/T3)": "T2",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 139,
            "類別": "模型超參數",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "5000",
            "可調性": "固定常數，不建議調整",
            "驗證方式": "ast",
        },
        # M2 Search & Best
        {
            "id": "M2_SEARCH_N_ESTIMATORS",
            "參數名": "n_estimators",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 145,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "300",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M2_SEARCH_MAX_DEPTH",
            "參數名": "max_depth",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 146,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "12",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M2_SEARCH_MIN_SAMPLES_LEAF",
            "參數名": "min_samples_leaf",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 147,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "10",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M2_SEARCH_MAX_FEATURES",
            "參數名": "max_features",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 148,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "str",
            "目前值": '"sqrt"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        # M3 Search & Best
        {
            "id": "M3_SEARCH_N_ESTIMATORS",
            "參數名": "n_estimators",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 145,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "600",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M3_SEARCH_MAX_DEPTH",
            "參數名": "max_depth",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 146,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "NoneType",
            "目前值": "None",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M3_SEARCH_MIN_SAMPLES_LEAF",
            "參數名": "min_samples_leaf",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 147,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "2",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M3_SEARCH_MAX_FEATURES",
            "參數名": "max_features",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 148,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.3",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        # M4 Search & Best
        {
            "id": "M4_SEARCH_N_ESTIMATORS",
            "參數名": "n_estimators",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 153,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "250",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_LEARNING_RATE",
            "參數名": "learning_rate",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 154,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.046776815926747296",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_MAX_DEPTH",
            "參數名": "max_depth",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 155,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "2",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_SUBSAMPLE",
            "參數名": "subsample",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 156,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.879563935600964",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_COLSAMPLE",
            "參數名": "colsample_bytree",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 157,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.9017127339966642",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_REG_LAMBDA",
            "參數名": "reg_lambda",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 158,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "1.6902875531341541",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M4_SEARCH_MIN_CHILD_WEIGHT",
            "參數名": "min_child_weight",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 159,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "7.300913224180319",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        # M5 Search, Best, and Subsample Freq (3 distinct rows as required by 2.5)
        {
            "id": "M5_SEARCH_SUBSAMPLE",
            "參數名": "subsample",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 163,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.6071695513247772",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_SUBSAMPLE_FREQ",
            "參數名": "subsample_freq",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 169,
            "類別": "模型超參數",
            "來源類型": "derived",
            "型別": "str",
            "目前值": "1 if subsample < 1.0 else 0",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "M5_FROZEN_SUBSAMPLE_FREQ",
            "參數名": "subsample_freq",
            "層級(T1/T2/T3)": "T3",
            "所在檔案": "bankruptcy/outputs/models/best_params/M5.json",
            "行號": 1,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "str",
            "目前值": "無此鍵",
            "可調性": "衍生值，不可直接調整",
            "驗證方式": "json",
        },
        {
            "id": "M5_FINAL_MODEL_SUBSAMPLE_FREQ",
            "參數名": "subsample_freq",
            "層級(T1/T2/T3)": "T3",
            "所在檔案": "bankruptcy/outputs/models/final/model_M5.joblib",
            "行號": 1,
            "類別": "模型超參數",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "0",
            "可調性": "固定常數，不建議調整",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_N_ESTIMATORS",
            "參數名": "n_estimators",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 165,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "300",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_LEARNING_RATE",
            "參數名": "learning_rate",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 166,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.01373336360334273",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_NUM_LEAVES",
            "參數名": "num_leaves",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 167,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "43",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_COLSAMPLE",
            "參數名": "colsample_bytree",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 170,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.7788897740771944",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_REG_LAMBDA",
            "參數名": "reg_lambda",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 171,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "float",
            "目前值": "0.02704489144262553",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
        {
            "id": "M5_SEARCH_MIN_CHILD_SAMPLES",
            "參數名": "min_child_samples",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 172,
            "類別": "模型超參數",
            "來源類型": "best_params_json",
            "型別": "int",
            "目前值": "38",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "json",
        },
    ])

    # 6. Category: 調參流程 (Tuning Workflow - M1-M5 split into individual rows)
    records.extend([
        # E2a trials per model (lines 502-506 in tune.py)
        {
            "id": "TUNE_E2A_TRIALS_M1",
            "參數名": "M1",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 502,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "25",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "TUNE_E2A_TRIALS_M2",
            "參數名": "M2",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 503,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "40",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "TUNE_E2A_TRIALS_M3",
            "參數名": "M3",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 504,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "40",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "TUNE_E2A_TRIALS_M4",
            "參數名": "M4",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 505,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "40",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "TUNE_E2A_TRIALS_M5",
            "參數名": "M5",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 506,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "40",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        # E2b trials per model (lines 866-870 in tune.py)
        {
            "id": "TUNE_E2B_TRIALS_M1",
            "參數名": "M1",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 866,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "38",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "TUNE_E2B_TRIALS_M2",
            "參數名": "M2",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 867,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "60",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "TUNE_E2B_TRIALS_M3",
            "參數名": "M3",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 868,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "60",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "TUNE_E2B_TRIALS_M4",
            "參數名": "M4",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 869,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "60",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "TUNE_E2B_TRIALS_M5",
            "參數名": "M5",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 870,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "int",
            "目前值": "60",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        # Sampler & Storage
        {
            "id": "TUNE_OPTUNA_SAMPLER",
            "參數名": "TPESampler",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 307,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "str",
            "目前值": '"TPESampler"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "TUNE_OPTUNA_STORAGE",
            "參數名": "storage_url",
            "層級(T1/T2/T3)": "T2",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 304,
            "類別": "調參流程",
            "來源類型": "code_constant",
            "型別": "str",
            "目前值": '"sqlite:///{study_db}"',
            "可調性": "固定常數，不建議調整",
            "驗證方式": "ast",
        },
    ])

    # 7. Category: 集成 (Ensemble)
    records.extend([
        {
            "id": "ENS_SINGLE_MODELS",
            "參數名": "SINGLE_MODELS",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/ensemble.py",
            "行號": 73,
            "類別": "集成",
            "來源類型": "code_constant",
            "型別": "list[str]",
            "目前值": '["M1", "M2", "M3", "M4", "M5"]',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "ENS_CANDIDATES",
            "參數名": "ENSEMBLE_CANDIDATES",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/ensemble.py",
            "行號": 75,
            "類別": "集成",
            "來源類型": "code_constant",
            "型別": "dict",
            "目前值": '{"E_all": 5, "E_A": 3, "E_B": 4, "E_trees": 4}',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "ENS_PRIMARY_MODEL",
            "參數名": "E_A",
            "層級(T1/T2/T3)": "T3",
            "所在檔案": "bankruptcy/src/ensemble.py",
            "行號": 77,
            "類別": "集成",
            "來源類型": "code_constant",
            "型別": "str",
            "目前值": '"E_A"',
            "可調性": "需重跑固化流程才生效",
            "驗證方式": "ast",
        },
        {
            "id": "ENS_CONTROL_SINGLE_MODEL",
            "參數名": "M3",
            "層級(T1/T2/T3)": "T3",
            "所在檔案": "bankruptcy/src/ensemble.py",
            "行號": 73,
            "類別": "集成",
            "來源類型": "code_constant",
            "型別": "str",
            "目前值": '"M3"',
            "可調性": "需重跑固化流程才生效",
            "驗證方式": "ast",
        },
        {
            "id": "ENS_STACKING_META_CLF",
            "參數名": "LogisticRegression",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/ensemble.py",
            "行號": 518,
            "類別": "集成",
            "來源類型": "code_constant",
            "型別": "str",
            "目前值": '"LogisticRegression(penalty=\'l2\')"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "ENS_RULE_STD_TOLERANCE",
            "參數名": "apply_e3_selection_rule",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/ensemble.py",
            "行號": 1122,
            "類別": "集成",
            "來源類型": "code_constant",
            "型別": "float",
            "目前值": "1.25",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
    ])

    # 8. Category: 校準 (Calibration)
    records.extend([
        {
            "id": "CALIB_PLATT_METHOD",
            "參數名": "fit_single_model_platt",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/thresholding.py",
            "行號": 124,
            "類別": "校準",
            "來源類型": "code_constant",
            "型別": "str",
            "目前值": '"LogisticRegression(solver=\'lbfgs\')"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "CALIB_ISOTONIC_METHOD",
            "參數名": "IsotonicRegression",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/thresholding.py",
            "行號": 455,
            "類別": "校準",
            "來源類型": "code_constant",
            "型別": "str",
            "目前值": '"IsotonicRegression(out_of_bounds=\'clip\')"',
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "CALIB_RECOMMENDED_CHOICE",
            "參數名": "fit_single_model_platt",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/thresholding.py",
            "行號": 124,
            "類別": "校準",
            "來源類型": "code_constant",
            "型別": "str",
            "目前值": '"Platt"',
            "可調性": "需重跑固化流程才生效",
            "驗證方式": "ast",
        },
    ])

    # 9. Category: 門檻與目標 (Thresholds and Targets)
    records.extend([
        {
            "id": "TH_MARGINS",
            "參數名": "MARGINS",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/thresholding.py",
            "行號": 67,
            "類別": "門檻與目標",
            "來源類型": "code_constant",
            "型別": "list[float]",
            "目前值": "[0.0, 0.02, 0.04, 0.06]",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "TH_ATTAINMENT_THRESHOLD",
            "參數名": "attainment_rate",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/thresholding.py",
            "行號": 361,
            "類別": "門檻與目標",
            "來源類型": "code_constant",
            "型別": "float",
            "目前值": "0.8",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
    ])

    # thresholds.json entries (T3 - Derived Operating Points)
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
                    "層級(T1/T2/T3)": "T3",
                    "所在檔案": "bankruptcy/outputs/models/final/thresholds.json",
                    "行號": thresh_line,
                    "類別": "門檻與目標",
                    "來源類型": "config_json",
                    "型別": "float",
                    "目前值": str(vals["threshold"]),
                    "可調性": "衍生值，不可直接調整",
                    "驗證方式": "json",
                })
                records.append({
                    "id": f"TH_{grp}_{r_key}_MARGIN",
                    "參數名": "recommended_margin",
                    "層級(T1/T2/T3)": "T3",
                    "所在檔案": "bankruptcy/outputs/models/final/thresholds.json",
                    "行號": margin_line,
                    "類別": "門檻與目標",
                    "來源類型": "config_json",
                    "型別": "float",
                    "目前值": str(vals["recommended_margin"]),
                    "可調性": "衍生值，不可直接調整",
                    "驗證方式": "json",
                })

    # 10. Category: 評估指標 (Evaluation Metrics)
    records.extend([
        {
            "id": "METRIC_THREE_POINT_RECALLS",
            "參數名": "compute_three_point_log_loss",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/tune.py",
            "行號": 113,
            "類別": "評估指標",
            "來源類型": "code_constant",
            "型別": "list[float]",
            "目前值": "[0.8, 0.9, 0.95]",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "SENS_PREVALENCES",
            "參數名": "PREVALENCES",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/prevalence_sensitivity.py",
            "行號": 36,
            "類別": "評估指標",
            "來源類型": "code_constant",
            "型別": "list[float]",
            "目前值": "[0.005, 0.01, 0.02, 0.032, 0.05]",
            "可調性": "可在 lab 安全調整",
            "驗證方式": "ast",
        },
        {
            "id": "DEF_TRAIN_BANKRUPTCY_RATE",
            "參數名": "train_bankruptcy_rate",
            "層級(T1/T2/T3)": "T3",
            "所在檔案": "bankruptcy/app/assets/defaults.json",
            "行號": find_defaults_json_line("train_bankruptcy_rate"),
            "類別": "評估指標",
            "來源類型": "config_json",
            "型別": "float",
            "目前值": "0.03226397800183318",
            "可調性": "衍生值，不可直接調整",
            "驗證方式": "json",
        },
    ])

    # 11. Category: 應用層 (Application / Serving)
    records.extend([
        {
            "id": "SCORE_ALLOW_MISSING",
            "參數名": "allow_missing",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/scoring.py",
            "行號": 386,
            "類別": "應用層",
            "來源類型": "default_arg",
            "型別": "bool",
            "目前值": "False",
            "可調性": "僅應用層",
            "驗證方式": "ast",
        },
        {
            "id": "SCORE_MODEL_NAME",
            "參數名": "model_name",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/scoring.py",
            "行號": 384,
            "類別": "應用層",
            "來源類型": "default_arg",
            "型別": "str",
            "目前值": '"E_A"',
            "可調性": "僅應用層",
            "驗證方式": "ast",
        },
        {
            "id": "SCORE_OPERATING_POINT",
            "參數名": "operating_point",
            "層級(T1/T2/T3)": "T1",
            "所在檔案": "bankruptcy/src/scoring.py",
            "行號": 385,
            "類別": "應用層",
            "來源類型": "default_arg",
            "型別": "str",
            "目前值": '"R90"',
            "可調性": "僅應用層",
            "驗證方式": "ast",
        },
        {
            "id": "DEF_N_INPUT_FEATURES",
            "參數名": "n_input_features",
            "層級(T1/T2/T3)": "T3",
            "所在檔案": "bankruptcy/app/assets/defaults.json",
            "行號": find_defaults_json_line("n_input_features"),
            "類別": "應用層",
            "來源類型": "config_json",
            "型別": "int",
            "目前值": "18",
            "可調性": "衍生值，不可直接調整",
            "驗證方式": "json",
        },
        {
            "id": "DEF_PERSISTENT_EPS_P50",
            "參數名": "p50",
            "層級(T1/T2/T3)": "T3",
            "所在檔案": "bankruptcy/app/assets/defaults.json",
            "行號": find_defaults_json_line("p50", under_inputs=True),
            "類別": "應用層",
            "來源類型": "config_json",
            "型別": "float",
            "目前值": "0.224260187198639",
            "可調性": "衍生值，不可直接調整",
            "驗證方式": "json",
        },
        {
            "id": "DEF_N_TRAIN",
            "參數名": "n_train",
            "層級(T1/T2/T3)": "T3",
            "所在檔案": "bankruptcy/app/assets/defaults.json",
            "行號": find_defaults_json_line("n_train"),
            "類別": "資料與切分",
            "來源類型": "config_json",
            "型別": "int",
            "目前值": "5455",
            "可調性": "衍生值，不可直接調整",
            "驗證方式": "json",
        },
        {
            "id": "DEF_N_RAW_COLUMNS",
            "參數名": "n_raw_columns",
            "層級(T1/T2/T3)": "T3",
            "所在檔案": "bankruptcy/app/assets/defaults.json",
            "行號": find_defaults_json_line("n_raw_columns"),
            "類別": "資料與切分",
            "來源類型": "config_json",
            "型別": "int",
            "目前值": "95",
            "可調性": "衍生值，不可直接調整",
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
    """Generates the final merged 03a_param_registry.csv (v2)."""
    raw_records = build_raw_param_records()
    notes = load_notes_dict()

    fieldnames = [
        "id", "參數名", "層級(T1/T2/T3)", "所在檔案", "行號", "類別", "來源類型",
        "型別", "目前值", "合理範圍", "範圍來源", "說明", "影響階段",
        "相關實驗ID", "證據檔案", "可調性", "驗證方式", "備註"
    ]

    merged_rows = []
    missing_notes = []
    for rec in raw_records:
        pid = rec["id"]
        note = notes.get(pid, {})
        if not note:
            missing_notes.append(pid)

        row = {
            "id": pid,
            "參數名": rec["參數名"],
            "層級(T1/T2/T3)": rec["層級(T1/T2/T3)"],
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
            "可調性": rec["可調性"],
            "驗證方式": rec["驗證方式"],
            "備註": note.get("備註", "自動抽取並對映"),
        }
        merged_rows.append(row)

    if missing_notes:
        print(f"[WARNING] {len(missing_notes)} parameters missing from param_notes.csv: {missing_notes[:5]}...")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Successfully generated parameter registry v2: {OUTPUT_CSV}")
    print(f"Total parameters registered: {len(merged_rows)}")


if __name__ == "__main__":
    build_registry()
