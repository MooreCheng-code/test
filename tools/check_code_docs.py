#!/usr/bin/env python3
"""
tools/check_code_docs.py
Consistency and compliance checker for docs/code_learning/ and parameter registry.

Checks:
1. Citations: file:line references exist, line numbers are valid, and cited symbols
   (functions, classes, constants, variables) appear within ±3 lines of cited lines.
2. Clean paths: No absolute paths, no file:/// URIs, no Windows user directories.
3. Language & Units:
   - No exaggerated/absolutist terms (e.g. 完勝, 唯一正道, 100% 保證, 嚴格無偏).
   - AR / fold standard deviation unit check (must use '倍' or 'x', not '%').
   - Strict 15-fold misnomer check (must use '5 折 × 3 次重複 = 15 次外層評估').
4. Registry mode (--registry):
   - Validates docs/code_learning/03a_param_registry.csv entries.
   - Checks file existence, parameter name within ±3 lines, and value consistency.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "code_learning"
REGISTRY_CSV = DOCS_DIR / "03a_param_registry.csv"
EXCLUSIONS_CSV = DOCS_DIR / "param_exclusions.csv"

VALID_LEVELS = {"T1", "T2", "T3"}
VALID_TUNABILITIES = {
    "可在 lab 安全調整",
    "需重跑固化流程才生效",
    "固定常數，不建議調整",
    "衍生值，不可直接調整",
    "已鎖定，不可調整（E6 相關）",
    "僅應用層",
}

# Regex patterns for citations
# Pattern 1: `symbol` (bankruptcy/src/foo.py:123) or symbol（bankruptcy/src/foo.py:123-145）
CITATION_PATTERN_PAREN = re.compile(
    r"[`'\"“”]?([a-zA-Z_]\w+)[`'\"“”]?\s*[\(\uff08]\s*[`'\"“”]?(bankruptcy/[\w/.-]+\.py):(\d+)(?:-(\d+))?[`'\"“”]?\s*[\)\uff09]"
)

# Pattern 2: `symbol` bankruptcy/src/foo.py:123
CITATION_PATTERN_SPACE = re.compile(
    r"[`]([a-zA-Z_]\w+)[`]\s+[`'\"“”]?(bankruptcy/[\w/.-]+\.py):(\d+)(?:-(\d+))?[`'\"“”]?"
)

# General file:line citation (without explicit symbol)
GENERAL_CITATION = re.compile(
    r"(bankruptcy/[\w/.-]+\.py):(\d+)(?:-(\d+))?"
)

# Forbidden absolute path patterns
ABS_PATH_PATTERNS = [
    re.compile(r"file:///", re.IGNORECASE),
    re.compile(r"[a-zA-Z]:[/\\]Users[/\\]", re.IGNORECASE),
    re.compile(r"/Users/\w+/", re.IGNORECASE),
]

# Forbidden exaggerated / absolutist terms
FORBIDDEN_WORDS = [
    "完勝",
    "唯一正道",
    "唯一正解",
    "100% 保證",
    "100%保證",
    "絕無可能失敗",
    "嚴格無偏",  # Requested by user in Stage 3 instruction
]

# Strict 15-fold misnomer (must use 5 折 × 3 次重複 = 15 次外層評估)
MISNOMER_15FOLD = re.compile(r"(?<![35\d])15\s*折(?!次)")


def verify_symbol_in_file(
    file_path: Path,
    symbol: str,
    start_line: int,
    end_line: int,
    slack: int = 3,
) -> bool:
    """Verifies that symbol appears within [start_line - slack, end_line + slack]."""
    if not file_path.exists():
        return False
    lines = file_path.read_text(encoding="utf-8").splitlines()
    n_lines = len(lines)
    s_chk = max(1, start_line - slack)
    e_chk = min(n_lines, end_line + slack)

    pattern = re.compile(r"\b" + re.escape(symbol) + r"\b")
    for i in range(s_chk - 1, e_chk):
        if pattern.search(lines[i]):
            return True
    return False


def check_markdown_file(md_path: Path) -> tuple[int, int]:
    """Checks a single markdown file for paths, symbols, citations, and terminology."""
    errors = 0
    warnings = 0
    try:
        rel_md = md_path.relative_to(REPO_ROOT)
    except ValueError:
        rel_md = md_path

    content = md_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    for idx, line in enumerate(lines, 1):
        # 1. Absolute paths
        for pat in ABS_PATH_PATTERNS:
            if pat.search(line):
                print(f"[{rel_md}:{idx}] [ERROR] Absolute or user path detected: {line.strip()[:80]}")
                errors += 1

        # 2. Forbidden words
        for word in FORBIDDEN_WORDS:
            if word in line:
                # Allow quoting the forbidden word in negative check instructions or findings
                if "不得使用" not in line and "移除" not in line and md_path.name != "_findings.md":
                    print(f"[{rel_md}:{idx}] [ERROR] Forbidden term '{word}': {line.strip()[:80]}")
                    errors += 1

        # 3. 15-fold misnomer
        if md_path.name != "_findings.md":
            if MISNOMER_15FOLD.search(line) and "絕不能稱為" not in line and "嚴禁" not in line:
                print(f"[{rel_md}:{idx}] [WARNING] Misnomer '15 折' detected (use '5 折 × 3 次重複 = 15 次外層評估'): {line.strip()[:80]}")
                warnings += 1

        # 4. General line bounds check
        for m in GENERAL_CITATION.finditer(line):
            target_rel, s_str, e_str = m.group(1), m.group(2), m.group(3)
            s_line = int(s_str)
            e_line = int(e_str) if e_str else s_line
            target_file = REPO_ROOT / target_rel
            if not target_file.exists():
                print(f"[{rel_md}:{idx}] [ERROR] Cited file does not exist: {target_rel}")
                errors += 1
                continue
            t_lines = target_file.read_text(encoding="utf-8").splitlines()
            if s_line > len(t_lines) or s_line <= 0:
                print(f"[{rel_md}:{idx}] [ERROR] Cited line {s_line} out of range in {target_rel} (1-{len(t_lines)})")
                errors += 1

        # 5. Symbol in citation check
        seen_citations = set()
        for pat in [CITATION_PATTERN_PAREN, CITATION_PATTERN_SPACE]:
            for m in pat.finditer(line):
                symbol = m.group(1)
                target_rel = m.group(2)
                s_line = int(m.group(3))
                e_line = int(m.group(4)) if m.group(4) else s_line
                key = (symbol, target_rel, s_line, e_line)
                if key in seen_citations:
                    continue
                seen_citations.add(key)

                # Skip non-code words like 'MD', 'CSV', etc. if not identifier
                target_file = REPO_ROOT / target_rel
                if not target_file.exists():
                    print(f"[{rel_md}:{idx}] [ERROR] Cited file does not exist: {target_rel}")
                    errors += 1
                    continue

                if not verify_symbol_in_file(target_file, symbol, s_line, e_line, slack=3):
                    print(
                        f"[{rel_md}:{idx}] [ERROR] Symbol '{symbol}' not found within ±3 lines of {target_rel}:{s_line}-{e_line}"
                    )
                    errors += 1

    return errors, warnings


def check_docs(target_dir: Path | None = None) -> tuple[int, int]:
    """Scans all markdown files in target directory."""
    check_dir = target_dir or DOCS_DIR
    rel_dir = str(check_dir.relative_to(REPO_ROOT) if check_dir.is_absolute() else check_dir).replace("\\", "/")
    if not check_dir.exists():
        print(f"[ERROR] Target directory not found: {rel_dir}")
        return 1, 0

    if check_dir.is_file():
        md_files = [check_dir]
    else:
        md_files = sorted(check_dir.rglob("*.md"))

    print(f"Checking {len(md_files)} markdown files in {rel_dir}...\n")
    total_errors = 0
    total_warnings = 0

    for md_path in md_files:
        errs, warns = check_markdown_file(md_path)
        total_errors += errs
        total_warnings += warns

    print("\n" + "=" * 50)
    print(f"Markdown check completed: {total_errors} error(s), {total_warnings} warning(s).")
    print("=" * 50)
    return total_errors, total_warnings


def check_param_registry(registry_path: Path | None = None) -> tuple[int, int]:
    """Validates 03a_param_registry.csv, param_notes.csv, and checks completeness against codebase."""
    reg_file = registry_path or REGISTRY_CSV
    try:
        rel_reg = str(reg_file.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        rel_reg = reg_file.name
    if not reg_file.exists():
        print(f"[ERROR] Parameter registry file not found: {rel_reg}")
        return 1, 0

    print(f"\nValidating Parameter Registry: {rel_reg}...\n")
    errors = 0
    warnings = 0

    expected_cols = [
        "id", "參數名", "層級(T1/T2/T3)", "所在檔案", "行號", "類別", "來源類型",
        "型別", "目前值", "合理範圍", "範圍來源", "說明", "影響階段",
        "相關實驗ID", "證據檔案", "可調性", "驗證方式", "備註"
    ]

    with open(reg_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        # Check required columns
        for col in expected_cols:
            if col not in header:
                print(f"[ERROR] Registry CSV missing required column: '{col}'")
                errors += 1

        rows = list(reader)

    file_trees: dict[str, ast.AST] = {}

    row_idx = 1
    for row in rows:
        row_idx += 1
        param_id = row.get("id", f"row_{row_idx}")
        name = row.get("參數名", "").strip()
        level = row.get("層級(T1/T2/T3)", "").strip()
        file_rel = row.get("所在檔案", "").strip().replace("\\", "/")
        line_str = row.get("行號", "").strip()
        curr_val = row.get("目前值", "").strip()
        src_type = row.get("來源類型", "").strip()
        tunability = row.get("可調性", "").strip()
        note = row.get("備註", "").strip()
        scope = row.get("合理範圍", "").strip()
        desc = row.get("說明", "").strip()
        evid = row.get("證據檔案", "").strip()

        # 1. Clean paths and forbidden words in registry fields
        for k, val in row.items():
            if not isinstance(val, str):
                continue
            for pat in ABS_PATH_PATTERNS:
                if pat.search(val):
                    print(f"[{param_id}] [ERROR] Absolute path in column '{k}': {val}")
                    errors += 1
            for word in FORBIDDEN_WORDS:
                if word in val:
                    print(f"[{param_id}] [ERROR] Forbidden word '{word}' in column '{k}': {val}")
                    errors += 1

        # 2. Level enum check
        if level not in VALID_LEVELS:
            print(f"[{param_id}] [ERROR] Invalid level '{level}', expected one of {VALID_LEVELS}")
            errors += 1

        # 3. Tunability enum check
        if tunability not in VALID_TUNABILITIES:
            print(f"[{param_id}] [ERROR] Invalid tunability '{tunability}', expected one of {VALID_TUNABILITIES}")
            errors += 1
        if file_rel == "bankruptcy/config.py" and name.endswith(("_DIR", "_PATH")):
            if tunability == "已鎖定，不可調整（E6 相關）":
                print(f"[{param_id}] [ERROR] Path constant '{name}' must not be marked '已鎖定，不可調整（E6 相關）'")
                errors += 1

        # 4. File existence
        if not file_rel:
            print(f"[{param_id}] [ERROR] Missing '所在檔案'")
            errors += 1
            continue

        target_file = REPO_ROOT / file_rel
        if not target_file.exists():
            print(f"[{param_id}] [ERROR] Target file does not exist: {file_rel}")
            errors += 1
            continue

        # 5. AST & Line verification
        if line_str and line_str.isdigit():
            line_no = int(line_str)
            if target_file.suffix in [".joblib", ".pkl"]:
                if param_id == "M5_FINAL_MODEL_SUBSAMPLE_FREQ":
                    try:
                        b_path = str((REPO_ROOT / "bankruptcy").resolve())
                        if b_path not in sys.path:
                            sys.path.insert(0, b_path)
                        import joblib
                        m_obj = joblib.load(target_file)
                        clf = m_obj.named_steps["classifier"] if hasattr(m_obj, "named_steps") and "classifier" in m_obj.named_steps else m_obj
                        actual_freq = clf.get_params().get("subsample_freq")
                        if str(actual_freq) != curr_val:
                            print(f"[{param_id}] [ERROR] M5 final model subsample_freq mismatch: expected {actual_freq}, got {curr_val}")
                            errors += 1
                    except Exception as e:
                        print(f"[{param_id}] [ERROR] Failed to load joblib: {e}")
                        errors += 1
            else:
                lines = target_file.read_text(encoding="utf-8").splitlines()
                if line_no <= 0 or line_no > len(lines):
                    print(f"[{param_id}] [ERROR] Line {line_no} out of range in {file_rel} (1-{len(lines)})")
                    errors += 1
                elif curr_val == "無此鍵":
                    pass
                else:
                    if src_type == "search_space":
                        if file_rel not in file_trees:
                            file_trees[file_rel] = ast.parse(target_file.read_text(encoding="utf-8"))
                        tree = file_trees[file_rel]
                        found_suggest = False
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("suggest_"):
                                if abs(getattr(node, "lineno", 0) - line_no) <= 3:
                                    if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == name:
                                        found_suggest = True
                                        break
                        if not found_suggest:
                            print(f"[{param_id}] [ERROR] Parameter '{name}' not found in suggest_* call AST around {file_rel}:{line_no}")
                            errors += 1
                    else:
                        if not verify_symbol_in_file(target_file, name, line_no, line_no, slack=3):
                            print(
                                f"[{param_id}] [ERROR] Parameter '{name}' not found within ±3 lines of {file_rel}:{line_no}"
                            )
                            errors += 1

        # 6. Value consistency check
        if file_rel == "bankruptcy/config.py":
            if file_rel not in file_trees:
                file_trees[file_rel] = ast.parse(target_file.read_text(encoding="utf-8"))
            tree = file_trees[file_rel]
            found_cfg = False
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == name:
                            try:
                                v = ast.literal_eval(node.value)
                                exp_str = str(v) if not isinstance(v, str) else f'"{v}"'
                            except Exception:
                                exp_str = ast.unparse(node.value)
                            if exp_str != curr_val:
                                print(f"[{param_id}] [ERROR] Value mismatch in config: expected {exp_str}, got {curr_val}")
                                errors += 1
                            found_cfg = True
            if not found_cfg:
                print(f"[{param_id}] [ERROR] Constant '{name}' not found in config.py AST")
                errors += 1

        elif src_type == "best_params_json":
            m = param_id.split("_")[0]
            bp_file = REPO_ROOT / f"bankruptcy/outputs/models/best_params/{m}.json"
            if bp_file.exists():
                data = json.load(open(bp_file, encoding="utf-8"))
                if curr_val == "無此鍵":
                    if name in data.get("best_params", {}) or name in data:
                        print(f"[{param_id}] [ERROR] Expected key '{name}' to NOT exist, but found in {bp_file.name}")
                        errors += 1
                elif name == "imbalance":
                    bp_val = data.get("imbalance")
                    exp_str = f'"{bp_val}"' if isinstance(bp_val, str) else str(bp_val)
                    if exp_str != curr_val:
                        print(f"[{param_id}] [ERROR] Best params imbalance mismatch: expected {exp_str}, got {curr_val}")
                        errors += 1
                else:
                    bp_val = data.get("best_params", {}).get(name)
                    exp_str = f'"{bp_val}"' if isinstance(bp_val, str) else str(bp_val)
                    if exp_str != curr_val:
                        print(f"[{param_id}] [ERROR] Best params value mismatch: expected {exp_str}, got {curr_val}")
                        errors += 1

        elif src_type == "config_json" and "thresholds.json" in file_rel:
            parts = param_id.split("_")
            grp = f"{parts[1]}_{parts[2]}" if parts[1] == "E" else parts[1]
            r_key = parts[3] if parts[1] == "E" else parts[2]
            field = "threshold" if param_id.endswith("THRESH") else "recommended_margin"
            t_data = json.load(open(target_file, encoding="utf-8"))
            exp_val = t_data.get(grp, {}).get(r_key, {}).get(field)
            if str(exp_val) != curr_val:
                print(f"[{param_id}] [ERROR] Threshold value mismatch: expected {exp_val}, got {curr_val}")
                errors += 1

        elif src_type == "config_json" and "defaults.json" in file_rel:
            d_data = json.load(open(target_file, encoding="utf-8"))
            if param_id == "DEF_PERSISTENT_EPS_P50":
                exp_val = d_data.get("inputs", {}).get("Persistent EPS in the Last Four Seasons", {}).get("p50")
            else:
                exp_val = d_data.get("meta", {}).get(name)
            if str(exp_val) != curr_val:
                print(f"[{param_id}] [ERROR] Defaults value mismatch: expected {exp_val}, got {curr_val}")
                errors += 1

        elif src_type == "derived":
            if curr_val not in ["動態：由訓練折計算", "1 if subsample < 1.0 else 0", "0", "無此鍵"]:
                print(f"[{param_id}] [ERROR] Derived parameter unexpected value: {curr_val}")
                errors += 1

        # 7. Note numbers sanity check (Rule 2.2)
        clean_note = re.sub(r'\b\w+\.py:\d+(?:-\d+)?', '', note)
        clean_note = re.sub(r'\b[FSMRE]_\w+', '', clean_note)
        clean_note = re.sub(r'\b(?:M[0-5]|S[0-6]|R80|R90|R95|R100|F-02|Rule 2)\b', '', clean_note)
        nums = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?', clean_note)
        for n in nums:
            raw_n = n.replace(',', '')
            found = False
            for field in [curr_val, scope, desc, evid, note]:
                if raw_n in field or n in field:
                    found = True
                    break
            if not found and n.endswith('%'):
                pct_val = float(raw_n[:-1]) / 100.0
                val_str = f"{pct_val:.1f}" if pct_val * 10 % 1 == 0 else f"{pct_val}"
                if val_str in curr_val or val_str in scope or val_str in desc:
                    found = True
            if not found:
                warnings += 1

    # 8. Check param_notes.csv if exists
    notes_csv = DOCS_DIR / "param_notes.csv"
    if notes_csv.exists():
        with open(notes_csv, "r", encoding="utf-8-sig") as f:
            n_reader = csv.DictReader(f)
            n_row = 1
            for row in n_reader:
                n_row += 1
                pid = row.get("id", f"note_row_{n_row}")
                for k, val in row.items():
                    if not isinstance(val, str):
                        continue
                    for pat in ABS_PATH_PATTERNS:
                        if pat.search(val):
                            print(f"[param_notes:{pid}] [ERROR] Absolute path in column '{k}': {val}")
                            errors += 1
                    for word in FORBIDDEN_WORDS:
                        if word in val:
                            print(f"[param_notes:{pid}] [ERROR] Forbidden word '{word}' in column '{k}': {val}")
                            errors += 1

    # 9. Completeness Check (Rule 2.3)
    unmatched = []
    reg_names = set(r["參數名"] for r in rows)
    ex_map = set()
    ex_count = 0
    if EXCLUSIONS_CSV.exists():
        with open(EXCLUSIONS_CSV, "r", encoding="utf-8-sig") as f:
            for ex in csv.DictReader(f):
                ex_count += 1
                ex_map.add((ex["來源"], ex["名稱"]))
                ex_map.add(ex["名稱"])

    # config.py
    cfg_tree = ast.parse((REPO_ROOT / "bankruptcy/config.py").read_text(encoding="utf-8"))
    for node in cfg_tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    if t.id not in reg_names and t.id not in ex_map:
                        unmatched.append(("bankruptcy/config.py", t.id))

    # tune.py suggest_*
    tune_tree = ast.parse((REPO_ROOT / "bankruptcy/src/tune.py").read_text(encoding="utf-8"))
    for node in ast.walk(tune_tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("suggest_"):
                if node.args and isinstance(node.args[0], ast.Constant):
                    p_name = node.args[0].value
                    if p_name not in reg_names and p_name not in ex_map:
                        unmatched.append(("bankruptcy/src/tune.py", p_name))

    # pipelines.py:build_pipeline defaults
    pipe_tree = ast.parse((REPO_ROOT / "bankruptcy/src/pipelines.py").read_text(encoding="utf-8"))
    for node in pipe_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_pipeline":
            args = node.args
            diff = len(args.args) - len(args.defaults)
            for idx, d in enumerate(args.defaults):
                arg_name = args.args[diff + idx].arg
                src = "pipelines.py:build_pipeline"
                if arg_name not in reg_names and (src, arg_name) not in ex_map:
                    unmatched.append((src, arg_name))

    # preprocessing.py transformer __init__ defaults
    prep_tree = ast.parse((REPO_ROOT / "bankruptcy/src/preprocessing.py").read_text(encoding="utf-8"))
    for node in prep_tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    diff = len(item.args.args) - len(item.args.defaults)
                    for idx, d in enumerate(item.args.defaults):
                        arg_name = item.args.args[diff + idx].arg
                        src = f"preprocessing.py:{node.name}"
                        if arg_name not in reg_names and (src, arg_name) not in ex_map:
                            unmatched.append((src, arg_name))

    # best_params/*.json
    for p in sorted((REPO_ROOT / "bankruptcy/outputs/models/best_params").glob("*.json")):
        d = json.load(open(p, encoding="utf-8"))
        for k in d.keys():
            if k == "best_params":
                for hp in d["best_params"].keys():
                    if hp not in reg_names and hp not in ex_map:
                        unmatched.append((f"{p.name}:best_params", hp))
            elif k in ["model", "inner_best_loss", "study_name", "created_at"]:
                if ("outputs/models/best_params/*.json", k) not in ex_map and k not in ex_map:
                    unmatched.append((f"{p.name}:top", k))
            elif k == "imbalance":
                if "imbalance" not in reg_names:
                    unmatched.append((f"{p.name}:top", k))
            elif k == "n_trials":
                pass

    # thresholds.json
    th_data = json.load(open(REPO_ROOT / "bankruptcy/outputs/models/final/thresholds.json", encoding="utf-8"))
    for grp, r_dict in th_data.items():
        for r_k, fields in r_dict.items():
            for field in fields.keys():
                if field in ["threshold", "recommended_margin"]:
                    if field not in reg_names:
                        unmatched.append(("thresholds.json", field))
                elif field == "target_recall":
                    pass
                else:
                    if ("outputs/models/final/thresholds.json", field) not in ex_map and field not in ex_map:
                        unmatched.append(("thresholds.json", field))

    # defaults.json
    def_data = json.load(open(REPO_ROOT / "bankruptcy/app/assets/defaults.json", encoding="utf-8"))
    for k in def_data.keys():
        if k in ["meta", "inputs"]:
            continue
        if ("app/assets/defaults.json:top", k) not in ex_map and k not in ex_map:
            unmatched.append(("defaults.json:top", k))
    for k in def_data.get("meta", {}).keys():
        if k not in reg_names and ("app/assets/defaults.json:meta", k) not in ex_map and k not in ex_map:
            unmatched.append(("defaults.json:meta", k))
    for k in def_data.get("inputs", {}).keys():
        if k == "Persistent EPS in the Last Four Seasons":
            pass
        else:
            if ("app/assets/defaults.json:inputs", "other_17_features") not in ex_map:
                unmatched.append(("defaults.json:inputs", k))

    if unmatched:
        print(f"[ERROR] Completeness check failed! {len(unmatched)} unmatched parameter(s):")
        for u in unmatched:
            print(f"  Unmatched: {u}")
        errors += len(unmatched)
    else:
        print(f"Completeness check passed: 0 unmatched items across all sources (param_exclusions.csv: {ex_count} items).")

    print("\n" + "=" * 50)
    print(f"Registry check completed: {errors} error(s), {warnings} warning(s).")
    print("=" * 50)
    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Consistency and compliance checker.")
    parser.add_argument("path", nargs="?", default=None, help="Specific file or directory to check.")
    parser.add_argument("--registry", action="store_true", help="Run parameter registry check.")
    parser.add_argument("--registry-file", type=str, default=None, help="Custom registry CSV to check.")
    args = parser.parse_args()

    total_err = 0
    total_warn = 0

    if args.registry or args.registry_file:
        reg_path = Path(args.registry_file) if args.registry_file else None
        e, w = check_param_registry(reg_path)
        total_err += e
        total_warn += w
    else:
        target = Path(args.path) if args.path else None
        e, w = check_docs(target)
        total_err += e
        total_warn += w

    sys.exit(1 if total_err > 0 else 0)


if __name__ == "__main__":
    main()
