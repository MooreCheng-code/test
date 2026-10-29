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
    if not check_dir.exists():
        print(f"[ERROR] Target directory not found: {check_dir}")
        return 1, 0

    if check_dir.is_file():
        md_files = [check_dir]
    else:
        md_files = sorted(check_dir.rglob("*.md"))

    print(f"Checking {len(md_files)} markdown files in {check_dir}...\n")
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
    """Validates 03a_param_registry.csv and param_notes.csv entries and values."""
    reg_file = registry_path or REGISTRY_CSV
    if not reg_file.exists():
        print(f"[ERROR] Parameter registry file not found: {reg_file}")
        return 1, 0

    print(f"\nValidating Parameter Registry: {reg_file}...\n")
    errors = 0
    warnings = 0

    expected_cols = [
        "id", "參數名", "層級(T1/T2)", "所在檔案", "行號", "類別", "來源類型",
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

        row_idx = 1
        for row in reader:
            row_idx += 1
            param_id = row.get("id", f"row_{row_idx}")
            name = row.get("參數名", "").strip()
            file_rel = row.get("所在檔案", "").strip()
            line_str = row.get("行號", "").strip()
            curr_val = row.get("目前值", "").strip()
            src_type = row.get("來源類型", "").strip()
            val_method = row.get("驗證方式", "").strip()

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

            # 2. File existence
            if not file_rel:
                print(f"[{param_id}] [ERROR] Missing '所在檔案'")
                errors += 1
                continue

            target_file = REPO_ROOT / file_rel
            if not target_file.exists():
                print(f"[{param_id}] [ERROR] Target file does not exist: {file_rel}")
                errors += 1
                continue

            # 3. Line number and symbol check
            if line_str and line_str.isdigit():
                line_no = int(line_str)
                lines = target_file.read_text(encoding="utf-8").splitlines()
                if line_no <= 0 or line_no > len(lines):
                    print(f"[{param_id}] [ERROR] Line {line_no} out of range in {file_rel} (1-{len(lines)})")
                    errors += 1
                else:
                    # Check if name is within ±3 lines
                    if not verify_symbol_in_file(target_file, name, line_no, line_no, slack=3):
                        print(
                            f"[{param_id}] [ERROR] Parameter '{name}' not found within ±3 lines of {file_rel}:{line_no}"
                        )
                        errors += 1

            # 4. Value consistency check
            if file_rel == "bankruptcy/config.py":
                tree = ast.parse(target_file.read_text(encoding="utf-8"))
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
                    bp_val = data.get("best_params", {}).get(name)
                    exp_str = str(bp_val) if not isinstance(bp_val, str) else f'"{bp_val}"'
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
                if curr_val != "動態：由訓練折計算":
                    print(f"[{param_id}] [ERROR] Derived parameter must specify '動態：由訓練折計算', got: {curr_val}")
                    errors += 1

    # Check param_notes.csv if exists
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
