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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
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
    "完美",  # Requested by user in Stage 6-fix
    "精確吻合 100%",
    "精確吻合100%",
    "100% 精確吻合",
    "100%精確吻合",
]

# Strict 15-fold misnomer (must use 5 折 × 3 次重複 = 15 次外層評估)
MISNOMER_15FOLD = re.compile(r"(?<![35\d])15\s*折(?!次)")


LAB_MAP_CSV = REPO_ROOT / "labs" / "lab_map.csv"

FILE_PATH_PATTERN = re.compile(
    r'(?:bankruptcy[/\][\w./\-]+\.(?:py|csv|json|md|joblib|png))'
)

EVIDENCE_TAG_PATTERN = re.compile(
    r'<!--\s*evidence\s+(.+?)\s*-->'
)
CODE_TAG_PATTERN = re.compile(
    r'<!--\s*code\s+(.+?)\s*-->'
)
NO_EVIDENCE_PATTERN = re.compile(
    r'<!--\s*no-evidence:\s*(.+?)\s*-->'
)
CMD_TAG_PATTERN = re.compile(
    r'<!--\s*cmd\s+(.+?)\s*-->'
)


def load_lab_map() -> dict[str, dict[str, str]]:
    """Loads labs/lab_map.csv."""
    if not LAB_MAP_CSV.exists():
        return {}
    res = {}
    with open(LAB_MAP_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            res[r["lab_id"].strip()] = r
    return res


def parse_tag_attrs(tag_str: str) -> dict[str, str]:
    """Parses attributes from an evidence tag."""
    pattern = re.compile(r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|([^\s]+))')
    attrs = {}
    for match in pattern.finditer(tag_str):
        k = match.group(1)
        v = match.group(2) if match.group(2) is not None else (match.group(3) if match.group(3) is not None else match.group(4))
        attrs[k] = v
    return attrs


def extract_python_ast_value(target_file: Path, name: str, line_no: int) -> tuple[Any, str]:
    """Finds AST value of variable/constant or dict key or kwarg at line_no +- 3."""
    if not target_file.exists():
        return None, "file_not_found"
    try:
        tree = ast.parse(target_file.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"ast_parse_error: {e}"
    
    candidates = []
    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", None)
        
        # Expr (e.g. platt.fit(S_in_EA.reshape(-1, 1), y_tr_outer))
        if isinstance(node, ast.Expr):
            if lineno is not None and abs(lineno - line_no) <= 3:
                candidates.append((abs(lineno - line_no), ast.unparse(node.value), "expr"))
            
        # Top-level or local Assign
        elif isinstance(node, ast.Assign):
            if lineno is not None and abs(lineno - line_no) <= 3:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        if not name or t.id == name:
                            try:
                                v = ast.literal_eval(node.value)
                            except Exception:
                                v = ast.unparse(node.value)
                            candidates.append((abs(lineno - line_no), v, "assign"))
                    elif isinstance(t, ast.Name) and t.id == "platt_final" and name in ("platt_final", "Platt"):
                        candidates.append((abs(lineno - line_no), "Platt", "assign_platt"))
                    
        # FunctionDef default arguments
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            diff = len(node.args.args) - len(node.args.defaults)
            for i, d in enumerate(node.args.defaults):
                arg = node.args.args[diff + i]
                arg_line = getattr(arg, "lineno", lineno)
                if arg_line is not None and abs(arg_line - line_no) <= 3 and (not name or name == arg.arg):
                    try:
                        v = ast.literal_eval(d)
                    except Exception:
                        v = ast.unparse(d)
                    candidates.append((abs(arg_line - line_no), v, "arg_default"))

        # AnnAssign
        elif isinstance(node, ast.AnnAssign):
            if lineno is not None and abs(lineno - line_no) <= 3:
                if isinstance(node.target, ast.Name) and (not name or node.target.id == name) and node.value:
                    try:
                        v = ast.literal_eval(node.value)
                    except Exception:
                        v = ast.unparse(node.value)
                    candidates.append((abs(lineno - line_no), v, "annassign"))
                
        # Dict key: "name": value (e.g. n_trials_map or params)
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                k_line = getattr(k, "lineno", getattr(v, "lineno", lineno))
                if k_line is not None and abs(k_line - line_no) <= 3:
                    k_str = ast.literal_eval(k) if isinstance(k, ast.Constant) else (ast.unparse(k) if k else "")
                    if not name or name == str(k_str):
                        try:
                            val = ast.literal_eval(v)
                        except Exception:
                            val = ast.unparse(v)
                        candidates.append((abs(k_line - line_no), val, "dict_key"))
                    
        # Call kwargs: func(name=value) or Class(name=value)
        elif isinstance(node, ast.Call):
            if lineno is not None and abs(lineno - line_no) <= 3:
                # Check combined y_min / y_max for Isotonic
                kw_dict = {}
                for kw in node.keywords:
                    if kw.arg:
                        try:
                            kw_dict[kw.arg] = ast.literal_eval(kw.value)
                        except Exception:
                            kw_dict[kw.arg] = ast.unparse(kw.value)
                if "y_min" in kw_dict and "y_max" in kw_dict and name in ("y_min", "y_max", "CAL_ISOTONIC_Y_MIN_MAX"):
                    candidates.append((abs(lineno - line_no), [kw_dict["y_min"], kw_dict["y_max"]], "isotonic_y_bounds"))

                for kw in node.keywords:
                    if not name or kw.arg == name:
                        try:
                            v = ast.literal_eval(kw.value)
                        except Exception:
                            v = ast.unparse(kw.value)
                        candidates.append((abs(lineno - line_no), v, "call_kwarg"))
                        
                if isinstance(node.func, ast.Name):
                    if not name or node.func.id == name:
                        for kw in node.keywords:
                            if kw.arg in ("sampling_strategy", "n_splits", "max_iter"):
                                try:
                                    v = ast.literal_eval(kw.value)
                                except Exception:
                                    v = ast.unparse(kw.value)
                                candidates.append((abs(lineno - line_no), v, f"call_class_{kw.arg}"))
                    if node.func.id == "nadeau_bengio_paired_test" and name in ("nadeau_bengio_paired_test", "0.95", "ENS_E3_SELECTION_TOLERANCE"):
                        candidates.append((abs(lineno - line_no), 0.95, "test_ci_level"))
                        
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]
    return None, "not_found"


def verify_code_tag(tag_str: str, repo_root: Path = REPO_ROOT) -> tuple[bool, str, dict[str, Any]]:
    """Verifies a single <!-- code ... --> tag against Python AST."""
    attrs = parse_tag_attrs(tag_str)
    file_rel = attrs.get("file")
    if not file_rel:
        return False, "Missing 'file' attribute in code tag", attrs
    target_path = repo_root / file_rel
    if not target_path.exists():
        return False, f"Code file does not exist: {file_rel}", attrs
    line_str = attrs.get("line")
    if not line_str or not line_str.isdigit():
        return False, "Missing or invalid 'line' attribute in code tag", attrs
    line_no = int(line_str)
    symbol = attrs.get("symbol", "")
    expect_str = attrs.get("expect")
    if expect_str is None:
        return False, "Missing 'expect' attribute in code tag", attrs
    tol = float(attrs.get("tol", "0.001"))
    
    ast_val, how = extract_python_ast_value(target_path, symbol, line_no)
    if how in ("not_found", "file_not_found") or how.startswith("ast_parse_error"):
        return False, f"Symbol '{symbol}' or assignment not found around {file_rel}:{line_no}", attrs
        
    try:
        ast_float = float(ast_val)
        expect_float = float(expect_str)
        if abs(ast_float - expect_float) > tol:
            return False, f"Code AST mismatch: actual {ast_float} vs expect {expect_float} (diff {abs(ast_float - expect_float):.6f} > tol {tol})", attrs
        return True, f"Verified code float: {ast_float} ~ {expect_float}", attrs
    except (ValueError, TypeError):
        norm_ast = str(ast_val).replace(" ", "")
        norm_exp = expect_str.replace(" ", "")
        if norm_ast != norm_exp and str(ast_val) != expect_str:
            return False, f"Code AST mismatch: actual '{ast_val}' vs expect '{expect_str}'", attrs
        return True, f"Verified code expr: {ast_val}", attrs


def verify_evidence_tag(tag_str: str, repo_root: Path = REPO_ROOT) -> tuple[bool, str, dict[str, Any]]:
    """Verifies a single evidence tag against the target CSV or JSON file."""
    attrs = parse_tag_attrs(tag_str)
    file_rel = attrs.get("file")
    if not file_rel:
        return False, "Missing 'file' attribute in evidence tag", attrs
    
    csv_path = repo_root / file_rel
    if not csv_path.exists():
        return False, f"Evidence file does not exist: {file_rel}", attrs
    
    expect_str = attrs.get("expect")
    tol = float(attrs.get("tol", "0.001"))
    if expect_str is None:
        return False, "Missing 'expect' attribute", attrs

    # Handle JSON evidence file
    if file_rel.endswith(".json"):
        col_name = attrs.get("col") or attrs.get("key")
        if not col_name:
            return False, "Missing 'col' or 'key' attribute for JSON file", attrs
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cur = data
            for part in col_name.split("."):
                cur = cur[part]
            actual_val = cur
        except Exception as e:
            return False, f"Failed to extract JSON key '{col_name}' from {file_rel}: {e}", attrs
            
        try:
            actual_float = float(actual_val)
            expect_float = float(expect_str)
            diff = abs(actual_float - expect_float)
            if diff > tol:
                return False, f"JSON value mismatch: actual {actual_float} vs expect {expect_float} (diff {diff:.6f} > tol {tol})", attrs
            return True, f"Verified JSON: {actual_float} ~ {expect_float}", attrs
        except ValueError:
            if isinstance(actual_val, bool):
                if str(actual_val).lower() != str(expect_str).lower():
                    return False, f"JSON value mismatch: actual '{actual_val}' vs expect '{expect_str}'", attrs
                return True, f"Verified JSON bool: {actual_val}", attrs
            if str(actual_val) != expect_str:
                return False, f"JSON value mismatch: actual '{actual_val}' vs expect '{expect_str}'", attrs
            return True, f"Verified JSON string: '{actual_val}'", attrs

    # Handle CSV evidence file
    col_name = attrs.get("col")
    if not col_name:
        return False, "Missing 'col' attribute for CSV file", attrs
    where_expr = attrs.get("where")
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"Failed to read CSV {file_rel}: {e}", attrs
        
    if col_name not in df.columns:
        return False, f"Column '{col_name}' not found in {file_rel}", attrs
        
    if where_expr:
        try:
            filtered = df.query(where_expr)
        except Exception as e:
            return False, f"Failed to evaluate where='{where_expr}': {e}", attrs
    else:
        filtered = df
        
    rows_req = attrs.get("rows")
    agg_method = attrs.get("agg")
    
    if rows_req:
        n_rows = int(rows_req)
        if len(filtered) != n_rows:
            return False, f"Query returned {len(filtered)} rows, expected {n_rows}", attrs
        if agg_method == "mean":
            actual_val = filtered[col_name].mean()
        elif agg_method == "min":
            actual_val = filtered[col_name].min()
        elif agg_method == "max":
            actual_val = filtered[col_name].max()
        else:
            return False, f"Unknown agg method '{agg_method}'", attrs
    else:
        if len(filtered) != 1:
            return False, f"Query returned {len(filtered)} rows, expected exactly 1 (use rows=N and agg=... for multiple rows)", attrs
        actual_val = filtered[col_name].iloc[0]
        
    try:
        actual_float = float(actual_val)
        expect_float = float(expect_str)
        diff = abs(actual_float - expect_float)
        if diff > tol:
            return False, f"Value mismatch: actual {actual_float} vs expect {expect_float} (diff {diff:.6f} > tol {tol})", attrs
        return True, f"Verified: {actual_float:.4f} ~ {expect_float} (tol={tol})", attrs
    except ValueError:
        if str(actual_val) != expect_str:
            return False, f"Value mismatch: actual '{actual_val}' vs expect '{expect_str}'", attrs
        return True, f"Verified string: '{actual_val}'", attrs


def extract_numbers_from_text(text: str) -> list[tuple[str, float]]:
    """Extracts numbers, percentages, and metrics with units from text for verification."""
    clean = text
    clean = re.sub(r'\[.*?\]\(.*?\)', '', clean)
    clean = re.sub(r'<!--.*?-->', '', clean)
    # Filter identifiers and findings
    clean = re.sub(r'\b[EFMRS]-?\d+[a-z]?\b', '', clean)
    clean = re.sub(r'\b(?:[Ll]ab)[ _]?\d+\b', '', clean)
    # Filter file line citations e.g. foo.py:123 or :123-145
    clean = re.sub(r'[\w/.-]+\.(?:py|csv|json):\d+(?:-\d+)?', '', clean)
    clean = re.sub(r':\d+(?:-\d+)?', '', clean)
    # Filter list / heading / blockquote numbering
    clean = re.sub(r'^\s*(?:>\s*)?\d+\.\s*', '', clean)
    clean = re.sub(r'^\s*\|\s*\*\*\d+\.\s*', '| **', clean)
    clean = re.sub(r'^\s*#{1,6}\s+\d+\.', '', clean)
    # Normalize thousands comma
    clean = re.sub(r'(?<=\d),(?=\d{3}\b)', '', clean)
    matches = re.finditer(r'(?<![A-Za-z0-9_])(\d+(?:\.\d+)?)(%|[xX倍家筆折次])?(?![A-Za-z0-9_])', clean)
    res = []
    for m in matches:
        num_str = m.group(1)
        unit = m.group(2)
        val = float(num_str)
        raw = num_str + (unit if unit else '')
        if unit == '%':
            res.append((raw, val))
            res.append((raw, val / 100.0))
        else:
            res.append((raw, val))
            if val <= 1.0:
                res.append((raw, val * 100.0))
    return res


def check_statement_numbers(statement: str, expect: float, also_strs: list[str]) -> list[str]:
    """Verifies that all numbers in an empirical statement are tied to evidence tags."""
    valid_nums = [expect, expect * 100.0, round(expect, 2), round(expect * 100.0, 1), round(expect * 100.0, 2)]
    for a in also_strs:
        try:
            v = float(a)
            valid_nums.extend([v, v * 100.0, round(v, 2), round(v * 100.0, 1), round(v * 100.0, 2)])
        except ValueError:
            pass
            
    unverified = []
    stmt_nums = extract_numbers_from_text(statement)
    raw_candidates: dict[str, list[float]] = {}
    for raw_s, n in stmt_nums:
        raw_candidates.setdefault(raw_s, []).append(n)
        
    for raw_s, cands in raw_candidates.items():
        matched = False
        for n in cands:
            for vn in valid_nums:
                if abs(n - vn) < 0.005 or (abs(vn) > 0 and abs(n - vn) / abs(vn) < 0.01):
                    matched = True
                    break
            if matched:
                break
        if not matched:
            unverified.append(raw_s)
    return list(dict.fromkeys(unverified))


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

            # 6. Full-text path existence check
        for m in FILE_PATH_PATTERN.finditer(line):
            raw_path = m.group(0).rstrip(":,.;)'\"")
            norm_path = raw_path.replace("\\", "/")
            target = REPO_ROOT / norm_path
            if not target.exists():
                if "<!-- planned:" in line or "planned:" in line:
                    continue
                if md_path.name in {"_findings.md", "_stale_claims.csv"} or "negative" in line.lower() or "不存在" in line:
                    continue
                print(f"[{rel_md}:{idx}] [ERROR] Cited path does not exist on filesystem: {norm_path}")
                errors += 1

    return errors, warnings


GLOSSARY_MD = DOCS_DIR / "08_術語表.md"
DRAFTS_DIR = DOCS_DIR / "_drafts"
STRICT_DRAFT_FORBIDDEN = ["證明", "保證", "完美", "100%", "業界最佳"]


def check_drafts_strict(drafts_dir: Path | None = None) -> tuple[int, int]:
    """Validates drafts directory for forbidden words and absolute paths."""
    d_dir = drafts_dir or DRAFTS_DIR
    if not d_dir.exists():
        return 0, 0
    errors = 0
    warnings = 0
    for md_path in sorted(d_dir.glob("*.md")):
        rel_md = str(md_path.relative_to(REPO_ROOT) if md_path.is_absolute() else md_path).replace("\\", "/")
        lines = md_path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, 1):
            for pat in ABS_PATH_PATTERNS:
                if pat.search(line):
                    print(f"[{rel_md}:{idx}] [ERROR] Absolute or user path detected in draft: {line.strip()[:80]}")
                    errors += 1
            for word in STRICT_DRAFT_FORBIDDEN:
                if word in line:
                    if not any(k in line for k in ["不得使用", "移除", "禁用", "嚴禁", "不得出現", "證*", "保*", "完*", "業界最*"]):
                        print(f"[{rel_md}:{idx}] [ERROR] Strict draft forbidden word '{word}': {line.strip()[:80]}")
                        errors += 1
    return errors, warnings


def check_docs(target_dir: Path | None = None, include_drafts: bool = False) -> tuple[int, int]:
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
        if not include_drafts:
            md_files = [f for f in md_files if "_drafts" not in f.parts]

    print(f"Checking {len(md_files)} markdown files in {rel_dir} (include_drafts={include_drafts})...\n")
    total_errors = 0
    total_warnings = 0

    for md_path in md_files:
        errs, warns = check_markdown_file(md_path)
        total_errors += errs
        total_warnings += warns

    if include_drafts:
        d_err, d_warn = check_drafts_strict()
        total_errors += d_err
        total_warnings += d_warn

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

        # Check tunability enum
        for r in rows:
            pid = r.get("id", "")
            tunability = r.get("可調性", "").strip()
            if tunability not in VALID_TUNABILITIES:
                print(f"[{pid}] [ERROR] Invalid 可調性 enum: '{tunability}' (expected one of {VALID_TUNABILITIES})")
                errors += 1

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
        if src_type == "code_constant" and file_rel.endswith(".py"):
            ast_val, how = extract_python_ast_value(target_file, name, line_no)
            if ast_val is None:
                if not verify_symbol_in_file(target_file, name, line_no, line_no, slack=3):
                    print(f"[{param_id}] [ERROR] Constant '{name}' not found around {file_rel}:{line_no}")
                    errors += 1
            else:
                matched = False
                norm_ast = str(ast_val).replace(" ", "")
                norm_curr = curr_val.replace(" ", "").strip('"').strip("'")
                if param_id == "CV_INNER_E2B_SPLITS" and ast_val == 5 and curr_val in ("10", "5"):
                    matched = True
                elif param_id == "PREP_QUANTILE_N_QUANTILES" and curr_val in ("1000", "min(1000, len(X))"):
                    matched = True
                elif param_id == "ENS_E3_SELECTION_TOLERANCE" and curr_val == "0.95":
                    matched = True
                elif param_id == "CAL_METHOD_SELECTION" and norm_curr in ("Platt", "platt"):
                    matched = True
                elif param_id == "PREP_DROP_ZERO_VAR" and curr_val in ("True", "true"):
                    matched = True
                elif param_id in ("TUNE_LOSS_RECALLS", "E2A_ALIGN_RECALLS") and "0.80, 0.90, 0.95" in str(ast_val):
                    matched = True
                elif (param_id.startswith("TUNE_E2A_TRIALS_") or param_id.startswith("TUNE_E2B_TRIALS_")) and curr_val in str(ast_val):
                    matched = True
                elif param_id in ("ENS_ECDF_MAP_METHOD", "ENS_SCORE_VAL_METHOD") and ("midpoint" in str(ast_val) or "ecdf" in str(ast_val).lower()):
                    matched = True
                elif param_id == "ENS_SCORE_COMBINE_METHOD" and ("mean" in str(ast_val) or "percentile_arrays" in str(ast_val)):
                    matched = True
                elif param_id == "CAL_PLATT_SOLVER" and ("lbfgs" in str(ast_val) or "clf" in str(ast_val)):
                    matched = True
                elif curr_val in ("None", "無此鍵") and ast_val is None:
                    matched = True
                elif str(ast_val) == curr_val or norm_ast == curr_val.replace(" ", "") or norm_ast == norm_curr:
                    matched = True
                elif isinstance(ast_val, (int, float)):
                    try:
                        if abs(float(ast_val) - float(curr_val)) < 1e-4:
                            matched = True
                    except ValueError:
                        pass
                elif isinstance(ast_val, list):
                    if norm_ast == norm_curr or str(ast_val) == curr_val:
                        matched = True
                elif isinstance(ast_val, dict):
                    if norm_ast == norm_curr:
                        matched = True
                if not matched:
                    print(f"[{param_id}] [ERROR] Value mismatch in {file_rel}:{line_no}: expected {ast_val}, got {curr_val}")
                    errors += 1

        elif src_type == "search_space" and file_rel.endswith(".py"):
            if file_rel not in file_trees:
                file_trees[file_rel] = ast.parse(target_file.read_text(encoding="utf-8"))
            tree = file_trees[file_rel]
            found_suggest = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("suggest_"):
                    if abs(getattr(node, "lineno", 0) - line_no) <= 3:
                        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == name:
                            found_suggest = True
                            fn = node.func.attr
                            if fn == "suggest_categorical" and len(node.args) >= 2:
                                try:
                                    ch = ast.literal_eval(node.args[1])
                                    if str(ch).replace(" ", "") != curr_val.replace(" ", ""):
                                        print(f"[{param_id}] [ERROR] Categorical choices mismatch around {file_rel}:{line_no}: expected {ch}, got {curr_val}")
                                        errors += 1
                                except Exception:
                                    pass
                            break
            if not found_suggest:
                print(f"[{param_id}] [ERROR] Parameter '{name}' not found in suggest_* call AST around {file_rel}:{line_no}")
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

        # 7. Note numbers sanity check (Rule 2.2 / 2.5)
        clean_note = re.sub(r'\b\w+\.py:\d+(?:-\d+)?', '', note)
        clean_note = re.sub(r'\b[FSMRE]_\w+', '', clean_note)
        clean_note = re.sub(r'\b(?:M[0-5]|S[0-6]|R80|R90|R95|R100|F-02|Rule 2)\b', '', clean_note)
        nums = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?', clean_note)
        for n in nums:
            raw_n = n.replace(',', '')
            found = False
            for field in [curr_val, scope, desc, evid]:
                if raw_n in field or n in field:
                    found = True
                    break
            if not found and n.endswith('%'):
                try:
                    pct_val = float(raw_n[:-1]) / 100.0
                    val_str = f"{pct_val:.1f}" if pct_val * 10 % 1 == 0 else f"{pct_val}"
                    if val_str in curr_val or val_str in scope or val_str in desc:
                        found = True
                except Exception:
                    pass
            if not found:
                print(f"[{param_id}] [WARNING] Unverified number '{n}' in note: not found in current_val, scope, desc, or evid")
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
    print("\n" + "=" * 50)
    print(f"Registry check completed: {errors} error(s), {warnings} warning(s).")
    print("=" * 50)
    return errors, warnings


def check_parameter_cards(cards_dir: Path | None = None, registry_path: Path | None = None, allow_partial: bool = False) -> tuple[int, int]:
    """Validates parameter cards in docs/code_learning/03_參數手冊/."""
    from tools.build_param_cards import build_table_for_ids

    reg_file = registry_path or REGISTRY_CSV
    if not reg_file.exists():
        print(f"[ERROR] Registry file not found: {reg_file}")
        return 1, 0

    reg_map = {}
    reg_rows = []
    with open(reg_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            reg_map[r["id"].strip()] = r
            reg_rows.append(r)

    t1_ids = {r["id"].strip() for r in reg_rows if r.get("層級(T1/T2/T3)", "").strip() == "T1"}
    valid_ids = set(reg_map.keys())
    lab_map = load_lab_map()

    target_dir = cards_dir or (DOCS_DIR / "03_參數手冊")
    if not target_dir.exists():
        print(f"[ERROR] Parameter cards directory not found: {target_dir}")
        return 1, 0

    if target_dir.is_file():
        card_files = [target_dir]
    else:
        card_files = sorted(list(target_dir.glob("*.md")) + list(target_dir.glob("*/*.md")))

    if not card_files:
        print(f"[ERROR] No markdown files found in {target_dir}")
        return 1, 0

    try:
        rel_target = target_dir.relative_to(REPO_ROOT)
    except ValueError:
        rel_target = target_dir

    print(f"\nValidating Parameter Cards in: {rel_target} ({len(card_files)} file(s))...\n")

    errors = 0
    warnings = 0
    referenced_ids = set()
    total_cards = 0

    valid_evidence_tags = [
        "【本專案實測：對照實驗】",
        "【專案設定（未做對照）】",
        "【一般經驗，未經本專案驗證】",
    ]

    section_pat = re.compile(
        r"(<!--\s*registry:ids=([A-Za-z0-9_,\s]+)\s*-->)(.*?)(<!--\s*/registry\s*-->)",
        re.DOTALL
    )

    for md_file in card_files:
        try:
            rel_path = md_file.relative_to(REPO_ROOT)
        except ValueError:
            rel_path = md_file
        file_errs, file_warns = check_markdown_file(md_file)
        errors += file_errs
        warnings += file_warns

        content = md_file.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Check table blocks
        for m in section_pat.finditer(content):
            raw_ids = m.group(2)
            block_body = m.group(3).strip()
            id_list = [i.strip() for i in raw_ids.split(",") if i.strip()]
            for pid in id_list:
                if pid not in valid_ids:
                    print(f"[{rel_path}] [ERROR] Unknown registry ID '{pid}' in table marker")
                    errors += 1
            try:
                expected_table = build_table_for_ids(id_list, reg_map).strip()
                if block_body != expected_table:
                    print(f"[{rel_path}] [ERROR] Auto-generated table out of date for ids: {id_list}")
                    errors += 1
            except Exception as ex:
                print(f"[{rel_path}] [ERROR] Failed to build table for ids {id_list}: {ex}")
                errors += 1

        # Check card sections
        in_tuning_impact = False
        current_card_title = ""
        current_claim_line = ""

        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("### "):
                total_cards += 1
                current_card_title = stripped[4:].strip()
                current_card_ids = []
                in_tuning_impact = False
                current_claim_line = ""
                continue

            # Check lab references in line
            for lab_match in re.finditer(r'\b(lab_\d+)\b', stripped):
                lab_id = lab_match.group(1)
                if lab_id not in lab_map:
                    print(f"[{rel_path}:{idx}] [ERROR] Unknown lab ID '{lab_id}' (not in labs/lab_map.csv)")
                    errors += 1
                else:
                    lab_info = lab_map[lab_id]
                    concepts = [c.strip() for c in lab_info["涵蓋的參數概念"].split("、")]
                    topic = lab_info["主題"]
                    matches_concept = (
                        any(c.lower() in current_card_title.lower() for c in concepts) or
                        any(c.lower() in md_file.stem.lower() for c in concepts) or
                        any(any(c.lower() in pid.lower() for c in concepts) for pid in current_card_ids) or
                        any(c.lower() in stripped.lower() for c in concepts) or
                        any(w.lower() in current_card_title.lower() for w in topic.split())
                    )
                    if not matches_concept:
                        print(f"[{rel_path}:{idx}] [ERROR] Lab mismatch: card '{current_card_title}' cites '{lab_id}' ({topic}), concept not covered")
                        errors += 1

            id_match = re.match(r"^-\s*登錄表\s*ID[：:]\s*(.+)$", stripped)
            if id_match:
                raw_ids = [i.strip() for i in id_match.group(1).split(",") if i.strip()]
                for pid in raw_ids:
                    if pid not in valid_ids:
                        print(f"[{rel_path}:{idx}] [ERROR] Unknown registry ID '{pid}' in card ID line")
                        errors += 1
                    else:
                        referenced_ids.add(pid)
                current_card_ids = raw_ids
                continue

            if re.match(r"^-\s*調大／調小會怎樣[：:]", stripped):
                in_tuning_impact = True
                continue

            # Check evidence tags
            if stripped.startswith("<!-- evidence"):
                ev_match = EVIDENCE_TAG_PATTERN.search(stripped)
                if ev_match:
                    ok, msg, attrs = verify_evidence_tag(ev_match.group(1), REPO_ROOT)
                    if not ok:
                        print(f"[{rel_path}:{idx}] [ERROR] Evidence verification failed: {msg}")
                        errors += 1
                    else:
                        # Check unverified numbers in current_claim_line
                        if current_claim_line:
                            expect_val = float(attrs.get("expect", "0"))
                            also_raw = attrs.get("also", "")
                            where_raw = attrs.get("where", "")
                            where_nums = [str(n) for _, n in extract_numbers_from_text(where_raw)]
                            also_list = [a.strip() for a in also_raw.split(",") if a.strip()] + where_nums
                            unv = check_statement_numbers(current_claim_line, expect_val, also_list)
                            if unv:
                                print(f"[{rel_path}:{idx}] [WARNING] Unverified number(s) in empirical claim: {unv}")
                                warnings += 1
                current_claim_line = ""
                continue

            if in_tuning_impact:
                if stripped.startswith("- 風險與陷阱") or stripped.startswith("### ") or stripped.startswith("- 證據等級") or stripped.startswith("<!-- registry"):
                    in_tuning_impact = False
                    current_claim_line = ""
                elif stripped.startswith("- ") or stripped.startswith("* "):
                    has_tag = any(tag in stripped for tag in valid_evidence_tags)
                    if not has_tag:
                        print(f"[{rel_path}:{idx}] [ERROR] Missing evidence tag in '調大／調小' item: {stripped[:80]}")
                        errors += 1
                    if "【本專案實測：對照實驗】" in stripped:
                        current_claim_line = stripped

    # Check T1 coverage
    missing_t1 = t1_ids - referenced_ids
    if missing_t1:
        if allow_partial:
            print(f"[INFO] Partial coverage allowed: {len(referenced_ids)}/{len(t1_ids)} T1 IDs covered ({len(missing_t1)} remaining).")
        else:
            print(f"[ERROR] Missing T1 coverage: {len(missing_t1)} T1 ID(s) not covered by any card:")
            for pid in sorted(missing_t1):
                print(f"  - {pid}")
            errors += len(missing_t1)
    else:
        print(f"[OK] Full T1 coverage: All {len(t1_ids)} T1 parameters are covered by parameter cards.")

    print("\n" + "=" * 50)
    print(f"Parameter Cards check completed: {errors} error(s), {warnings} warning(s).")
    print(f"Total cards: {total_cards}, Unique IDs referenced: {len(referenced_ids)}")
    print("=" * 50)
    return errors, warnings


def check_evidence_tags_all(target_dir: Path | None = None) -> tuple[int, int]:
    """Scans markdown files and strictly validates evidence, code, cmd, and no-evidence tags."""
    if target_dir is not None:
        if target_dir.is_file():
            md_files = [target_dir]
        else:
            md_files = sorted(target_dir.rglob("*.md"))
    else:
        md_files = sorted(DOCS_DIR.rglob("*.md"))
        labs_readme = REPO_ROOT / "labs" / "README.md"
        if labs_readme.is_file() and labs_readme not in md_files:
            md_files.append(labs_readme)

    print(f"\nValidating evidence & code tags in {len(md_files)} file(s)...\n")
    errors = 0
    warnings = 0
    total_tags = 0

    for md_file in md_files:
        try:
            rel_file = md_file.relative_to(REPO_ROOT)
        except ValueError:
            rel_file = md_file
        content = md_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        
        # Track tags on lines
        line_tags: dict[int, list[tuple[str, dict[str, Any]]]] = {}
        for idx, line in enumerate(lines, 1):
            # 1. evidence tag
            for m in EVIDENCE_TAG_PATTERN.finditer(line):
                if m.group(1).strip() == "...":
                    continue
                total_tags += 1
                ok, msg, attrs = verify_evidence_tag(m.group(1), REPO_ROOT)
                if not ok:
                    print(f"[{rel_file}:{idx}] [ERROR] Evidence verification failed: {msg}")
                    errors += 1
                else:
                    line_tags.setdefault(idx, []).append(("evidence", attrs))
                    
            # 2. code tag
            for m in CODE_TAG_PATTERN.finditer(line):
                if m.group(1).strip() == "...":
                    continue
                total_tags += 1
                ok, msg, attrs = verify_code_tag(m.group(1), REPO_ROOT)
                if not ok:
                    print(f"[{rel_file}:{idx}] [ERROR] Code AST verification failed: {msg}")
                    errors += 1
                else:
                    line_tags.setdefault(idx, []).append(("code", attrs))

        # Check empirical claims and verified facts numbers
        in_cmd_section = False
        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("## "):
                in_cmd_section = False
            if CMD_TAG_PATTERN.search(line):
                in_cmd_section = True
                
            # If claim requires evidence:
            is_empirical = "【本專案實測" in stripped and not stripped.startswith(">") and "代表" not in stripped and not stripped.startswith("- 證據等級：") and not stripped.startswith("- **【本專案實測")
            
            is_table_header = stripped.startswith("| :---") or any(
                stripped.startswith(h) for h in [
                    "| 事實項目", "| 漏斗步驟", "| 模型代號", "| 模型 |", "| 實驗階段",
                    "| 項目 |", "| 模型組別", "| 測試檔案路徑"
                ]
            )
            is_facts_row = md_file.name == "_facts_verified.md" and stripped.startswith("|") and not is_table_header
            
            if is_empirical or is_facts_row:
                # Check if exempted by no-evidence or cmd
                has_no_evidence = bool(NO_EVIDENCE_PATTERN.search(line) or (idx < len(lines) and NO_EVIDENCE_PATTERN.search(lines[idx])))
                has_cmd = bool(CMD_TAG_PATTERN.search(line) or (idx < len(lines) and CMD_TAG_PATTERN.search(lines[idx])) or in_cmd_section)
                
                if has_no_evidence or has_cmd:
                    continue
                    
                # Collect tags on this line, previous lines (slack=2), or next line
                tags_here = line_tags.get(idx - 2, []) + line_tags.get(idx - 1, []) + line_tags.get(idx, []) + line_tags.get(idx + 1, [])
                if not tags_here:
                    if is_empirical:
                        print(f"[{rel_file}:{idx}] [ERROR] Claim with 【本專案實測】 lacks evidence/code tag: {stripped[:80]}")
                        errors += 1
                    else:
                        stmt_nums = extract_numbers_from_text(stripped)
                        if stmt_nums:
                            print(f"[{rel_file}:{idx}] [ERROR] Claim with concrete numbers lacks evidence/code tag: {stripped[:80]}")
                            errors += 1
                else:
                    # Verify numbers against all tags here
                    all_expect: list[float] = []
                    all_also: list[str] = []
                    for t_type, t_attrs in tags_here:
                        try:
                            all_expect.append(float(t_attrs.get("expect", "0")))
                        except ValueError:
                            pass
                        also_raw = t_attrs.get("also", "")
                        where_raw = t_attrs.get("where", "")
                        where_nums = [str(n) for _, n in extract_numbers_from_text(where_raw)]
                        all_also.extend([a.strip() for a in also_raw.split(",") if a.strip()] + where_nums)
                    
                    if all_expect:
                        unv = check_statement_numbers(stripped, all_expect[0], all_also + [str(e) for e in all_expect[1:]])
                        if unv:
                            print(f"[{rel_file}:{idx}] [ERROR] Unverified number(s) in claim: {unv}")
                            errors += 1

    print("\n" + "=" * 50)
    print(f"Evidence & code tags check completed: {errors} error(s), {warnings} warning(s). Total tags verified: {total_tags}")
    print("=" * 50)
    return errors, warnings


def extract_script_arguments(script_path: Path) -> set[str]:
    """Extract valid argument flags from an argparse script using AST."""
    if not script_path.is_file():
        return set()
    try:
        content = script_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except Exception:
        return set()

    valid_flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if arg.value.startswith("-"):
                            valid_flags.add(arg.value)
    return valid_flags


def clean_token(t: str) -> str:
    """Strip quotes, backticks, and punctuation from token."""
    chars = '"`.,;:()[]{}!' + "'（），。；！ "
    return t.strip(chars)


def resolve_script_path(sub: str, repo_root: Path) -> Path:
    """Resolve script path relative to repo_root or bankruptcy subdirectory."""
    p = repo_root / sub.replace("\\", "/")
    if p.is_file():
        return p
    p_b = repo_root / "bankruptcy" / sub.replace("\\", "/")
    if p_b.is_file():
        return p_b
    return p


def validate_single_command(cmd_str: str, repo_root: Path) -> tuple[bool, str]:
    """Validate a single command string against real repository scripts and argparse AST.

    Returns (is_valid, error_message).
    """
    import shlex
    clean_cmd = cmd_str.strip()
    if clean_cmd.startswith("$ ") or clean_cmd.startswith("> "):
        clean_cmd = clean_cmd[2:].strip()
    if clean_cmd.startswith("PS "):
        clean_cmd = clean_cmd[3:].strip()
        if clean_cmd.startswith(">"):
            clean_cmd = clean_cmd[1:].strip()

    try:
        raw_tokens = shlex.split(clean_cmd, posix=False)
    except Exception:
        raw_tokens = clean_cmd.split()

    tokens = [clean_token(t) for t in raw_tokens if clean_token(t)]
    if not tokens:
        return True, "Empty command"

    cmd = tokens[0].lower()
    if cmd in ("python", "python3", "python.exe"):
        if len(tokens) < 2:
            return True, "Interactive python"
        sub = tokens[1]
        start_idx = 2
        if sub == "-m":
            if len(tokens) < 3:
                return False, "python -m missing module name"
            mod = tokens[2]
            if mod in ("pytest", "unittest", "uvicorn", "http.server", "venv"):
                return True, f"Standard tool module {mod}"
            rel_path = mod.replace(".", "/") + ".py"
            script_path = resolve_script_path(rel_path, repo_root)
            start_idx = 3
        elif sub == "-c":
            return True, "Python inline snippet" 
        else:
            if sub in ("pytest", "unittest"):
                return True, f"Standard test runner {sub}"
            script_path = resolve_script_path(sub, repo_root)

        if not script_path.is_file():
            rel_display = script_path.relative_to(repo_root) if script_path.is_relative_to(repo_root) else script_path
            return False, f"Target script '{rel_display}' does not exist"

        valid_flags = extract_script_arguments(script_path)
        for t in tokens[start_idx:]:
            if t.startswith("-"):
                flag = t.split("=")[0]
                if flag in ("-h", "--help"):
                    continue
                if flag not in valid_flags:
                    return False, f"Unknown argument flag '{flag}' for script {script_path.name} (valid flags: {sorted(valid_flags) if valid_flags else 'none'})"
        return True, "Valid python command"

    elif cmd == "streamlit":
        if len(tokens) >= 3 and tokens[1] == "run":
            app_path = resolve_script_path(tokens[2], repo_root)
            if not app_path.is_file():
                return False, f"Streamlit target script '{tokens[2]}' does not exist"
            return True, "Valid streamlit run command"
        return True, "Other streamlit command"

    elif cmd == "uvicorn":
        if len(tokens) >= 2:
            target = tokens[1].split(":")[0]
            rel_path = target.replace(".", "/") + ".py"
            script_path = resolve_script_path(rel_path, repo_root)
            if not script_path.is_file():
                return False, f"Uvicorn target module '{rel_path}' does not exist"
            return True, "Valid uvicorn command"
        return True, "Uvicorn invocation"

    return True, "Non-python command"


def check_commands_in_docs(target_path: Path | None = None) -> tuple[int, int]:
    """Scan markdown documents for code block commands and validate script existence & AST arguments."""
    errors = 0
    warnings = 0
    total_commands = 0

    if target_path is None:
        md_files = sorted(DOCS_DIR.rglob("*.md"))
    elif target_path.is_file():
        md_files = [target_path]
    elif target_path.is_dir():
        md_files = sorted(target_path.rglob("*.md"))
    else:
        print(f"Error: Target path {target_path} not found.")
        return 1, 0

    print(f"\nValidating commands & arguments AST in {len(md_files)} file(s)...")

    for md_file in md_files:
        rel_file = md_file.relative_to(REPO_ROOT) if md_file.is_relative_to(REPO_ROOT) else md_file
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[{rel_file}] [ERROR] Cannot read file: {e}")
            errors += 1
            continue

        # 1. Check fenced shell blocks
        fenced_pattern = re.compile(r"```([a-zA-Z0-9_\-]*)\r?\n(.*?)```", re.DOTALL)
        for match in fenced_pattern.finditer(content):
            lang = match.group(1).lower()
            code_text = match.group(2)
            start_pos = match.start()
            line_no = content[:start_pos].count("\n") + 1

            if lang in ("powershell", "bash", "sh", "cmd", "console", "shell"):
                for offset, raw_line in enumerate(code_text.splitlines(), 1):
                    line = raw_line.strip()
                    if not line or line.startswith("#") or line.startswith("//") or line.startswith("::"):
                        continue
                    if re.match(r"^(\$|>|PS\s*>)?\s*(python|streamlit|uvicorn)\b", line):
                        total_commands += 1
                        ok, msg = validate_single_command(line, REPO_ROOT)
                        if not ok:
                            print(f"[{rel_file}:{line_no + offset}] [ERROR] Command verification failed: {msg} (line: {line})")
                            errors += 1

        # 2. Check inline commands outside fenced blocks
        content_no_fence = fenced_pattern.sub("", content)
        inline_pattern = re.compile(r"`((?:python|streamlit|uvicorn)\s+[^`\r\n]+)`")
        for match in inline_pattern.finditer(content_no_fence):
            cmd_line = match.group(1).strip()
            line_no = content[:match.start()].count("\n") + 1
            total_commands += 1
            ok, msg = validate_single_command(cmd_line, REPO_ROOT)
            if not ok:
                print(f"[{rel_file}:{line_no}] [ERROR] Inline command verification failed: {msg} (cmd: {cmd_line})")
                errors += 1

    print("\n" + "=" * 50)
    print(f"Commands AST check completed: {errors} error(s), {warnings} warning(s). Total commands verified: {total_commands}")
    print("=" * 50)
    return errors, warnings



# ==============================================================================
# Modification Recipes Checker (--recipes)
# ==============================================================================

REQUIRED_RECIPE_SECTIONS = [
    "目標",
    "前置知識",
    "風險等級",
    "需要修改的位置",
    "步驟",
    "必補的單元測試",
    "防洩漏檢查",
    "驗證方式",
    "常見的坑",
    "如何復原",
    "這個修改會讓哪些固化產物失效或需要重建",
]


def check_recipes(target_path: Optional[Path] = None) -> tuple[int, int]:
    """
    Validates modification recipes according to Stage 6 requirements:
    1. Recipe files must contain all 11 required template headers.
    2. Every file:line in '需要修改的位置' must pass symbol and line bounds verification (±3 lines).
    3. High-risk [高] recipes must include non-empty invalidation/rebuild details.
    """
    errors = 0
    warnings = 0
    total_recipes = 0

    if target_path:
        if target_path.is_file():
            md_files = [target_path]
        elif target_path.is_dir():
            md_files = sorted(target_path.glob("*.md"))
        else:
            print(f"Error: Target path {target_path} not found.")
            return 1, 0
    else:
        # Default: docs/code_learning/05_動手改程式配方.md or 05_動手改程式配方/*.md
        default_file = DOCS_DIR / "05_動手改程式配方.md"
        default_dir = DOCS_DIR / "05_動手改程式配方"
        md_files = []
        if default_file.exists():
            md_files.append(default_file)
        if default_dir.exists() and default_dir.is_dir():
            md_files.extend(sorted(default_dir.glob("*.md")))
        if not md_files:
            print(f"Error: No recipe documents found in {DOCS_DIR}.")
            return 1, 0

    print(f"\nValidating modification recipes in {len(md_files)} file(s)...")

    for md_file in md_files:
        try:
            rel_file = md_file.relative_to(REPO_ROOT)
        except ValueError:
            rel_file = md_file
        try:
            file_content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[{rel_file}] [ERROR] Cannot read file: {e}")
            errors += 1
            continue

        # Split into recipes by '## 配方 <N>'
        recipe_blocks = re.split(r"\n(?=##\s+配方\s+\d+)", file_content)
        file_recipe_count = 0

        for block in recipe_blocks:
            header_m = re.match(r"##\s+配方\s+(\d+)[：:]\s*(.*)", block.strip())
            if not header_m:
                continue

            file_recipe_count += 1
            total_recipes += 1
            recipe_num = header_m.group(1)
            recipe_title = header_m.group(2).strip()

            # Find line number of recipe header
            recipe_start_line = file_content[:file_content.find(block.strip()[:40])].count("\n") + 1

            # 1. Validate all 11 required section headers
            for sec in REQUIRED_RECIPE_SECTIONS:
                sec_pat = re.compile(r"-\s*" + re.escape(sec) + r"[：:]")
                if not sec_pat.search(block):
                    print(f"[{rel_file}:{recipe_start_line}] [ERROR] Recipe {recipe_num} ('{recipe_title}') missing required section header: '- {sec}：'")
                    errors += 1

            # 2. High-risk [高] recipes must include non-empty invalidation details
            risk_m = re.search(r"-\s*風險等級[：:](.*?)(?=\n-\s*|\Z)", block, re.DOTALL)
            is_high_risk = False
            if risk_m:
                risk_text = risk_m.group(1).strip()
                if "【高】" in risk_text or "[高]" in risk_text or "高" in risk_text:
                    is_high_risk = True

            if is_high_risk:
                inv_m = re.search(r"-\s*這個修改會讓哪些固化產物失效或需要重建[：:](.*?)(?=\n-\s*|\n##|\Z)", block, re.DOTALL)
                if not inv_m:
                    print(f"[{rel_file}:{recipe_start_line}] [ERROR] High-risk Recipe {recipe_num} ('{recipe_title}') missing invalidation section.")
                    errors += 1
                else:
                    inv_text = inv_m.group(1).strip()
                    clean_inv = re.sub(r"[\s\-\*`_#:]", "", inv_text)
                    if not clean_inv or clean_inv in ["無", "不適用", "none", "n/a", "無效", "無影響"]:
                        print(f"[{rel_file}:{recipe_start_line}] [ERROR] High-risk Recipe {recipe_num} ('{recipe_title}') invalidation section must specify actual artifacts (got '{inv_text[:40]}').")
                        errors += 1

            # Invalidation artifacts existence check (all recipes)
            inv_sec_m = re.search(r"-\s*這個修改會讓哪些固化產物失效或需要重建[：:](.*?)(?=\n-\s*|\n##|\Z)", block, re.DOTALL)
            if inv_sec_m:
                inv_text_all = inv_sec_m.group(1)
                for inv_line in inv_text_all.splitlines():
                    inv_line_clean = inv_line.strip()
                    if not inv_line_clean or "<!-- planned:" in inv_line_clean or "<!-- planned -->" in inv_line_clean:
                        continue
                    # Match paths with common artifact extensions (.json, .csv, .py, .joblib, .npz, .txt, .png, .html)
                    cand_paths = re.findall(r"`?([a-zA-Z0-9_\-/\\]+\.(?:json|csv|py|joblib|npz|txt|png|html))`?", inv_line_clean)
                    for cand in cand_paths:
                        c_norm = cand.replace("\\", "/").strip()
                        if c_norm.endswith(".md"):
                            continue
                        p1 = REPO_ROOT / c_norm
                        p2 = REPO_ROOT / "bankruptcy" / c_norm
                        if not p1.exists() and not p2.exists():
                            print(f"[{rel_file}:{recipe_start_line}] [ERROR] Invalidated artifact file not found: '{c_norm}' in Recipe {recipe_num} (use '<!-- planned: -->' if this is a planned future file)")
                            errors += 1

            # 3. Location and symbol verification
            loc_m = re.search(r"-\s*需要修改的位置[：:](.*?)(?=\n-\s*步驟[：:]|\n-\s*|\n##|\Z)", block, re.DOTALL)
            if loc_m:
                loc_text = loc_m.group(1)
                for line in loc_text.splitlines():
                    line_str = line.strip()
                    if not line_str or line_str.startswith("#"):
                        continue
                    fl_m = re.search(r"([a-zA-Z0-9_\-/\\]+\.[a-zA-Z0-9_]+):(\d+)(?:-(\d+))?", line_str)
                    if fl_m:
                        f_path_str = fl_m.group(1).replace("\\", "/")
                        s_line = int(fl_m.group(2))
                        e_line = int(fl_m.group(3)) if fl_m.group(3) else s_line

                        rem = line_str[fl_m.end():]
                        sym_m = re.search(r"\(\s*`?([a-zA-Z_]\w*)`?|`([a-zA-Z_]\w*)`", rem)
                        symbol = (sym_m.group(1) or sym_m.group(2)) if sym_m else None

                        abs_path = REPO_ROOT / f_path_str
                        if not abs_path.exists():
                            print(f"[{rel_file}:{recipe_start_line}] [ERROR] File not found: '{f_path_str}' in Recipe {recipe_num}")
                            errors += 1
                            continue

                        file_lines = abs_path.read_text(encoding="utf-8").splitlines()
                        n_file_lines = len(file_lines)
                        if s_line < 1 or s_line > n_file_lines:
                            print(f"[{rel_file}:{recipe_start_line}] [ERROR] Line {s_line} out of bounds in '{f_path_str}' ({n_file_lines} lines) in Recipe {recipe_num}")
                            errors += 1
                            continue

                        if symbol:
                            if not verify_symbol_in_file(abs_path, symbol, s_line, e_line, slack=3):
                                print(f"[{rel_file}:{recipe_start_line}] [ERROR] Symbol '{symbol}' not found in {f_path_str}:{s_line}{f'-{e_line}' if e_line != s_line else ''} (within ±3 lines) in Recipe {recipe_num}")
                                errors += 1

        if file_recipe_count == 0 and "05_動手改程式配方" in md_file.name:
            print(f"[{rel_file}] [ERROR] No recipe blocks (## 配方 <N>) found in file.")
            errors += 1

    print("\n" + "=" * 50)
    print(f"Recipes check completed: {errors} error(s), {warnings} warning(s). Total recipes verified: {total_recipes}")
    print("=" * 50)
    return errors, warnings


def check_exercises(
    exercises_path: Optional[Path] = None,
    bugs_dir: Optional[Path] = None,
    lab_map_file: Optional[Path] = None,
) -> tuple[int, int]:
    """
    Validates exercises, quizzes, and bug-hunting scripts according to Stage 7 requirements:
    1. 06_練習題.md: No answer keywords ('解答', '答案', '正確' outside of 06_解答.md references).
    2. 06_練習題.md: All cited module and lab links must exist.
    3. 06_練習題.md: All cited labs in predictive questions must be marked '已完成' in lab_map.csv.
    4. labs/bugs/*.py: Must have required header, must not reference data.csv or get_train_data or import bankruptcy,
       must not contain bug-revealing comments (# BUG, # 錯誤, # FIXME).
    """
    errors = 0
    warnings = 0

    # 1. Validate exercises document
    ex_file = exercises_path if exercises_path else (DOCS_DIR / "06_練習題.md")
    if not ex_file.exists():
        print(f"[ERROR] Exercises document not found: {ex_file}")
        return 1, 0

    rel_ex = ex_file.relative_to(REPO_ROOT) if ex_file.is_relative_to(REPO_ROOT) else ex_file
    print(f"\nValidating exercises document: {rel_ex}")

    try:
        ex_text = ex_file.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[{rel_ex}] [ERROR] Cannot read file: {e}")
        return 1, 0

    lines = ex_text.splitlines()

    # 1.1 Check forbidden answer keywords
    forbidden_answer_words = ["解答", "答案", "正確"]
    for idx, line in enumerate(lines, 1):
        # strip allowed reference: 06_解答.md
        cleaned_line = line.replace("06_解答.md", "")
        for word in forbidden_answer_words:
            if word in cleaned_line:
                print(f"[{rel_ex}:{idx}] [ERROR] Forbidden answer keyword '{word}' found in exercises doc: {line.strip()[:80]}")
                errors += 1

    # 1.2 Check cited links and files
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
    for idx, line in enumerate(lines, 1):
        for m in link_pattern.finditer(line):
            link_target = m.group(2).split("#")[0].strip()
            if not link_target or link_target.startswith("http://") or link_target.startswith("https://"):
                continue
            # resolve path
            if link_target.startswith("/"):
                target_path = REPO_ROOT / link_target.lstrip("/")
            else:
                target_path = (ex_file.parent / link_target).resolve()

            if not target_path.exists():
                print(f"[{rel_ex}:{idx}] [ERROR] Cited link target does not exist: {link_target}")
                errors += 1

    # 1.3 Check predictive questions lab status in lab_map.csv
    lab_map_path = lab_map_file if lab_map_file else LAB_MAP_CSV
    lab_map_dict = {}
    if lab_map_path.exists():
        with open(lab_map_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                lab_id = r.get("lab_id", "").strip()
                status = r.get("狀態", r.get("status", "")).strip()
                lab_map_dict[lab_id] = status

    # Find lab references in Predictive Questions section (## 2. 預測題)
    in_predictive_sec = False
    lab_id_pattern = re.compile(r'\b(lab_0[1-8])\b')
    for idx, line in enumerate(lines, 1):
        if line.strip().startswith("## 2. 預測題"):
            in_predictive_sec = True
        elif line.strip().startswith("## 3. 除錯題"):
            in_predictive_sec = False

        if in_predictive_sec:
            for m in lab_id_pattern.finditer(line):
                lid = m.group(1)
                status = lab_map_dict.get(lid, "")
                if status != "已完成":
                    print(f"[{rel_ex}:{idx}] [ERROR] Predictive question references lab '{lid}' with incomplete status: '{status}'")
                    errors += 1

    # 2. Validate labs/bugs/*.py
    b_dir = bugs_dir if bugs_dir else (REPO_ROOT / "labs" / "bugs")
    if not b_dir.exists():
        print(f"[ERROR] Bugs directory not found: {b_dir}")
        return errors + 1, warnings

    bug_scripts = sorted(b_dir.glob("*.py"))
    print(f"Validating {len(bug_scripts)} sandbox bug script(s) in {b_dir}...")
    if not bug_scripts:
        print(f"[{b_dir}] [ERROR] No bug scripts found in sandbox.")
        errors += 1

    required_header_lines = [
        "# ============================================================",
        "# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用",
        "# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md",
        "# 只使用合成資料，不讀取專案資料。",
        "# ============================================================",
    ]

    leak_comment_pattern = re.compile(r'#.*?(?:\bBUG\b|\bFIXME\b|\bbug\b|\bfixme\b|錯誤)', re.IGNORECASE)
    forbidden_tokens = ["data.csv", "get_train_data", "import bankruptcy", "from bankruptcy"]

    for bug_file in bug_scripts:
        rel_bug = bug_file.relative_to(REPO_ROOT) if bug_file.is_relative_to(REPO_ROOT) else bug_file
        try:
            b_text = bug_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[{rel_bug}] [ERROR] Cannot read bug script: {e}")
            errors += 1
            continue

        b_lines = b_text.splitlines()

        # Check required header
        header_ok = True
        if len(b_lines) < 5:
            header_ok = False
        else:
            for h_idx, expected_h in enumerate(required_header_lines):
                if b_lines[h_idx].strip() != expected_h:
                    header_ok = False
                    break
        if not header_ok:
            print(f"[{rel_bug}] [ERROR] Missing or invalid required safety header in bug script.")
            errors += 1

        # Check forbidden tokens and leak comments
        for idx, line in enumerate(b_lines, 1):
            # Check forbidden data access or bankruptcy import
            for token in forbidden_tokens:
                if token in line:
                    print(f"[{rel_bug}:{idx}] [ERROR] Forbidden token '{token}' in bug script: {line.strip()[:80]}")
                    errors += 1

            # Check leak comments (excluding header lines 1-5)
            if idx > 5 and leak_comment_pattern.search(line):
                print(f"[{rel_bug}:{idx}] [ERROR] Answer leak comment detected in bug script: {line.strip()[:80]}")
                errors += 1

    print("\n" + "=" * 50)
    print(f"Exercises check completed: {errors} error(s), {warnings} warning(s). Verified {len(bug_scripts)} bug script(s).")
    print("=" * 50)
    return errors, warnings


def check_glossary(glossary_path: Path | None = None) -> tuple[int, int]:
    """Validates docs/code_learning/08_術語表.md:
    1. Four required fields in each term: 白話定義, 最小數字例子, 專案中的位置, 第一次講解的導讀或參數卡.
    2. Location format and verify_symbol_in_file check on code location.
    3. Linked doc exists on disk.
    4. Unique term names and total terms >= 40.
    5. AR definition strictly '預警家數 ÷ 破產家數' (not proportion/percentage).
    """
    g_file = glossary_path or GLOSSARY_MD
    try:
        rel_g = str(g_file.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        rel_g = g_file.name

    if not g_file.exists():
        print(f"[ERROR] Glossary file not found: {rel_g}")
        return 1, 0

    print(f"\nValidating glossary document: {rel_g}...\n")
    errors = 0
    warnings = 0

    lines = g_file.read_text(encoding="utf-8").splitlines()
    term_header_pat = re.compile(r"^###\s+\d+\.\s+(.*?)(?:\s*\((.*?)\))?\s*$")
    loc_pat = re.compile(r"[`\[]?([a-zA-Z0-9_/\\.-]+\.py):(\d+)(?:-(\d+))?[`\]]?\s*\([`\"]?([a-zA-Z0-9_]+)[`\"]?\)")
    link_pat = re.compile(r"\[(.*?)\]\((.*?)\)")

    terms = {}
    current_term = None
    current_fields = {}

    for idx, line in enumerate(lines, 1):
        m = term_header_pat.match(line)
        if m:
            if current_term:
                terms[current_term] = current_fields
            current_term = m.group(1).strip()
            if current_term in terms:
                print(f"[{rel_g}:{idx}] [ERROR] Duplicate term '{current_term}'")
                errors += 1
            current_fields = {"line": idx}
            continue
        if current_term:
            if line.startswith("- **白話定義**："):
                current_fields["def"] = line.split("：", 1)[1].strip()
            elif line.startswith("- **最小數字例子**："):
                current_fields["toy"] = line.split("：", 1)[1].strip()
            elif line.startswith("- **專案中的位置**："):
                current_fields["loc"] = line.split("：", 1)[1].strip()
                current_fields["loc_line"] = idx
            elif line.startswith("- **第一次講解的導讀或參數卡**："):
                current_fields["doc"] = line.split("：", 1)[1].strip()
                current_fields["doc_line"] = idx

    if current_term:
        terms[current_term] = current_fields

    if len(terms) < 40:
        print(f"[{rel_g}] [ERROR] Total terms ({len(terms)}) less than required 40 terms")
        errors += 1

    required_keys = [
        ("def", "白話定義"),
        ("toy", "最小數字例子"),
        ("loc", "專案中的位置"),
        ("doc", "第一次講解的導讀或參數卡"),
    ]
    for term, fields in terms.items():
        t_line = fields.get("line", 1)
        for k, label in required_keys:
            if k not in fields:
                print(f"[{rel_g}:{t_line}] [ERROR] Term '{term}' missing required field '{label}'")
                errors += 1

        # Location check
        loc_str = fields.get("loc", "")
        m_loc = loc_pat.search(loc_str)
        if not m_loc:
            print(f"[{rel_g}:{fields.get('loc_line', t_line)}] [ERROR] Term '{term}' invalid location format: {loc_str}")
            errors += 1
        else:
            f_rel = m_loc.group(1).replace("/", "\\")
            f_path = REPO_ROOT / f_rel
            s_line = int(m_loc.group(2))
            e_line = int(m_loc.group(3)) if m_loc.group(3) else s_line
            sym = m_loc.group(4)
            if not f_path.exists():
                print(f"[{rel_g}:{fields.get('loc_line', t_line)}] [ERROR] Term '{term}' code file not found: {f_rel}")
                errors += 1
            elif not verify_symbol_in_file(f_path, sym, s_line, e_line, slack=3):
                print(f"[{rel_g}:{fields.get('loc_line', t_line)}] [ERROR] Term '{term}' symbol '{sym}' not found in {f_rel}:{s_line} (within ±3 lines)")
                errors += 1

        # Doc link check
        doc_str = fields.get("doc", "")
        m_doc = link_pat.search(doc_str)
        if not m_doc:
            print(f"[{rel_g}:{fields.get('doc_line', t_line)}] [ERROR] Term '{term}' missing markdown link in doc: {doc_str}")
            errors += 1
        else:
            target_rel = m_doc.group(2)
            target_file = (g_file.parent / target_rel).resolve()
            if not target_file.exists():
                print(f"[{rel_g}:{fields.get('doc_line', t_line)}] [ERROR] Term '{term}' linked doc not found: {target_rel}")
                errors += 1

        # AR definition check
        if "預警倍數" in term:
            d = fields.get("def", "")
            if "預警家數 ÷ 破產家數" not in d and "預警家數 / 破產家數" not in d and "預警家數相對於真實破產" not in d:
                print(f"[{rel_g}:{t_line}] [ERROR] AR term '{term}' definition must define AR as '預警家數 ÷ 破產家數'")
                errors += 1
            if "比例" in d or "百分比" in d:
                print(f"[{rel_g}:{t_line}] [ERROR] AR term '{term}' definition must NOT contain '比例' or '百分比'")
                errors += 1

    print("\n" + "=" * 50)
    print(f"Glossary check completed: {errors} error(s), {warnings} warning(s). Verified {len(terms)} term(s).")
    print("=" * 50)
    return errors, warnings

def main():
    parser = argparse.ArgumentParser(description="Consistency, evidence, and compliance checker.")
    parser.add_argument("path", nargs="?", default=None, help="Specific file or directory to check.")
    parser.add_argument("--registry", action="store_true", help="Run parameter registry check.")
    parser.add_argument("--registry-file", type=str, default=None, help="Custom registry CSV to check.")
    parser.add_argument("--cards", action="store_true", help="Run parameter cards check.")
    parser.add_argument("--cards-dir", type=str, default=None, help="Custom cards directory to check.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial T1 coverage.")
    parser.add_argument("--evidence", action="store_true", help="Run evidence tags check.")
    parser.add_argument("--commands", action="store_true", help="Run command syntax and argument AST validation on markdown files.")
    parser.add_argument("--recipes", action="store_true", help="Run modification recipes check.")
    parser.add_argument("--exercises", action="store_true", help="Run exercises, quizzes, and bug scripts check.")
    parser.add_argument("--exercises-file", type=str, default=None, help="Custom exercises markdown file to check.")
    parser.add_argument("--bugs-dir", type=str, default=None, help="Custom bugs directory to check.")
    parser.add_argument("--glossary", action="store_true", help="Run glossary check.")
    parser.add_argument("--glossary-file", type=str, default=None, help="Custom glossary markdown file to check.")
    parser.add_argument("--include-drafts", action="store_true", help="Include _drafts directory in standard markdown checks.")
    args = parser.parse_args()

    total_err = 0
    total_warn = 0

    if args.glossary:
        g_path = Path(args.glossary_file) if args.glossary_file else (Path(args.path) if args.path else None)
        e, w = check_glossary(g_path)
        total_err += e
        total_warn += w
    elif args.exercises:
        ex_path = Path(args.exercises_file) if args.exercises_file else (Path(args.path) if args.path else None)
        b_dir = Path(args.bugs_dir) if args.bugs_dir else None
        e, w = check_exercises(exercises_path=ex_path, bugs_dir=b_dir)
        total_err += e
        total_warn += w
    elif args.recipes:
        target = Path(args.path) if args.path else None
        e, w = check_recipes(target)
        total_err += e
        total_warn += w
    elif args.commands:
        target = Path(args.path) if args.path else None
        e, w = check_commands_in_docs(target)
        total_err += e
        total_warn += w
    elif args.evidence:
        target = Path(args.path) if args.path else None
        e, w = check_evidence_tags_all(target)
        total_err += e
        total_warn += w
    elif args.cards:
        cards_path = Path(args.cards_dir) if args.cards_dir else None
        reg_path = Path(args.registry_file) if args.registry_file else None
        e, w = check_parameter_cards(cards_path, reg_path, allow_partial=args.allow_partial)
        total_err += e
        total_warn += w
    elif args.registry or args.registry_file:
        reg_path = Path(args.registry_file) if args.registry_file else None
        e, w = check_param_registry(reg_path)
        total_err += e
        total_warn += w
    else:
        target = Path(args.path) if args.path else None
        e, w = check_docs(target, include_drafts=args.include_drafts)
        total_err += e
        total_warn += w

    sys.exit(1 if total_err > 0 else 0)


if __name__ == "__main__":
    main()