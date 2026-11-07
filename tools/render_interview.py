#!/usr/bin/env python3
"""
tools/render_interview.py
Renders interview draft markdown from a template with code-interpolated numbers from CSVs and JSONs.
Extracts numbers from CSV/JSON files, evaluates arithmetic, verifies against bare metric numbers,
and writes out docs/code_learning/_drafts/09_面試說明稿.md and 09_numbers_used.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = REPO_ROOT / "docs" / "code_learning" / "_drafts" / "09_面試說明稿.template.md"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "code_learning" / "_drafts" / "09_面試說明稿.md"
DEFAULT_CSV_OUT = REPO_ROOT / "docs" / "code_learning" / "_drafts" / "09_numbers_used.csv"

# Regex patterns for placeholders
# {{csv:file_path|where_expr|col_name|fmt}}
CSV_PLACEHOLDER_PATTERN = re.compile(
    r"\{\{csv:([^|{}]+)\|([^|{}]+)\|([^|{}]+)\|([^|{}]+)\}\}"
)

# {{calc:expr|csv:file_path|where_expr|fmt}}
CALC_PLACEHOLDER_PATTERN = re.compile(
    r"\{\{calc:([^|{}]+)\|csv:([^|{}]+)\|([^|{}]+)\|([^|{}]+)\}\}"
)

# {{json:file_path|key_path|fmt}}
JSON_PLACEHOLDER_PATTERN = re.compile(
    r"\{\{json:([^|{}]+)\|([^|{}]+)\|([^|{}]+)\}\}"
)

# Whitelist patterns for bare numbers that are allowed in the template
ALLOWED_BARE_PATTERNS = [
    re.compile(r"95%\s*(?:CI|信賴區間|區間)", re.IGNORECASE),
    re.compile(r"1\.5\s*倍(?:目標|預警目標|內部目標)?"),
    re.compile(r"1\s*分鐘版"),
    re.compile(r"5\s*分鐘版"),
    re.compile(r"10\s*個\s*Q&A"),
    re.compile(r"10\s*道"),
    re.compile(r"8\s*道"),
    re.compile(r"8\s*項"),
    re.compile(r"5\s*折"),
    re.compile(r"3\s*次"),
    re.compile(r"15\s*次"),
    re.compile(r"3\s*折"),
    re.compile(r"20\s*次"),
    re.compile(r"45\s*倍"),
    re.compile(r"44\s*家"),
    re.compile(r"1364\s*家"),
    re.compile(r"1320\s*家"),
    re.compile(r"294\s*家"),
    re.compile(r"21\s*家"),
    re.compile(r"66\s*家"),
    re.compile(r"29\s*家"),
    re.compile(r"43\s*家"),
    re.compile(r"40\s*家"),
    re.compile(r"42\s*家"),
    re.compile(r"1364\s*筆"),
    re.compile(r"200\s*筆"),
    re.compile(r"200\s*個"),
    re.compile(r"1091\s*筆"),
    re.compile(r"4364\s*筆"),
    re.compile(r"5455\s*筆"),
    re.compile(r"25\s*次"),
    re.compile(r"38\s*次"),
    re.compile(r"40\s*次"),
    re.compile(r"60\s*次"),
    re.compile(r"72\s*欄"),
    re.compile(r"92\s*欄"),
    re.compile(r"95\s*欄"),
    re.compile(r"20\s*個"),
    re.compile(r"24\s*個"),
    re.compile(r"190\s*項"),
    re.compile(r"193\s*項"),
    re.compile(r"AR@\d+(?:\.\d+)?"),
    re.compile(r"rec@ar_\d+(?:\.\d+)?"),
    re.compile(r"SMOTE\s*0\.\d+"),
    re.compile(r"\b(?:80|90|95|98|100)%\s*(?:召回|召回率)\b"),
    re.compile(r"門檻\s*0\.95"),
    re.compile(r"\bm\s*=\s*0(?:\.0)?\b"),
]


def check_bare_metric_numbers(template_text: str) -> list[str]:
    """Scans the template for bare metric numbers that lack placeholders.
    Returns list of unallowed bare numbers found in non-exempted lines.
    """
    lines = template_text.splitlines()
    unallowed = []
    in_code_block = False

    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # Skip headers, comments, horizontal rules, alerts
        if stripped.startswith("#") or stripped == "---" or stripped.startswith("> [!"):
            continue

        # If line contains exempt comment, skip
        if "<!-- allow-bare-numbers -->" in line or "<!-- no-evidence" in line:
            continue

        # Remove placeholders from line before scanning
        clean = CSV_PLACEHOLDER_PATTERN.sub("", line)
        clean = CALC_PLACEHOLDER_PATTERN.sub("", clean)
        clean = JSON_PLACEHOLDER_PATTERN.sub("", clean)
        clean = re.sub(r"<!--.*?-->", "", clean)
        clean = re.sub(r"\[.*?\]\(.*?\)", "", clean)
        clean = re.sub(r"`.*?`", "", clean)
        clean = re.sub(r"\b[EFMRS]-?\d+[a-z]?\b", "", clean)
        clean = re.sub(r"\b(?:[Ll]ab)[ _]?\d+\b", "", clean)
        clean = re.sub(r"[\w/.-]+\.(?:py|csv|json):\d+(?:-\d+)?", "", clean)
        clean = re.sub(r"^\s*(?:>\s*)?\d+\.\s*", "", clean)
        clean = re.sub(r"^\s*\|\s*\*\*\d+\.\s*", "| **", clean)

        # Check allowed bare patterns and remove them
        for pat in ALLOWED_BARE_PATTERNS:
            clean = pat.sub("", clean)

        # Check for unallowed bare decimals e.g. 0.0255, 3.95, 7.84
        for m in re.finditer(r"(?<![A-Za-z0-9_])([+-]?\d+\.\d+)(?![A-Za-z0-9_%xX倍])", clean):
            num = m.group(1)
            unallowed.append(f"Line {idx}: Bare decimal '{num}'")

        # Check for unallowed percentages e.g. 12.8%
        for m in re.finditer(r"(?<![A-Za-z0-9_])([+-]?\d+(?:\.\d+)?)%", clean):
            num = m.group(0)
            unallowed.append(f"Line {idx}: Bare percentage '{num}'")

        # Check for unallowed multipliers e.g. 7.66x, 7.66 倍
        for m in re.finditer(r"(?<![A-Za-z0-9_])([+-]?\d+(?:\.\d+)?)\s*([xX倍])", clean):
            raw = m.group(0)
            unallowed.append(f"Line {idx}: Bare multiplier '{raw}'")

        # Check for unallowed fractions e.g. 43/44
        for m in re.finditer(r"(?<![A-Za-z0-9_])(\d+/\d+)(?![A-Za-z0-9_])", clean):
            raw = m.group(1)
            unallowed.append(f"Line {idx}: Bare fraction '{raw}'")

    return unallowed


def format_value(raw_val: Any, fmt_str: str) -> str:
    """Formats raw_val using fmt_str supporting %, d, s, f, and signs."""
    if fmt_str.endswith("%"):
        spec = fmt_str.lstrip("%")
        return ("{:" + spec + "}").format(float(raw_val))
    if any(fmt_str.endswith(c) for c in ("d", "i")):
        return fmt_str % int(round(float(raw_val)))
    if fmt_str.endswith("s"):
        return fmt_str % str(raw_val)
    if any(fmt_str.endswith(c) for c in ("f", "e", "g")):
        return fmt_str % float(raw_val)
    return fmt_str % raw_val


def render_template(
    template_text: str, repo_root: Path = REPO_ROOT
) -> tuple[str, list[dict[str, Any]]]:
    """Parses placeholders, interpolates numbers from CSVs/JSONs, and returns rendered text and ledger."""
    # First check bare numbers in template
    bare_errors = check_bare_metric_numbers(template_text)
    if bare_errors:
        err_msg = "Bare metric numbers detected in template without placeholder:\n" + "\n".join(bare_errors[:10])
        if len(bare_errors) > 10:
            err_msg += f"\n... and {len(bare_errors) - 10} more"
        raise ValueError(err_msg)

    numbers_used: list[dict[str, Any]] = []
    csv_cache: dict[str, pd.DataFrame] = {}
    json_cache: dict[str, Any] = {}

    def get_df(file_rel: str) -> pd.DataFrame:
        if file_rel not in csv_cache:
            csv_path = repo_root / file_rel
            if not csv_path.is_file():
                raise FileNotFoundError(f"Evidence CSV file not found: {file_rel}")
            csv_cache[file_rel] = pd.read_csv(csv_path)
        return csv_cache[file_rel]

    def get_json_data(file_rel: str) -> Any:
        if file_rel not in json_cache:
            json_path = repo_root / file_rel
            if not json_path.is_file():
                raise FileNotFoundError(f"Evidence JSON file not found: {file_rel}")
            with open(json_path, "r", encoding="utf-8") as f:
                json_cache[file_rel] = json.load(f)
        return json_cache[file_rel]

    # 1. Process calc placeholders first
    def replace_calc(m: re.Match) -> str:
        full_match = m.group(0)
        expr_str = m.group(1).strip()
        file_rel = m.group(2).strip()
        where_expr = m.group(3).strip()
        fmt_str = m.group(4).strip()

        df = get_df(file_rel)
        try:
            filtered = df.query(where_expr)
        except Exception as e:
            raise ValueError(f"Failed to query '{where_expr}' in {file_rel}: {e}")

        if len(filtered) == 0:
            raise ValueError(f"Query '{where_expr}' returned 0 rows in {file_rel}")
        if len(filtered) > 1:
            raise ValueError(
                f"Query '{where_expr}' returned {len(filtered)} rows in {file_rel}, expected exactly 1"
            )

        row_dict = {}
        for col in df.columns:
            val = filtered[col].iloc[0]
            if isinstance(val, (int, float, np.number)) and not pd.isna(val):
                row_dict[col] = float(val)

        safe_builtins = {"int": int, "float": float, "round": round, "abs": abs, "min": min, "max": max}
        try:
            calc_val = eval(expr_str, {"__builtins__": safe_builtins}, row_dict)
        except Exception as e:
            raise ValueError(f"Failed to evaluate calc expression '{expr_str}' on {file_rel}: {e}")

        try:
            formatted = format_value(calc_val, fmt_str)
        except Exception as e:
            raise ValueError(f"Failed to format calc result {calc_val} with format '{fmt_str}': {e}")

        numbers_used.append(
            {
                "placeholder": full_match,
                "file": file_rel,
                "where": where_expr,
                "col": expr_str,
                "raw_value": calc_val,
                "formatted_value": formatted,
            }
        )
        return formatted

    rendered = CALC_PLACEHOLDER_PATTERN.sub(replace_calc, template_text)

    # 2. Process csv placeholders
    def replace_csv(m: re.Match) -> str:
        full_match = m.group(0)
        file_rel = m.group(1).strip()
        where_expr = m.group(2).strip()
        col_name = m.group(3).strip()
        fmt_str = m.group(4).strip()

        df = get_df(file_rel)
        if col_name not in df.columns:
            raise KeyError(f"Column '{col_name}' not found in {file_rel} (available: {list(df.columns)})")

        try:
            filtered = df.query(where_expr)
        except Exception as e:
            raise ValueError(f"Failed to query '{where_expr}' in {file_rel}: {e}")

        if len(filtered) == 0:
            raise ValueError(f"Query '{where_expr}' returned 0 rows in {file_rel}")
        if len(filtered) > 1:
            raise ValueError(
                f"Query '{where_expr}' returned {len(filtered)} rows in {file_rel}, expected exactly 1"
            )

        raw_val = filtered[col_name].iloc[0]

        try:
            formatted = format_value(raw_val, fmt_str)
        except Exception as e:
            raise ValueError(f"Failed to format value '{raw_val}' with format '{fmt_str}': {e}")

        numbers_used.append(
            {
                "placeholder": full_match,
                "file": file_rel,
                "where": where_expr,
                "col": col_name,
                "raw_value": raw_val,
                "formatted_value": formatted,
            }
        )
        return formatted

    rendered = CSV_PLACEHOLDER_PATTERN.sub(replace_csv, rendered)

    # 3. Process json placeholders
    def replace_json(m: re.Match) -> str:
        full_match = m.group(0)
        file_rel = m.group(1).strip()
        key_path = m.group(2).strip()
        fmt_str = m.group(3).strip()

        data = get_json_data(file_rel)
        cur = data
        for part in key_path.split("."):
            if isinstance(cur, list):
                cur = cur[int(part)]
            elif isinstance(cur, dict):
                if part not in cur:
                    raise KeyError(f"Key '{part}' not found in JSON path '{key_path}' of {file_rel}")
                cur = cur[part]
            else:
                raise KeyError(f"Cannot access '{part}' on non-container in {file_rel}")

        raw_val = cur
        try:
            formatted = format_value(raw_val, fmt_str)
        except Exception as e:
            raise ValueError(f"Failed to format JSON value '{raw_val}' with format '{fmt_str}': {e}")

        numbers_used.append(
            {
                "placeholder": full_match,
                "file": file_rel,
                "where": "N/A",
                "col": key_path,
                "raw_value": raw_val,
                "formatted_value": formatted,
            }
        )
        return formatted

    rendered = JSON_PLACEHOLDER_PATTERN.sub(replace_json, rendered)
    return rendered, numbers_used


def main() -> None:
    parser = argparse.ArgumentParser(description="Render interview draft from template with code interpolation.")
    parser.add_argument("--template", type=str, default=str(DEFAULT_TEMPLATE), help="Template markdown file path.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output rendered markdown path.")
    parser.add_argument("--csv-out", type=str, default=str(DEFAULT_CSV_OUT), help="Output ledger CSV path.")
    args = parser.parse_args()

    tpl_path = Path(args.template)
    if not tpl_path.is_file():
        print(f"[ERROR] Template file not found: {tpl_path}", file=sys.stderr)
        sys.exit(1)

    template_text = tpl_path.read_text(encoding="utf-8")
    try:
        rendered, ledger = render_template(template_text, REPO_ROOT)
    except Exception as e:
        print(f"[ERROR] Template rendering failed: {e}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Successfully rendered markdown to {out_path}")

    csv_out_path = Path(args.csv_out)
    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["placeholder", "file", "where", "col", "raw_value", "formatted_value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ledger)
    print(f"Successfully wrote {len(ledger)} interpolated numbers to {csv_out_path}")


if __name__ == "__main__":
    main()
