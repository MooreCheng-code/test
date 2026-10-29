#!/usr/bin/env python3
"""
tools/build_param_cards.py
Automated table injector for parameter cards in docs/code_learning/03_參數手冊/.

Reads:
- docs/code_learning/03a_param_registry.csv

Injects markdown tables into parameter cards containing:
<!-- registry:ids=ID1,ID2,... -->
...
<!-- /registry -->

Supports:
- python tools/build_param_cards.py           # Injects/updates all card tables
- python tools/build_param_cards.py --check   # Verifies tables are up-to-date (exit 1 if diff)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "code_learning"
REGISTRY_CSV = DOCS_DIR / "03a_param_registry.csv"
CARDS_DIR = DOCS_DIR / "03_參數手冊"

SECTION_PATTERN = re.compile(
    r"(<!--\s*registry:ids=([A-Za-z0-9_,\s]+)\s*-->)(.*?)(<!--\s*/registry\s*-->)",
    re.DOTALL
)


def get_module_or_model(param_id: str) -> str:
    """Derives a human-readable model/module location from param_id."""
    prefix = param_id.split("_")[0]
    if prefix in {"M1", "M2", "M3", "M4", "M5"}:
        return prefix
    mapping = {
        "CFG": "全域 config",
        "PREP": "前處理模組",
        "PIPE": "管線構建",
        "IMB": "不平衡重抽樣",
        "TUNE": "調參模組",
        "ENS": "集成模組",
        "TH": "門檻模組",
        "CAL": "機率校準",
        "EXP": "模型解釋",
        "DEF": "前端 defaults",
        "E2A": "E2a 調參",
        "E2B": "E2b 調參",
        "E6": "E6 盲測",
        "SENS": "敏感度分析",
        "SCORE": "推論評分",
        "CV": "交叉驗證",
    }
    return mapping.get(prefix, prefix)


def load_registry_map() -> dict[str, dict[str, str]]:
    """Loads docs/code_learning/03a_param_registry.csv into {id: row}."""
    if not REGISTRY_CSV.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_CSV}")
    res = {}
    with open(REGISTRY_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            res[row["id"].strip()] = row
    return res


def build_table_for_ids(ids: list[str], reg_map: dict[str, dict[str, str]]) -> str:
    """Generates the Markdown table string for the specified IDs."""
    headers = ["登錄 ID", "模型／處", "參數名", "目前值 / 固化值", "合理範圍 / 搜尋空間", "所在檔案:行號", "可調性"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for pid in ids:
        if pid not in reg_map:
            raise KeyError(f"ID '{pid}' referenced in card not found in {REGISTRY_CSV}")
        r = reg_map[pid]
        mod = get_module_or_model(pid)
        file_line = f"{r['所在檔案']}:{r['行號']}"
        row = [
            f"`{pid}`",
            mod,
            f"`{r['參數名']}`",
            f"`{r['目前值']}`" if (" " not in r["目前值"] or r["目前值"].startswith("[")) else r["目前值"],
            r["合理範圍"],
            f"`{file_line}`",
            r["可調性"],
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n" + "\n".join(lines) + "\n"


def process_card_file(file_path: Path, reg_map: dict[str, dict[str, str]], check_only: bool = False) -> tuple[int, bool]:
    """Processes a single card markdown file. Returns (tables_updated_or_checked, has_diff)."""
    text = file_path.read_text(encoding="utf-8")
    
    def replacer(match: re.Match) -> str:
        tag_open = match.group(1)
        raw_ids = match.group(2)
        tag_close = match.group(4)
        id_list = [i.strip() for i in raw_ids.split(",") if i.strip()]
        new_table = build_table_for_ids(id_list, reg_map)
        return f"{tag_open}{new_table}{tag_close}"

    new_text, count = SECTION_PATTERN.subn(replacer, text)
    if count == 0:
        return 0, False

    has_diff = (new_text != text)
    if has_diff and not check_only:
        file_path.write_text(new_text, encoding="utf-8")

    return count, has_diff


def main():
    parser = argparse.ArgumentParser(description="Automated table injector for parameter cards.")
    parser.add_argument("--check", action="store_true", help="Check that cards are up-to-date without writing.")
    parser.add_argument("--dir", type=str, default=None, help="Custom cards directory to process.")
    args = parser.parse_args()

    reg_map = load_registry_map()
    target_dir = Path(args.dir) if args.dir else CARDS_DIR
    if not target_dir.exists():
        print(f"Cards directory does not exist: {target_dir}")
        sys.exit(0)

    card_files = sorted(list(target_dir.glob("*.md")) + list(target_dir.glob("*/*.md")))
    if not card_files:
        print(f"No markdown card files found in {target_dir}")
        sys.exit(0)

    total_tables = 0
    files_with_diff = []

    for f in card_files:
        try:
            count, has_diff = process_card_file(f, reg_map, check_only=args.check)
            total_tables += count
            if has_diff:
                files_with_diff.append(f)
        except Exception as e:
            print(f"[{f.name}] [ERROR] {e}")
            sys.exit(1)

    if args.check:
        if files_with_diff:
            print(f"[ERROR] {len(files_with_diff)} card file(s) have out-of-date tables:")
            for f in files_with_diff:
                print(f"  - {f.relative_to(REPO_ROOT)}")
            sys.exit(1)
        else:
            print(f"[OK] All {total_tables} card table(s) in {len(card_files)} file(s) match registry.")
    else:
        print(f"Successfully updated {total_tables} table(s) across {len(card_files)} card file(s).")
        if files_with_diff:
            print(f"Modified {len(files_with_diff)} file(s).")


if __name__ == "__main__":
    main()
