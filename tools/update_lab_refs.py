"""
update_lab_refs.py - 自動更新參數卡片與速查表中的 Lab 狀態引用

【功能】
1. 將 labs/lab_map.csv 中所有 Lab 狀態更新為「已完成」。
2. 將 docs/code_learning/03_參數卡片/*.md 中的「（規劃中：lab_XX）」更新為「（lab_XX，已完成：見 labs/README.md）」。
3. 將 docs/code_learning/03c_速查表.md 中的「（規劃中：lab_XX）」更新為「（lab_XX，已完成：見 labs/README.md）」。
4. 同步更新 docs/code_learning/_cards_audit.csv。
"""

from __future__ import annotations

import csv
import glob
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABS_DIR = REPO_ROOT / "labs"
DOCS_DIR = REPO_ROOT / "docs" / "code_learning"
CARDS_DIR = DOCS_DIR / "03_參數卡片"
MAP_CSV = LABS_DIR / "lab_map.csv"
CHEATSHEET_MD = DOCS_DIR / "03c_速查表.md"
AUDIT_CSV = DOCS_DIR / "_cards_audit.csv"


def update_lab_map():
    if not MAP_CSV.exists():
        print(f"[ERROR] {MAP_CSV} does not exist.")
        return

    with open(MAP_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated_rows = []
    for r in rows:
        r["狀態"] = "已完成"
        updated_rows.append(r)

    with open(MAP_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"[已更新] {MAP_CSV} 全數標記為「已完成」({len(updated_rows)} 筆)。")


def replace_planned_refs(text: str) -> tuple[str, int]:
    # 匹配 （規劃中：lab_XX） 或 （規劃中:lab_XX）
    pattern = r'（規劃中[：:](lab_\d+)）'
    count = len(re.findall(pattern, text))
    new_text = re.sub(pattern, r'（\1，已完成：見 labs/README.md）', text)
    return new_text, count


def update_markdown_files():
    total_replaced = 0

    # 1. 03c_速查表.md
    if CHEATSHEET_MD.exists():
        content = CHEATSHEET_MD.read_text(encoding="utf-8")
        new_content, count = replace_planned_refs(content)
        if count > 0:
            CHEATSHEET_MD.write_text(new_content, encoding="utf-8")
            print(f"[已更新] {CHEATSHEET_MD.name}: 替換了 {count} 處 Lab 引用。")
            total_replaced += count

    # 2. 03_參數手冊/*.md 或 03_參數卡片/*.md
    card_dirs = [d for d in DOCS_DIR.iterdir() if d.is_dir() and d.name.startswith("03_")]
    for c_dir in card_dirs:
        for card_path in sorted(c_dir.glob("*.md")):
            content = card_path.read_text(encoding="utf-8")
            new_content, count = replace_planned_refs(content)
            if count > 0:
                card_path.write_text(new_content, encoding="utf-8")
                print(f"[已更新] {card_path.name}: 替換了 {count} 處 Lab 引用。")
                total_replaced += count

    # 3. _cards_audit.csv
    if AUDIT_CSV.exists():
        content = AUDIT_CSV.read_text(encoding="utf-8")
        new_content, count = replace_planned_refs(content)
        if count > 0:
            AUDIT_CSV.write_text(new_content, encoding="utf-8")
            print(f"[已更新] {AUDIT_CSV.name}: 替換了 {count} 處 Lab 引用。")
            total_replaced += count

    print(f"[完成] 全部檔案共替換了 {total_replaced} 處規劃中標記。")


def main():
    print("=" * 70)
    print("正在執行 Lab 引用狀態自動更新 (規劃中 -> 已完成)...")
    print("=" * 70)
    update_lab_map()
    update_markdown_files()


if __name__ == "__main__":
    main()
