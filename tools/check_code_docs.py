#!/usr/bin/env python3
"""
tools/check_code_docs.py
Minimal consistency and compliance checker for docs/code_learning/.

Checks:
1. Citations: file:line references exist and line numbers are within bounds.
2. Clean paths: No absolute paths, no file:/// URIs, no Windows user directories.
3. Language: No exaggerated, derogatory, or absolutist claims (e.g. 完勝, 唯一正道, 100% 保證).
"""

import os
import re
import sys
from pathlib import Path

# Repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "code_learning"

# Patterns
# Match citations like bankruptcy/src/cv.py:258 or bankruptcy/config.py:24-25
CITATION_PATTERN = re.compile(
    r"(bankruptcy/[\w/.-]+\.py):(\d+)(?:-(\d+))?"
)

# Forbidden absolute path patterns
ABS_PATH_PATTERNS = [
    re.compile(r"file:///", re.IGNORECASE),
    re.compile(r"[a-zA-Z]:[/\\]Users[/\\]", re.IGNORECASE),
    re.compile(r"/Users/\w+/", re.IGNORECASE),
]

# Forbidden exaggerated / absolutist terms (Rule 8 in guidelines)
FORBIDDEN_WORDS = [
    "完勝",
    "唯一正道",
    "唯一正解",
    "100% 保證",
    "100%保證",
    "絕無可能失敗",
]

# Strict 15-fold misnomer (must use 5 折 × 3 次重複 = 15 次外層評估)
MISNOMER_15FOLD = re.compile(r"(?<![35\d])15\s*折(?!次)")


def check_docs() -> tuple[int, int]:
    """Scans all markdown files in docs/code_learning/ and returns (n_errors, n_warnings)."""
    errors = 0
    warnings = 0

    if not DOCS_DIR.exists():
        print(f"[ERROR] Docs directory not found: {DOCS_DIR}")
        return 1, 0

    md_files = list(DOCS_DIR.rglob("*.md"))
    print(f"Checking {len(md_files)} markdown files in {DOCS_DIR}...\n")

    for md_path in sorted(md_files):
        rel_md = md_path.relative_to(REPO_ROOT)
        content = md_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # 1. Check forbidden absolute paths
        for idx, line in enumerate(lines, 1):
            for pat in ABS_PATH_PATTERNS:
                if pat.search(line):
                    print(f"[{rel_md}:{idx}] [ERROR] Absolute or user path detected: {line.strip()[:80]}")
                    errors += 1

        # 2. Check forbidden exaggerated terms
        for idx, line in enumerate(lines, 1):
            for word in FORBIDDEN_WORDS:
                if word in line:
                    print(f"[{rel_md}:{idx}] [ERROR] Forbidden exaggerated term '{word}': {line.strip()[:80]}")
                    errors += 1

        # 3. Check 15-fold misnomers (except in _findings.md where it critiques historical claims)
        if md_path.name != "_findings.md":
            for idx, line in enumerate(lines, 1):
                # allow explanatory sentences that explicitly warn against using '15 折'
                if MISNOMER_15FOLD.search(line) and "絕不能稱為" not in line and "嚴禁" not in line:
                    print(f"[{rel_md}:{idx}] [WARNING] Misnomer '15 折' detected (use '5 折 × 3 次重複 = 15 次外層評估'): {line.strip()[:80]}")
                    warnings += 1

        # 4. Check citations
        for idx, line in enumerate(lines, 1):
            matches = CITATION_PATTERN.findall(line)
            for m in matches:
                target_file_rel, start_line, end_line = m[0], int(m[1]), int(m[2]) if m[2] else int(m[1])
                target_file = REPO_ROOT / target_file_rel

                if not target_file.exists():
                    print(f"[{rel_md}:{idx}] [ERROR] Cited file does not exist: {target_file_rel}")
                    errors += 1
                    continue

                target_lines = target_file.read_text(encoding="utf-8").splitlines()
                n_target_lines = len(target_lines)

                if start_line > n_target_lines or start_line <= 0:
                    print(f"[{rel_md}:{idx}] [ERROR] Cited line {start_line} out of range in {target_file_rel} (1-{n_target_lines})")
                    errors += 1

    print("\n" + "=" * 50)
    print(f"Check completed: {errors} error(s), {warnings} warning(s).")
    print("=" * 50)
    return errors, warnings


if __name__ == "__main__":
    n_err, n_warn = check_docs()
    sys.exit(1 if n_err > 0 else 0)
