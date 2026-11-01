"""tools/run_labs_smoke.py - Smoke test runner for lab_01 through lab_09.

Runs each lab in --quick mode with --pipeline project,
verifies zero exit codes, checks output file generation and git_dirty status,
and writes docs/code_learning/lab_snapshots/_smoke_report.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

LAB_SCRIPTS = [
    ("lab_01", "labs/lab_01_tree_depth.py"),
    ("lab_02", "labs/lab_02_forest_hyperparams.py"),
    ("lab_03", "labs/lab_03_boosting_f02.py"),
    ("lab_04", "labs/lab_04_imbalance_strategies.py"),
    ("lab_05", "labs/lab_05_preprocessing_ablation.py"),
    ("lab_06", "labs/lab_06_recall_alert_curve.py"),
    ("lab_07", "labs/lab_07_cv_folds_repeats.py"),
    ("lab_08", "labs/lab_08_seed_stability.py"),
    ("lab_09", "labs/lab_09_tree_count_ablation.py"),
]


def get_git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
        ).strip()
        return out
    except Exception:
        return "UNKNOWN"


def get_git_dirty_status() -> tuple[bool, int]:
    """Check if labs/ or tools/ have uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT),
            text=True,
        )
        labs_dirty = False
        other_dirty = 0
        for line in out.splitlines():
            if not line.strip():
                continue
            if line.startswith("??"):
                other_dirty += 1
                continue
            status = line[:2]
            path = line[3:].strip().strip('"')
            if path.startswith("labs/") or path.startswith("tools/"):
                labs_dirty = True
            else:
                other_dirty += 1
        return labs_dirty, other_dirty
    except Exception:
        return True, 0


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="backslashreplace").decode("ascii"))


def main():
    head_sha = get_git_head()
    overall_dirty, other_count = get_git_dirty_status()

    safe_print("=" * 80)
    safe_print(f"Running Labs Smoke Suite on HEAD: {head_sha} (Git Dirty: {overall_dirty})")
    safe_print("=" * 80)

    report_rows = []
    overall_success = True

    for lab_id, script_rel in LAB_SCRIPTS:
        script_path = REPO_ROOT / script_rel
        if not script_path.is_file():
            safe_print(f"[FAIL] Script not found: {script_rel}")
            overall_success = False
            continue

        # Run only --quick --pipeline project
        pipe = "project"
        cmd = [sys.executable, str(script_path), "--quick", "--pipeline", pipe]
        safe_print(f"\n>> Executing: {' '.join(cmd)}")
        t0 = time.time()
        res = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - t0
        exit_code = res.returncode

        # Output artifacts check & manifest git_dirty read
        out_dir = REPO_ROOT / "labs" / "outputs" / lab_id
        out_files = list(out_dir.glob("*")) if out_dir.is_dir() else []
        manifest_file = out_dir / "run_manifest.json"
        lab_git_dirty = overall_dirty
        if manifest_file.is_file():
            try:
                with open(manifest_file, "r", encoding="utf-8") as mf:
                    m_data = json.load(mf)
                    lab_git_dirty = m_data.get("git_dirty", overall_dirty)
            except Exception:
                pass

        status_str = "PASS" if exit_code == 0 else "FAIL"
        safe_print(f"   [{status_str}] ExitCode={exit_code}, Elapsed={elapsed:.2f}s, Artifacts={len(out_files)}, GitDirty={lab_git_dirty}")

        if exit_code != 0:
            safe_print("   --- STDERR TAIL ---")
            tail_err = "\n".join(res.stderr.splitlines()[-10:])
            safe_print(tail_err)
            safe_print("   --- STDOUT TAIL ---")
            tail_out = "\n".join(res.stdout.splitlines()[-10:])
            safe_print(tail_out)
            overall_success = False

        report_rows.append({
            "lab": lab_id,
            "script": script_rel,
            "pipeline": pipe,
            "exit_code": exit_code,
            "elapsed_sec": f"{elapsed:.2f}",
            "artifact_count": len(out_files),
            "head": head_sha,
            "git_dirty": str(lab_git_dirty),
        })

    # Generate Markdown Report
    report_path = REPO_ROOT / "docs" / "code_learning" / "lab_snapshots" / "_smoke_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Labs 冒煙測試執行報告 (Smoke Report)",
        "",
        f"- **測試基準 Commit (HEAD)**: `{head_sha}`",
        f"- **Git Dirty 狀態**: `{'Dirty' if overall_dirty else 'Clean (False)'}`",
        f"- **執行時間**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **整體狀態**: `{'全數通過 (PASS)' if overall_success else '存在失敗 (FAIL)'}`",
        "",
        "| Lab 代號 | 腳本路徑 | 管線模式 | 退出碼 | 執行秒數 | 產物數量 | Commit HEAD | Git Dirty |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in report_rows:
        lines.append(
            f"| `{r['lab']}` | `{r['script']}` | `{r['pipeline']}` | `{r['exit_code']}` | {r['elapsed_sec']}s | {r['artifact_count']} 檔 | `{r['head']}` | `{r['git_dirty']}` |"
        )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    safe_print("\n" + "=" * 80)
    safe_print(f"Smoke report written to: {report_path.relative_to(REPO_ROOT)}")
    safe_print("=" * 80)

    # Restore lab_snapshots CSVs to avoid overwriting standard snapshots with quick runs
    subprocess.run(
        ["git", "checkout", "HEAD", "--", "docs/code_learning/lab_snapshots/*.csv"],
        cwd=str(REPO_ROOT),
        capture_output=True,
    )

    if not overall_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
