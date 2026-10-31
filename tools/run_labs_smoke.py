"""tools/run_labs_smoke.py - Smoke test runner for lab_01 through lab_09.

Runs each lab in --quick mode with --pipeline project and --pipeline teaching,
verifies zero exit codes, checks output file generation, and writes
docs/code_learning/lab_snapshots/_smoke_report.md.
"""

from __future__ import annotations

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


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="backslashreplace").decode("ascii"))


def main():
    head_sha = get_git_head()
    safe_print("=" * 80)
    safe_print(f"Running Labs Smoke Suite on HEAD: {head_sha}")
    safe_print("=" * 80)

    report_rows = []
    overall_success = True

    for lab_id, script_rel in LAB_SCRIPTS:
        script_path = REPO_ROOT / script_rel
        if not script_path.is_file():
            safe_print(f"[FAIL] Script not found: {script_rel}")
            overall_success = False
            continue

        # Test both project and teaching pipelines
        for pipe in ["project", "teaching"]:
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

            # Output artifacts check
            out_dir = REPO_ROOT / "labs" / "outputs" / lab_id
            out_files = list(out_dir.glob("*")) if out_dir.is_dir() else []

            status_str = "PASS" if exit_code == 0 else "FAIL"
            safe_print(f"   [{status_str}] ExitCode={exit_code}, Elapsed={elapsed:.2f}s, OutputArtifacts={len(out_files)}")

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
            })

    # Generate Markdown Report
    report_path = REPO_ROOT / "docs" / "code_learning" / "lab_snapshots" / "_smoke_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Labs 冒煙測試執行報告 (Smoke Report)",
        "",
        f"- **測試基準 Commit (HEAD)**: `{head_sha}`",
        f"- **執行時間**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **整體狀態**: `{'全數通過 (PASS)' if overall_success else '存在失敗 (FAIL)'}`",
        "",
        "| Lab 代號 | 腳本路徑 | 管線模式 | 退出碼 | 執行秒數 | 產物數量 | Commit HEAD |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for r in report_rows:
        lines.append(
            f"| `{r['lab']}` | `{r['script']}` | `{r['pipeline']}` | `{r['exit_code']}` | {r['elapsed_sec']}s | {r['artifact_count']} 檔 | `{r['head']}` |"
        )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    safe_print("\n" + "=" * 80)
    safe_print(f"Smoke report written to: {report_path.relative_to(REPO_ROOT)}")
    safe_print("=" * 80)

    if not overall_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
