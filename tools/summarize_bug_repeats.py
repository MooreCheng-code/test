# -*- coding: utf-8 -*-
"""
tools/summarize_bug_repeats.py
Aggregates the 20 simulation repeats from labs/outputs/bugs/bug_XX_repeats.csv
into docs/code_learning/lab_snapshots/bugs_summary.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
BUGS_OUTPUT_DIR = REPO_ROOT / "labs" / "outputs" / "bugs"
SNAPSHOT_FILE = REPO_ROOT / "docs" / "code_learning" / "lab_snapshots" / "bugs_summary.csv"


def summarize_all_bugs() -> pd.DataFrame:
    repeat_files = sorted(BUGS_OUTPUT_DIR.glob("bug_*_repeats.csv"))
    if not repeat_files:
        raise FileNotFoundError(f"No bug repeat files found in {BUGS_OUTPUT_DIR}")

    summary_rows = []
    for f in repeat_files:
        df = pd.read_csv(f)
        # Extract bug_id like bug_01, bug_02
        stem = f.stem.replace("_repeats", "")
        parts = stem.split("_")
        bug_id = f"{parts[0]}_{parts[1]}"  # e.g. bug_01
        
        metric = str(df["metric"].iloc[0])
        n_repeats = len(df)
        b_vals = df["buggy"].to_numpy(dtype=float)
        c_vals = df["clean"].to_numpy(dtype=float)
        deltas = b_vals - c_vals

        b_mean = float(np.mean(b_vals))
        b_std = float(np.std(b_vals, ddof=1)) if n_repeats > 1 else 0.0
        c_mean = float(np.mean(c_vals))
        c_std = float(np.std(c_vals, ddof=1)) if n_repeats > 1 else 0.0

        d_mean = float(np.mean(deltas))
        d_std = float(np.std(deltas, ddof=1)) if n_repeats > 1 else 0.0
        d_se = d_std / np.sqrt(n_repeats) if n_repeats > 0 else 0.0

        t_crit = float(stats.t.ppf(0.975, df=n_repeats - 1)) if n_repeats > 1 else 1.96
        ci_lower = d_mean - t_crit * d_se
        ci_upper = d_mean + t_crit * d_se

        frac_pos = float(np.mean(deltas > 0))

        summary_rows.append({
            "bug_id": bug_id,
            "metric": metric,
            "n_repeats": n_repeats,
            "buggy_mean": b_mean,
            "buggy_std": b_std,
            "clean_mean": c_mean,
            "clean_std": c_std,
            "delta_mean": d_mean,
            "delta_se": d_se,
            "delta_ci_lower": ci_lower,
            "delta_ci_upper": ci_upper,
            "frac_delta_positive": frac_pos,
        })

    sum_df = pd.DataFrame(summary_rows)
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sum_df.to_csv(SNAPSHOT_FILE, index=False)
    print(f"Saved summary of {len(sum_df)} bugs to {SNAPSHOT_FILE.relative_to(REPO_ROOT)}")
    return sum_df


if __name__ == "__main__":
    df = summarize_all_bugs()
    print("\n=== bugs_summary.csv ===")
    print(df.to_string())
