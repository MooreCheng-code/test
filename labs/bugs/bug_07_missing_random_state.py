# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

METRIC_NAME = "PR-AUC Variation"


def eval_single(seed: int = 42) -> tuple[float, float]:
    X, y = make_classification(
        n_samples=500, n_features=12,
        weights=[0.95, 0.05], random_state=seed
    )

    # 1. 實作 A (待檢視)：未傳入固定隨機種子，兩次執行產生隨機漂移
    skf1 = StratifiedKFold(n_splits=5, shuffle=True)
    sc1 = []
    for tr, te in skf1.split(X, y):
        m = RandomForestClassifier(n_estimators=25)
        m.fit(X[tr], y[tr])
        sc1.append(average_precision_score(y[te], m.predict_proba(X[te])[:, 1]))

    skf2 = StratifiedKFold(n_splits=5, shuffle=True)
    sc2 = []
    for tr, te in skf2.split(X, y):
        m = RandomForestClassifier(n_estimators=25)
        m.fit(X[tr], y[tr])
        sc2.append(average_precision_score(y[te], m.predict_proba(X[te])[:, 1]))

    drift_buggy = float(abs(np.mean(sc1) - np.mean(sc2)))

    # 2. 正確版：傳入固定種子，兩次評估完全確定性重現，漂移為 0
    skf_c1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    sc_c1 = []
    for tr, te in skf_c1.split(X, y):
        m = RandomForestClassifier(n_estimators=25, random_state=seed)
        m.fit(X[tr], y[tr])
        sc_c1.append(average_precision_score(y[te], m.predict_proba(X[te])[:, 1]))

    skf_c2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    sc_c2 = []
    for tr, te in skf_c2.split(X, y):
        m = RandomForestClassifier(n_estimators=25, random_state=seed)
        m.fit(X[tr], y[tr])
        sc_c2.append(average_precision_score(y[te], m.predict_proba(X[te])[:, 1]))

    drift_clean = float(abs(np.mean(sc_c1) - np.mean(sc_c2)))

    return drift_buggy, drift_clean


def run_experiment():
    b_val, c_val = eval_single(seed=42)
    print("=== Bug 07: Missing Random State ===")
    print(f"Buggy PR-AUC Run-to-Run Drift (no seed):     {b_val:.4f}")
    print(f"Clean PR-AUC Run-to-Run Drift (locked seed): {c_val:.4f}")
    print(f"Delta (Instability): {b_val - c_val:+.4f}")


def run_repeats(n_repeats: int = 20):
    out_dir = Path("labs/outputs/bugs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bug_07_repeats.csv"
    rows = []
    for i in range(n_repeats):
        s = 100 + i * 17
        b, c = eval_single(seed=s)
        rows.append({"seed": s, "buggy": b, "clean": c, "metric": METRIC_NAME})
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} repeats to {out_file.as_posix()}")


if __name__ == "__main__":
    run_experiment()
    run_repeats()
