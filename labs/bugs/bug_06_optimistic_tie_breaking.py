# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

METRIC_NAME = "Alerts"


def eval_single(seed: int = 42) -> tuple[float, float]:
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=5,
        weights=[0.95, 0.05], random_state=seed
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    # 淺層決策樹會產生大量離散同分
    clf = DecisionTreeClassifier(max_depth=3, random_state=seed)
    clf.fit(X_tr, y_tr)

    prob_te = clf.predict_proba(X_te)[:, 1]
    target_recall = 0.90
    pos_scores = prob_te[y_te == 1]
    cutoff = float(np.percentile(pos_scores, (1.0 - target_recall) * 100.0))

    # 1. 實作 A (待檢視)：使用嚴格大於（> cutoff），臨界同分被漏抓，預警數少但召回率失真
    alerts_buggy = float(np.sum(prob_te > cutoff))

    # 2. 正確版：使用大於等於（>= cutoff），保守納入臨界同分，確保召回率達標
    alerts_clean = float(np.sum(prob_te >= cutoff))

    return alerts_buggy, alerts_clean


def run_experiment():
    b_val, c_val = eval_single(seed=42)
    print("=== Bug 06: Tie Breaking Policy ===")
    print(f"Buggy Alerts (strict > cutoff, ties excluded):     {b_val:.0f}")
    print(f"Clean Alerts (conservative >= cutoff, ties included): {c_val:.0f}")
    print(f"Delta (Alerts Difference): {b_val - c_val:+.0f}")


def run_repeats(n_repeats: int = 20):
    out_dir = Path("labs/outputs/bugs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bug_06_repeats.csv"
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
