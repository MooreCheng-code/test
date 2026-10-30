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
from sklearn.model_selection import train_test_split

METRIC_NAME = "Recall (percentage points)"


def eval_single(seed: int = 42) -> tuple[float, float]:
    X, y = make_classification(
        n_samples=800, n_features=15, n_informative=6,
        weights=[0.95, 0.05], random_state=seed
    )
    # 50% train, 25% val, 25% test
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=seed
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )

    clf = RandomForestClassifier(n_estimators=40, random_state=seed)
    clf.fit(X_tr, y_tr)

    prob_val = clf.predict_proba(X_val)[:, 1]
    prob_te = clf.predict_proba(X_te)[:, 1]

    target_recall = 0.90
    pos_val = prob_val[y_val == 1]
    cutoff = float(np.percentile(pos_val, (1.0 - target_recall) * 100.0))

    # 1. 實作 A (待檢視)：在挑門檻的同一個驗證集上回報召回率（樣本內過度樂觀）
    rec_val = float(np.sum((prob_val >= cutoff) & (y_val == 1)) / np.sum(y_val == 1)) * 100.0

    # 2. 正確版：將門檻外推至完全獨立的測試集檢驗真實泛化召回率
    rec_te = float(np.sum((prob_te >= cutoff) & (y_te == 1)) / np.sum(y_te == 1)) * 100.0

    return rec_val, rec_te


def run_experiment():
    b_val, c_val = eval_single(seed=42)
    print("=== Bug 04: Validation Threshold Overfitting ===")
    print(f"Buggy In-Sample Val Recall:  {b_val:.2f}%")
    print(f"Clean Out-of-Sample Recall:  {c_val:.2f}%")
    print(f"Delta: {b_val - c_val:+.2f} percentage points")


def run_repeats(n_repeats: int = 20):
    out_dir = Path("labs/outputs/bugs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bug_04_repeats.csv"
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
