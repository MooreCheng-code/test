# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================
# 題名：除錯第 2 題：重採樣時機不當導致之合成樣本跨折滲透

from pathlib import Path
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

METRIC_NAME = "PR-AUC"


def eval_single(seed: int = 42) -> tuple[float, float]:
    X, y = make_classification(
        n_samples=500, n_features=16, n_informative=8,
        weights=[0.95, 0.05], random_state=seed
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 1. 實作 A (待檢視)：全域先做 SMOTE 再切折（合成樣本跨折滲透）
    smote_leak = SMOTE(random_state=seed)
    X_res, y_res = smote_leak.fit_resample(X, y)
    scores_buggy = []
    for tr, val in skf.split(X_res, y_res):
        clf = RandomForestClassifier(n_estimators=30, random_state=seed)
        clf.fit(X_res[tr], y_res[tr])
        prob = clf.predict_proba(X_res[val])[:, 1]
        scores_buggy.append(average_precision_score(y_res[val], prob))

    # 2. 正確版：僅在訓練折內做 SMOTE，驗證折保持原始真實樣本
    scores_clean = []
    for tr, val in skf.split(X, y):
        smote_clean = SMOTE(random_state=seed)
        X_tr_res, y_tr_res = smote_clean.fit_resample(X[tr], y[tr])
        clf = RandomForestClassifier(n_estimators=30, random_state=seed)
        clf.fit(X_tr_res, y_tr_res)
        prob = clf.predict_proba(X[val])[:, 1]
        scores_clean.append(average_precision_score(y[val], prob))

    return float(np.mean(scores_buggy)), float(np.mean(scores_clean))


def run_experiment():
    b_val, c_val = eval_single(seed=42)
    print("=== Bug 02: Resampling Leakage ===")
    print(f"Buggy PR-AUC (global SMOTE with synthetic leakage): {b_val:.4f}")
    print(f"Clean PR-AUC (in-fold isolated training SMOTE):     {c_val:.4f}")
    print(f"Delta (Inflated PR-AUC): {b_val - c_val:+.4f}")


def run_repeats(n_repeats: int = 20):
    out_dir = Path("labs/outputs/bugs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bug_02_repeats.csv"
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
