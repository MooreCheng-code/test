# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

METRIC_NAME = "PR-AUC"


def eval_single(seed: int = 42) -> tuple[float, float]:
    X, y = make_classification(
        n_samples=500, n_features=60, n_informative=6,
        weights=[0.95, 0.05], random_state=seed
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 1. 實作 A (待檢視)：全域特徵選擇後再切折（特徵洩漏）
    selector_leak = SelectKBest(score_func=f_classif, k=10)
    X_leaked = selector_leak.fit_transform(X, y)
    scores_buggy = []
    for tr, val in skf.split(X_leaked, y):
        clf = LogisticRegression(random_state=seed, max_iter=200)
        clf.fit(X_leaked[tr], y[tr])
        prob = clf.predict_proba(X_leaked[val])[:, 1]
        scores_buggy.append(average_precision_score(y[val], prob))

    # 2. 正確版：嚴格在訓練折內 fit 特徵選擇器，驗證折僅 transform
    scores_clean = []
    for tr, val in skf.split(X, y):
        selector_clean = SelectKBest(score_func=f_classif, k=10)
        X_tr = selector_clean.fit_transform(X[tr], y[tr])
        X_val = selector_clean.transform(X[val])
        clf = LogisticRegression(random_state=seed, max_iter=200)
        clf.fit(X_tr, y[tr])
        prob = clf.predict_proba(X_val)[:, 1]
        scores_clean.append(average_precision_score(y[val], prob))

    return float(np.mean(scores_buggy)), float(np.mean(scores_clean))


def run_experiment():
    b_val, c_val = eval_single(seed=42)
    print("=== Bug 01: Preprocessing Leakage ===")
    print(f"Buggy PR-AUC (with global feature selection leakage): {b_val:.4f}")
    print(f"Clean PR-AUC (in-fold isolated feature selection):    {c_val:.4f}")
    print(f"Delta (Optimistic Bias): {b_val - c_val:+.4f}")


def run_repeats(n_repeats: int = 20):
    out_dir = Path("labs/outputs/bugs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bug_01_repeats.csv"
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
