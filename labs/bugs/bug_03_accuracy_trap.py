# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, recall_score
from sklearn.model_selection import train_test_split

METRIC_NAME = "PR-AUC"


def eval_single(seed: int = 42) -> tuple[float, float]:
    X, y = make_classification(
        n_samples=1000, n_features=12, n_informative=6,
        weights=[0.97, 0.03], random_state=seed
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    # 1. 實作 A (待檢視)：以 Accuracy 為目標的極端淺樹，幾乎將所有樣本預測為多數類
    clf_buggy = GradientBoostingClassifier(n_estimators=10, max_depth=1, random_state=seed)
    clf_buggy.fit(X_tr, y_tr)
    prob_buggy = clf_buggy.predict_proba(X_te)[:, 1]
    pr_buggy = float(average_precision_score(y_te, prob_buggy))

    # 2. 正確模型：適當調參並關注少數類排序結構
    clf_clean = GradientBoostingClassifier(n_estimators=50, max_depth=3, subsample=0.8, random_state=seed)
    clf_clean.fit(X_tr, y_tr)
    prob_clean = clf_clean.predict_proba(X_te)[:, 1]
    pr_clean = float(average_precision_score(y_te, prob_clean))

    return pr_buggy, pr_clean


def run_experiment():
    seed = 42
    X, y = make_classification(
        n_samples=1000, n_features=12, n_informative=6,
        weights=[0.97, 0.03], random_state=seed
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    clf_buggy = GradientBoostingClassifier(n_estimators=10, max_depth=1, random_state=seed)
    clf_buggy.fit(X_tr, y_tr)
    pred_buggy = clf_buggy.predict(X_te)
    acc_buggy = accuracy_score(y_te, pred_buggy)
    rec_buggy = recall_score(y_te, pred_buggy, zero_division=0)
    pr_buggy = average_precision_score(y_te, clf_buggy.predict_proba(X_te)[:, 1])

    clf_clean = GradientBoostingClassifier(n_estimators=50, max_depth=3, subsample=0.8, random_state=seed)
    clf_clean.fit(X_tr, y_tr)
    pr_clean = average_precision_score(y_te, clf_clean.predict_proba(X_te)[:, 1])

    print("=== Bug 03: Accuracy Trap ===")
    print(f"Buggy Accuracy: {acc_buggy:.4f} (looks very high)")
    print(f"Buggy Recall:   {rec_buggy:.4f} (fails to catch minority bankruptcies)")
    print(f"Buggy PR-AUC:   {pr_buggy:.4f}")
    print(f"Clean PR-AUC:   {pr_clean:.4f}")
    print(f"Delta PR-AUC:   {pr_buggy - pr_clean:+.4f}")


def run_repeats(n_repeats: int = 20):
    out_dir = Path("labs/outputs/bugs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bug_03_repeats.csv"
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
