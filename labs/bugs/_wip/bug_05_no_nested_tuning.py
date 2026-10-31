# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

METRIC_NAME = "PR-AUC"


def eval_single(seed: int = 42) -> tuple[float, float]:
    # 250 筆樣本、30 維特徵（含雜訊特徵），10% 破產正樣本
    X, y = make_classification(
        n_samples=250, n_features=30, n_informative=5, n_redundant=5,
        weights=[0.90, 0.10], random_state=seed
    )
    # 搜尋空間覆蓋 60 組隨機超參數，足夠大以展示多選一產生的機會性過度擬合
    param_dist = {
        "max_depth": [2, 3, 4, 6, 8, 12, None],
        "min_samples_leaf": [1, 2, 4, 8, 15],
        "max_features": ["sqrt", "log2", 0.2, 0.5, 0.8, None],
        "criterion": ["gini", "entropy"],
    }

    # 1. 作法 A (犯錯版)：非巢狀調參，直接取多次搜尋的最佳 CV 驗證折分數作為泛化指標（樂觀選擇偏差）
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search_buggy = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=seed),
        param_dist, n_iter=60, cv=inner_cv, scoring="average_precision",
        random_state=seed
    )
    search_buggy.fit(X, y)
    best_cv_score = float(search_buggy.best_score_)

    # 2. 作法 B (正確版)：嚴格巢狀 CV，外層折未參與內層超參數挑選，提供無偏泛化估計
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    nested_scores = []
    for tr, te in outer_cv.split(X, y):
        search_inner = RandomizedSearchCV(
            DecisionTreeClassifier(random_state=seed),
            param_dist, n_iter=60, cv=3, scoring="average_precision",
            random_state=seed
        )
        search_inner.fit(X[tr], y[tr])
        best_model = search_inner.best_estimator_
        prob = best_model.predict_proba(X[te])[:, 1]
        nested_scores.append(average_precision_score(y[te], prob))
    nested_pr_auc = float(np.mean(nested_scores))

    return best_cv_score, nested_pr_auc


def run_experiment():
    b_val, c_val = eval_single(seed=42)
    print("=== Bug 05: Non-Nested Tuning Bias ===")
    print(f"Buggy Best CV PR-AUC (optimistic selection bias): {b_val:.4f}")
    print(f"Clean Nested PR-AUC (unbiased outer evaluation):  {c_val:.4f}")
    print(f"Delta (Optimistic Bias): {b_val - c_val:+.4f}")


def run_repeats(n_repeats: int = 20):
    out_dir = Path("labs/outputs/bugs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bug_05_repeats.csv"
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
