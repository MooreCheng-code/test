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
from sklearn.model_selection import GridSearchCV, StratifiedKFold

METRIC_NAME = "PR-AUC"


def eval_single(seed: int = 42) -> tuple[float, float]:
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=8,
        weights=[0.95, 0.05], random_state=seed
    )
    param_grid = {"max_depth": [2, 4, 8], "min_samples_split": [2, 5]}

    # 1. 實作 A (待檢視)：非巢狀調參，直接取最佳 CV 驗證折分數作為泛化指標（樂觀選擇偏差）
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    grid_buggy = GridSearchCV(
        RandomForestClassifier(n_estimators=30, random_state=seed),
        param_grid, cv=inner_cv, scoring="average_precision"
    )
    grid_buggy.fit(X, y)
    best_cv_score = float(grid_buggy.best_score_)

    # 2. 正確版：巢狀交叉驗證，外層折未參與超參數挑選，提供無偏泛化估計
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    nested_scores = []
    for tr, te in outer_cv.split(X, y):
        grid_inner = GridSearchCV(
            RandomForestClassifier(n_estimators=30, random_state=seed),
            param_grid, cv=3, scoring="average_precision"
        )
        grid_inner.fit(X[tr], y[tr])
        best_model = grid_inner.best_estimator_
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
