# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================
# 題名：除錯第 9 題：切分不一致導致之成對檢定假設破壞

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

METRIC_NAME = "p-value"


def eval_single(seed: int = 42) -> tuple[float, float]:
    X, y = make_classification(
        n_samples=600, n_features=12,
        weights=[0.95, 0.05], random_state=seed
    )

    # 1. 實作 A (待檢視)：Model A 與 Model B 切分器隨機種子不同（未對齊切分），直接做成對 t 檢定
    skf_a = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores_a = []
    for tr, val in skf_a.split(X, y):
        clf_a = RandomForestClassifier(n_estimators=40, random_state=seed)
        clf_a.fit(X[tr], y[tr])
        scores_a.append(average_precision_score(y[val], clf_a.predict_proba(X[val])[:, 1]))

    skf_b_unmatched = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 999)
    scores_b_unmatched = []
    for tr, val in skf_b_unmatched.split(X, y):
        clf_b = GradientBoostingClassifier(n_estimators=40, random_state=seed)
        clf_b.fit(X[tr], y[tr])
        scores_b_unmatched.append(average_precision_score(y[val], clf_b.predict_proba(X[val])[:, 1]))

    _, p_val_unmatched = stats.ttest_rel(scores_a, scores_b_unmatched)

    # 2. 正確版：Model A 與 Model B 嚴格在完全相同的折內進行訓練與驗證（對齊切分）
    scores_b_matched = []
    for tr, val in skf_a.split(X, y):
        clf_b = GradientBoostingClassifier(n_estimators=40, random_state=seed)
        clf_b.fit(X[tr], y[tr])
        scores_b_matched.append(average_precision_score(y[val], clf_b.predict_proba(X[val])[:, 1]))

    _, p_val_matched = stats.ttest_rel(scores_a, scores_b_matched)

    return float(p_val_unmatched), float(p_val_matched)


def run_experiment():
    b_val, c_val = eval_single(seed=42)
    print("=== Bug 09: Unmatched Splits Paired Test ===")
    print(f"Buggy p-value (unmatched splits ttest_rel): {b_val:.4f}")
    print(f"Clean p-value (matched splits ttest_rel):   {c_val:.4f}")
    print(f"Delta: {b_val - c_val:+.4f}")


def run_repeats(n_repeats: int = 20):
    out_dir = Path("labs/outputs/bugs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bug_09_repeats.csv"
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
