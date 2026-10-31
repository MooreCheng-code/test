# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================
# 題名：除錯第 10 題：跨折未校準分數直接混合之排序失真

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

METRIC_NAME = "PR-AUC"


def eval_single(seed: int = 42) -> tuple[float, float]:
    X, y = make_classification(
        n_samples=700, n_features=14, n_informative=7,
        weights=[0.95, 0.05], random_state=seed
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    raw_probs_all = []
    y_val_all = []
    fold_scores = []

    for fold, (tr, val) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(random_state=seed)
        clf.fit(X[tr], y[tr])
        probs = clf.predict_proba(X[val])[:, 1]
        
        # 刻意模擬折間機率校準基準線微幅漂移（未校準尺度漂移）
        drifted_probs = probs * (0.85 + (fold % 3) * 0.15)
        
        raw_probs_all.extend(drifted_probs)
        y_val_all.extend(y[val])
        fold_scores.append(average_precision_score(y[val], probs))

    # 1. 實作 A (待檢視)：未經校準直接池化（Pooled）全域計算 PR-AUC
    pooled_pr_auc = float(average_precision_score(y_val_all, raw_probs_all))

    # 2. 正確版：逐折計算 PR-AUC 後平均（避免折間機率尺度漂移破壞排序）
    mean_fold_pr_auc = float(np.mean(fold_scores))

    return pooled_pr_auc, mean_fold_pr_auc


def run_experiment():
    b_val, c_val = eval_single(seed=42)
    print("=== Bug 10: Uncalibrated Score Pooling ===")
    print(f"Buggy Pooled PR-AUC (scale drift across folds): {b_val:.4f}")
    print(f"Clean Mean In-Fold PR-AUC:                      {c_val:.4f}")
    print(f"Delta: {b_val - c_val:+.4f}")


def run_repeats(n_repeats: int = 20):
    out_dir = Path("labs/outputs/bugs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bug_10_repeats.csv"
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
