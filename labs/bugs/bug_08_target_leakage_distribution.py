# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

METRIC_NAME = "Alerts"


def eval_single(seed: int = 42) -> tuple[float, float]:
    # 訓練集破產盛行率為 5.0%
    X_tr, y_tr = make_classification(
        n_samples=500, n_features=10, weights=[0.95, 0.05], random_state=seed
    )
    # 測試集在現實情境中盛行率偏低（2.0%），反映不同經濟週期或部署情境
    X_te, y_te = make_classification(
        n_samples=300, n_features=10, weights=[0.98, 0.02], random_state=seed + 999
    )

    clf = LogisticRegression(random_state=seed)
    clf.fit(X_tr, y_tr)
    prob_te = clf.predict_proba(X_te)[:, 1]

    # 1. 作法 A (犯錯版)：偷看未標記測試集的真實盛行率（2.0%）來動態調整門檻（資料分佈洩漏）
    test_prev_leaked = float(np.mean(y_te))
    th_leaked = float(np.quantile(prob_te, 1.0 - test_prev_leaked))
    alerts_buggy = float(np.sum(prob_te >= th_leaked))

    # 2. 作法 B (正確版)：嚴格僅依據訓練集已知先驗盛行率（5.0%）決定生產固定門檻
    train_prev_prior = float(np.mean(y_tr))
    th_clean = float(np.quantile(prob_te, 1.0 - train_prev_prior))
    alerts_clean = float(np.sum(prob_te >= th_clean))

    return alerts_buggy, alerts_clean


def run_experiment():
    b_val, c_val = eval_single(seed=42)
    print("=== Bug 08: Target Distribution Leakage ===")
    print(f"Buggy Alerts (test prevalence leaked): {b_val:.0f}")
    print(f"Clean Alerts (training prior fixed):   {c_val:.0f}")
    print(f"Delta: {b_val - c_val:+.0f}")


def run_repeats(n_repeats: int = 20):
    out_dir = Path("labs/outputs/bugs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bug_08_repeats.csv"
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
