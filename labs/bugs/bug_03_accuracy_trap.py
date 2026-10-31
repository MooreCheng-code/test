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
from sklearn.metrics import accuracy_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split

METRIC_NAME = "Accuracy vs Recall"


def eval_single(seed: int = 42) -> tuple[float, float]:
    # 不平衡資料：3% 破產樣本，97% 正常樣本
    X, y = make_classification(
        n_samples=1000, n_features=12, n_informative=2,
        weights=[0.97, 0.03], random_state=seed
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    # 同一個基準模型（未調整門檻之預設閥值 0.5 羅吉斯迴歸）
    clf = LogisticRegression(C=0.01, random_state=seed)
    clf.fit(X_tr, y_tr)
    pred_te = clf.predict(X_te)

    # 1. 軌 A（表面指標）：Accuracy，因負樣本佔 97%，全猜負樣本即可獲得 ~97% 準確率
    acc_val = float(accuracy_score(y_te, pred_te))

    # 2. 軌 B（實務檢驗）：Recall，預設閥值下少數破產樣本全數漏抓，召回率為 0.0%
    rec_val = float(recall_score(y_te, pred_te, zero_division=0))

    return acc_val, rec_val


def run_experiment():
    acc_val, rec_val = eval_single(seed=42)
    print("=== Bug 03: Accuracy Trap ===")
    print(f"Superficial Accuracy (Buggy Metric): {acc_val:.4f} (looks very high e.g. ~97%)")
    print(f"True Operational Recall (Clean Metric): {rec_val:.4f} (fails to catch bankruptcies)")
    print(f"Gap (Accuracy - Recall): {acc_val - rec_val:+.4f}")


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
