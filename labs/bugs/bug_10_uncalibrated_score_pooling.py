# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

def run_experiment():
    X, y = make_classification(
        n_samples=600,
        n_features=10,
        weights=[0.95, 0.05],
        random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_val_probs = []
    all_val_targets = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]
        all_val_probs.extend(probs)
        all_val_targets.extend(y_val)

    pooled_ap = average_precision_score(all_val_targets, all_val_probs)
    print(f'池化全域 OOF PR-AUC: {pooled_ap:.4f}')

if __name__ == '__main__':
    run_experiment()
