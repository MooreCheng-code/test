# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

def run_experiment():
    X, y = make_classification(
        n_samples=500,
        n_features=12,
        weights=[0.95, 0.05],
        random_state=42
    )

    skf1 = StratifiedKFold(n_splits=5, shuffle=True)
    scores1 = []
    for train_idx, val_idx in skf1.split(X, y):
        model1 = RandomForestClassifier(n_estimators=30)
        model1.fit(X[train_idx], y[train_idx])
        scores1.append(average_precision_score(y[val_idx], model1.predict_proba(X[val_idx])[:, 1]))

    skf2 = StratifiedKFold(n_splits=5, shuffle=True)
    scores2 = []
    for train_idx, val_idx in skf2.split(X, y):
        model2 = RandomForestClassifier(n_estimators=30)
        model2.fit(X[train_idx], y[train_idx])
        scores2.append(average_precision_score(y[val_idx], model2.predict_proba(X[val_idx])[:, 1]))

    m1, m2 = np.mean(scores1), np.mean(scores2)
    print(f'Run 1 PR-AUC: {m1:.4f}')
    print(f'Run 2 PR-AUC: {m2:.4f}')
    print(f'差異: {m2 - m1:+.4f}，研判參數改動帶來顯著性能變化。')

if __name__ == '__main__':
    run_experiment()
