# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from scipy import stats

def run_experiment():
    X, y = make_classification(
        n_samples=600,
        n_features=12,
        weights=[0.95, 0.05],
        random_state=42
    )

    skf_a = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
    scores_a = []
    for train_idx, val_idx in skf_a.split(X, y):
        clf_a = RandomForestClassifier(n_estimators=40, random_state=42)
        clf_a.fit(X[train_idx], y[train_idx])
        scores_a.append(average_precision_score(y[val_idx], clf_a.predict_proba(X[val_idx])[:, 1]))

    skf_b = StratifiedKFold(n_splits=5, shuffle=True, random_state=202)
    scores_b = []
    for train_idx, val_idx in skf_b.split(X, y):
        clf_b = GradientBoostingClassifier(n_estimators=40, random_state=42)
        clf_b.fit(X[train_idx], y[train_idx])
        scores_b.append(average_precision_score(y[val_idx], clf_b.predict_proba(X[val_idx])[:, 1]))

    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)

    print(f'Model A 均值: {np.mean(scores_a):.4f}')
    print(f'Model B 均值: {np.mean(scores_b):.4f}')
    print(f'成對 t 檢定 p-value: {p_val:.4f} (t={t_stat:.2f})')

if __name__ == '__main__':
    run_experiment()
