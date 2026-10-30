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
from imblearn.over_sampling import SMOTE

def run_experiment():
    X, y = make_classification(
        n_samples=600,
        n_features=15,
        n_informative=6,
        weights=[0.95, 0.05],
        random_state=42
    )

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_resampled, y_resampled)):
        X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
        y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, y_prob)
        scores.append(ap)
        print(f'Fold {fold}: PR-AUC = {ap:.4f}')

    print(f'Mean PR-AUC: {np.mean(scores):.4f}')

if __name__ == '__main__':
    run_experiment()
