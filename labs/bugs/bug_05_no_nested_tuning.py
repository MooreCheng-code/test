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
        n_features=15,
        weights=[0.95, 0.05],
        random_state=42
    )

    max_depth_candidates = [2, 3, 5, 8, 12, None]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_param = None
    best_score = -1.0

    for depth in max_depth_candidates:
        fold_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = RandomForestClassifier(max_depth=depth, n_estimators=40, random_state=42)
            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_val)[:, 1]
            fold_scores.append(average_precision_score(y_val, y_prob))

        mean_score = np.mean(fold_scores)
        print(f'max_depth={depth}: CV PR-AUC = {mean_score:.4f}')
        if mean_score > best_score:
            best_score = mean_score
            best_param = depth

    print(f'最佳參數: max_depth={best_param}')
    print(f'對外報告最終泛化 PR-AUC: {best_score:.4f}')

if __name__ == '__main__':
    run_experiment()
