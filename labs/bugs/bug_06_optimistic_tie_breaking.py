# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def run_experiment():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        weights=[0.95, 0.05],
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    pos_probs = test_probs[y_test == 1]

    target_recall = 0.90
    cutoff = float(np.percentile(pos_probs, (1.0 - target_recall) * 100))

    is_alert = test_probs > cutoff

    alerts = np.sum(is_alert)
    caught = np.sum(is_alert & (y_test == 1))
    total_pos = np.sum(y_test == 1)

    print(f'選定門檻 cutoff: {cutoff:.4f}')
    print(f'預警數量: {alerts}')
    print(f'捕獲破產數: {caught}/{total_pos} (召回率: {caught/total_pos:.4f})')
    print(f'預警倍數 (AR): {alerts/total_pos:.2f}x')

if __name__ == '__main__':
    run_experiment()
