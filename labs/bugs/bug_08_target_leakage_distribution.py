# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def run_experiment():
    X, y = make_classification(
        n_samples=800,
        n_features=10,
        weights=[0.95, 0.05],
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    raw_probs = model.predict_proba(X_test)[:, 1]

    test_prevalence = np.mean(y_test)
    adjusted_threshold = float(np.quantile(raw_probs, 1.0 - test_prevalence))

    alerts = np.sum(raw_probs >= adjusted_threshold)
    caught = np.sum((raw_probs >= adjusted_threshold) & (y_test == 1))
    total_pos = np.sum(y_test == 1)

    print(f'依測試集真實盛行率 ({test_prevalence:.4f}) 設定動態門檻: {adjusted_threshold:.4f}')
    print(f'預警數量: {alerts}')
    print(f'破產捕獲: {caught}/{total_pos} (召回率: {caught/total_pos:.4f})')

if __name__ == '__main__':
    run_experiment()
