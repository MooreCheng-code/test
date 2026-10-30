# ============================================================
# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用
# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md
# 只使用合成資料，不讀取專案資料。
# ============================================================

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def run_experiment():
    X, y = make_classification(
        n_samples=800,
        n_features=15,
        weights=[0.95, 0.05],
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]

    target_recall = 0.90
    pos_probs = val_probs[y_val == 1]
    best_cutoff = float(np.percentile(pos_probs, (1.0 - target_recall) * 100))

    alerts = np.sum(val_probs >= best_cutoff)
    caught = np.sum((val_probs >= best_cutoff) & (y_val == 1))
    total_pos = np.sum(y_val == 1)

    actual_recall = caught / total_pos
    alert_ratio = alerts / total_pos

    print(f'在驗證集選取之最優門檻: {best_cutoff:.4f}')
    print(f'驗證集實測召回率: {actual_recall:.4f}')
    print(f'驗證集實測預警倍數 (AR): {alert_ratio:.2f}x')

if __name__ == '__main__':
    run_experiment()
