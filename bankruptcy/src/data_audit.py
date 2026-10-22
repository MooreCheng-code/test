"""
Data Audit Module for Bankruptcy Prediction Project.

Performs comprehensive descriptive data audit:
1. Basic summary (shape, target distribution, missing values, duplicates)
2. Column-level anomaly checks (constant columns, identical column pairs,
   large values > 1, low-cardinality/flags, high-correlation pairs)
3. Univariate signal analysis (ROC-AUC and Mann-Whitney U test)
4. Near-duplicate sample inspection (QuantileTransformer + NearestNeighbors)
5. Visualization figures (target distribution, top-30 correlation heatmap, top-20 AUC bar chart)

Can be executed via:
    python -m src.data_audit
or:
    python src/data_audit.py
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow standalone script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

from config import (
    DATA_PATH,
    FIGURES_DIR,
    SEED,
    TABLES_DIR,
    TARGET_COL,
)


def load_and_clean_data(data_path: Path) -> pd.DataFrame:
    """Load dataset and strip leading/trailing whitespace from column names."""
    df = pd.read_csv(data_path)
    # Strip whitespace from column names (raw CSV has leading spaces)
    df.columns = df.columns.str.strip()
    return df


def audit_basic_info(df: pd.DataFrame, target_col: str) -> dict:
    """Compute and save basic dataset information."""
    n_rows, n_cols = df.shape
    features = [c for c in df.columns if c != target_col]
    target_counts = df[target_col].value_counts().to_dict()
    total_missing = int(df.isnull().sum().sum())
    dup_rows = int(df.duplicated().sum())

    n_bankrupt = target_counts.get(1, 0)
    n_non_bankrupt = target_counts.get(0, 0)
    pct_bankrupt = (n_bankrupt / n_rows) * 100.0
    pct_non_bankrupt = (n_non_bankrupt / n_rows) * 100.0

    summary_records = [
        {"metric": "total_rows", "value": n_rows},
        {"metric": "total_columns", "value": n_cols},
        {"metric": "feature_count", "value": len(features)},
        {"metric": "target_column", "value": target_col},
        {"metric": "non_bankrupt_count_0", "value": n_non_bankrupt},
        {"metric": "non_bankrupt_percent_0", "value": f"{pct_non_bankrupt:.2f}%"},
        {"metric": "bankrupt_count_1", "value": n_bankrupt},
        {"metric": "bankrupt_percent_1", "value": f"{pct_bankrupt:.2f}%"},
        {"metric": "imbalance_ratio_0_to_1", "value": f"{n_non_bankrupt / max(n_bankrupt, 1):.2f}:1"},
        {"metric": "missing_values_total", "value": total_missing},
        {"metric": "duplicate_rows_total", "value": dup_rows},
    ]

    summary_df = pd.DataFrame(summary_records)
    summary_path = TABLES_DIR / "audit_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_features": len(features),
        "target_counts": target_counts,
        "n_bankrupt": n_bankrupt,
        "n_non_bankrupt": n_non_bankrupt,
        "pct_bankrupt": pct_bankrupt,
        "pct_non_bankrupt": pct_non_bankrupt,
        "total_missing": total_missing,
        "dup_rows": dup_rows,
        "features": features,
    }


def audit_column_anomalies(df: pd.DataFrame, features: list[str]) -> dict:
    """Inspect columns for constants, duplicates, large values, low cardinality, and high correlations."""
    # 1. Constant columns
    constant_cols = [c for c in features if df[c].nunique() == 1]
    const_records = []
    for c in constant_cols:
        const_records.append({"column": c, "constant_value": df[c].iloc[0]})
    const_df = pd.DataFrame(const_records)
    const_df.to_csv(TABLES_DIR / "constant_columns.csv", index=False)

    # 2. Identical duplicate column pairs
    dup_pairs = []
    n_feat = len(features)
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            c1, c2 = features[i], features[j]
            if df[c1].equals(df[c2]):
                dup_pairs.append({"column_1": c1, "column_2": c2, "identical": True})
    dup_df = pd.DataFrame(dup_pairs)
    dup_df.to_csv(TABLES_DIR / "duplicate_columns.csv", index=False)

    # 3. Columns with max > 1.0 (extreme or truncated values)
    large_val_records = []
    for c in features:
        max_val = df[c].max()
        if max_val > 1.0:
            large_val_records.append({
                "column": c,
                "min": df[c].min(),
                "median": df[c].median(),
                "p95": df[c].quantile(0.95),
                "p99": df[c].quantile(0.99),
                "max": max_val,
            })
    large_val_df = pd.DataFrame(large_val_records).sort_values("max", ascending=False)
    large_val_df.to_csv(TABLES_DIR / "large_value_columns.csv", index=False)

    # 4. Low-cardinality / binary columns (nunique <= 2)
    low_card_records = []
    for c in features:
        nunique = df[c].nunique()
        if nunique <= 2:
            unique_vals = sorted(df[c].unique().tolist())
            prop_ones = float((df[c] == 1).mean()) if 1 in unique_vals else 0.0
            low_card_records.append({
                "column": c,
                "n_unique": nunique,
                "unique_values": str(unique_vals),
                "proportion_of_1s": prop_ones,
            })
    low_card_df = pd.DataFrame(low_card_records)
    low_card_df.to_csv(TABLES_DIR / "low_cardinality_columns.csv", index=False)

    # 5. Highly correlated column pairs (|corr| > 0.95 and > 0.99)
    corr_matrix = df[features].corr().abs()
    high_corr_records = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            c1, c2 = features[i], features[j]
            corr_val = corr_matrix.loc[c1, c2]
            if corr_val > 0.95:
                tier = ">0.99" if corr_val > 0.99 else ">0.95"
                high_corr_records.append({
                    "column_1": c1,
                    "column_2": c2,
                    "abs_pearson_corr": corr_val,
                    "tier": tier,
                })
    high_corr_df = pd.DataFrame(high_corr_records).sort_values("abs_pearson_corr", ascending=False)
    high_corr_df.to_csv(TABLES_DIR / "high_correlation_pairs.csv", index=False)

    return {
        "constant_cols": constant_cols,
        "dup_pairs": dup_pairs,
        "large_val_df": large_val_df,
        "low_card_df": low_card_df,
        "high_corr_df": high_corr_df,
    }


def audit_univariate_signals(df: pd.DataFrame, features: list[str], target_col: str) -> pd.DataFrame:
    """Compute univariate ROC-AUC and Mann-Whitney U test p-values for all features against target."""
    y = df[target_col].values
    univariate_records = []

    for c in features:
        x = df[c].values
        # If feature is constant across all rows
        if np.all(x == x[0]):
            univariate_records.append({
                "feature": c,
                "raw_auc": 0.5,
                "adjusted_auc": 0.5,
                "direction": "constant",
                "mwu_p_value": 1.0,
            })
            continue

        raw_auc = roc_auc_score(y, x)
        if raw_auc < 0.5:
            adj_auc = 1.0 - raw_auc
            direction = "negative"  # lower feature value indicates higher bankruptcy risk
        else:
            adj_auc = raw_auc
            direction = "positive"  # higher feature value indicates higher bankruptcy risk

        # Mann-Whitney U test between bankrupt (y=1) and non-bankrupt (y=0) groups
        x_pos = x[y == 1]
        x_neg = x[y == 0]
        _, p_val = mannwhitneyu(x_pos, x_neg, alternative="two-sided")

        univariate_records.append({
            "feature": c,
            "raw_auc": raw_auc,
            "adjusted_auc": adj_auc,
            "direction": direction,
            "mwu_p_value": p_val,
        })

    auc_df = pd.DataFrame(univariate_records).sort_values("adjusted_auc", ascending=False)
    auc_df.to_csv(TABLES_DIR / "univariate_auc.csv", index=False)
    return auc_df


def audit_near_duplicates(df: pd.DataFrame, features: list[str], target_col: str) -> dict:
    """
    Check for near-duplicate samples (data leakage risk) using QuantileTransformer + NearestNeighbors.
    Evaluates distance percentiles and label consistency among the nearest 1% sample pairs.
    """
    y = df[target_col].values

    # Quantile normalization to normal distribution
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    X_trans = qt.fit_transform(df[features])

    # 2 nearest neighbors (1st neighbor is self with dist 0, 2nd is true nearest neighbor)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(X_trans)
    distances, indices = nn.kneighbors(X_trans)

    nn_dist = distances[:, 1]
    nn_idx = indices[:, 1]

    # Distance percentiles for all samples
    p1 = float(np.percentile(nn_dist, 1))
    p5 = float(np.percentile(nn_dist, 5))
    p25 = float(np.percentile(nn_dist, 25))
    p50 = float(np.percentile(nn_dist, 50))
    p75 = float(np.percentile(nn_dist, 75))

    # Identify samples whose nearest neighbor distance is in bottom 1%
    mask_1pct = nn_dist <= p1
    indices_1pct = np.where(mask_1pct)[0]

    # Extract unique pairs
    pairs = []
    seen_pairs = set()
    for i in indices_1pct:
        j = nn_idx[i]
        pair_key = tuple(sorted([int(i), int(j)]))
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            l1, l2 = int(y[pair_key[0]]), int(y[pair_key[1]])
            pairs.append({
                "sample_idx_1": pair_key[0],
                "sample_idx_2": pair_key[1],
                "nn_distance": float(nn_dist[i]),
                "label_1": l1,
                "label_2": l2,
                "same_label": bool(l1 == l2),
                "both_bankrupt": bool(l1 == 1 and l2 == 1),
            })

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(TABLES_DIR / "near_duplicates.csv", index=False)

    n_unique_pairs = len(pairs_df)
    n_same_label = int(pairs_df["same_label"].sum()) if n_unique_pairs > 0 else 0
    prop_same_label = float(pairs_df["same_label"].mean()) if n_unique_pairs > 0 else 0.0
    n_both_bankrupt = int(pairs_df["both_bankrupt"].sum()) if n_unique_pairs > 0 else 0

    # --- Supplementary Bankrupt-Specific Near-Duplicate Inspection ---
    bankrupt_indices = np.where(y == 1)[0]
    b_nn_dists = nn_dist[bankrupt_indices]
    b_nn_idxs = nn_idx[bankrupt_indices]
    b_nn_labels = y[b_nn_idxs]

    p_b_5 = float(np.percentile(b_nn_dists, 5))
    p_b_25 = float(np.percentile(b_nn_dists, 25))
    p_b_50 = float(np.percentile(b_nn_dists, 50))
    p_b_75 = float(np.percentile(b_nn_dists, 75))

    b_nn_is_bankrupt_count = int(np.sum(b_nn_labels == 1))
    b_nn_is_bankrupt_ratio = float(np.mean(b_nn_labels == 1))
    expected_random_ratio = float((len(bankrupt_indices) - 1) / (len(df) - 1))

    b_in_bottom_5pct_count = int(np.sum(b_nn_dists <= p5))

    # Save bankrupt near-duplicates table
    b_records = []
    for idx, b_idx in enumerate(bankrupt_indices):
        b_records.append({
            "bankrupt_sample_idx": int(b_idx),
            "nearest_neighbor_idx": int(b_nn_idxs[idx]),
            "nn_distance": float(b_nn_dists[idx]),
            "nn_label": int(b_nn_labels[idx]),
            "nn_is_bankrupt": bool(b_nn_labels[idx] == 1),
            "in_overall_bottom_5pct": bool(b_nn_dists[idx] <= p5),
        })
    b_df = pd.DataFrame(b_records)
    b_df.to_csv(TABLES_DIR / "near_duplicates_bankrupt.csv", index=False)

    return {
        "p1": p1,
        "p5": p5,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "n_samples_in_bottom_1pct": len(indices_1pct),
        "n_unique_pairs": n_unique_pairs,
        "n_same_label": n_same_label,
        "prop_same_label": prop_same_label,
        "n_both_bankrupt": n_both_bankrupt,
        "bankrupt_nn": {
            "n_bankrupt": len(bankrupt_indices),
            "nn_is_bankrupt_count": b_nn_is_bankrupt_count,
            "nn_is_bankrupt_ratio": b_nn_is_bankrupt_ratio,
            "expected_random_ratio": expected_random_ratio,
            "p5": p_b_5,
            "p25": p_b_25,
            "p50": p_b_50,
            "p75": p_b_75,
            "in_overall_bottom_5pct_count": b_in_bottom_5pct_count,
        },
    }


def generate_figures(df: pd.DataFrame, auc_df: pd.DataFrame, target_col: str):
    """Generate and save audit figures: target distribution, top 30 correlation heatmap, top 20 AUC."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # 1. Target Distribution Bar Chart
    plt.figure(figsize=(7, 5))
    target_counts = df[target_col].value_counts().sort_index()
    total = len(df)
    ax = sns.barplot(
        x=target_counts.index.astype(str),
        y=target_counts.values,
        hue=target_counts.index.astype(str),
        palette=["#3498db", "#e74c3c"],
        legend=False,
    )
    plt.title("Target Distribution (Bankrupt? 0 vs 1)", fontsize=14, pad=12)
    plt.xlabel("Bankrupt? Flag", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    plt.xticks([0, 1], ["0: Non-Bankrupt", "1: Bankrupt"])

    for i, count in enumerate(target_counts.values):
        pct = (count / total) * 100.0
        ax.text(i, count + 80, f"{count:,} ({pct:.2f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.ylim(0, max(target_counts.values) * 1.12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=300)
    plt.close()

    # 2. Correlation Heatmap for Top 30 Features by Univariate AUC
    top30_features = auc_df["feature"].head(30).tolist()
    corr_top30 = df[top30_features].corr()

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_top30,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        annot=False,
    )
    plt.title("Pearson Correlation Heatmap (Top 30 Features by Univariate AUC)", fontsize=15, pad=15)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap_top30.png", dpi=300)
    plt.close()

    # 3. Univariate AUC Top 20 Horizontal Bar Chart
    top20_df = auc_df.head(20).copy()
    top20_df = top20_df.sort_values("adjusted_auc", ascending=True)

    plt.figure(figsize=(12, 9))
    colors = ["#e74c3c" if d == "positive" else "#3498db" for d in top20_df["direction"]]
    bars = plt.barh(top20_df["feature"], top20_df["adjusted_auc"], color=colors, alpha=0.85)

    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance Level (AUC=0.5)")
    plt.title("Top 20 Features by Univariate Adjusted ROC-AUC", fontsize=14, pad=12)
    plt.xlabel("Adjusted ROC-AUC (Score >= 0.5)", fontsize=12)
    plt.xlim(0.45, 1.0)

    # Custom legend for direction
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#e74c3c", lw=6, label="Positive (Higher value -> Higher risk)"),
        Line2D([0], [0], color="#3498db", lw=6, label="Negative (Lower value -> Higher risk)"),
    ]
    plt.legend(handles=legend_elements, loc="lower right", fontsize=10)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "univariate_auc_top20.png", dpi=300)
    plt.close()


def print_audit_report(basic: dict, anomalies: dict, auc_df: pd.DataFrame, near_dup: dict):
    """Print comprehensive summary report to stdout."""
    print("=" * 80)
    print("                      DATA AUDIT REPORT: SUMMARY")
    print("=" * 80)
    print("\n[1] BASIC DATASET CHARACTERISTICS")
    print(f"  * Shape:                 {basic['n_rows']} rows, {basic['n_cols']} columns")
    print(f"  * Feature Count:         {basic['n_features']}")
    print(f"  * Target Distribution:   0 (Non-bankrupt): {basic['n_non_bankrupt']} ({basic['pct_non_bankrupt']:.2f}%)")
    print(f"                           1 (Bankrupt):     {basic['n_bankrupt']} ({basic['pct_bankrupt']:.2f}%)")
    print(f"  * Total Missing Values:  {basic['total_missing']}")
    print(f"  * Duplicate Rows:        {basic['dup_rows']}")

    print("\n[2] COLUMN-LEVEL ANOMALIES")
    const_cols = anomalies["constant_cols"]
    print(f"  * Constant Columns ({len(const_cols)}):")
    for c in const_cols:
        print(f"      - '{c}' (All values identical)")

    dup_pairs = anomalies["dup_pairs"]
    print(f"  * Identical Column Pairs ({len(dup_pairs)}):")
    for p in dup_pairs:
        print(f"      - '{p['column_1']}' == '{p['column_2']}'")

    large_val_df = anomalies["large_val_df"]
    print(f"  * Columns with Max Value > 1.0 ({len(large_val_df)} columns):")
    print(f"      - Columns reaching 10^9 ~ 10^10: {int((large_val_df['max'] >= 1e9).sum())}")
    print(f"      - Example large value columns:")
    for _, row in large_val_df.head(5).iterrows():
        print(f"          * {row['column'][:35]:<35} | Median: {row['median']:.4e} | P99: {row['p99']:.4e} | Max: {row['max']:.4e}")

    low_card_df = anomalies["low_card_df"]
    print(f"  * Low-Cardinality / Binary Columns ({len(low_card_df)}):")
    for _, row in low_card_df.iterrows():
        print(f"      - '{row['column']}': nunique={row['n_unique']}, values={row['unique_values']}, % of 1s={row['proportion_of_1s']*100:.3f}%")

    high_corr_df = anomalies["high_corr_df"]
    n_corr_095 = len(high_corr_df)
    n_corr_099 = len(high_corr_df[high_corr_df["tier"] == ">0.99"])
    print(f"  * High Pearson Correlation Pairs (|r| > 0.95: {n_corr_095} pairs, of which |r| > 0.99: {n_corr_099} pairs)")
    for _, row in high_corr_df.head(6).iterrows():
        print(f"      - {row['column_1'][:30]:<30} <-> {row['column_2'][:30]:<30} : r = {row['abs_pearson_corr']:.5f}")

    print("\n[3] UNIVARIATE PREDICTIVE SIGNAL (TOP 10 FEATURES BY ADJUSTED AUC)")
    for i, (_, row) in enumerate(auc_df.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['feature'][:45]:<45} | Adj AUC: {row['adjusted_auc']:.4f} | Dir: {row['direction']:<8} | MWU p: {row['mwu_p_value']:.2e}")

    print("\n[4] NEAR-DUPLICATE SAMPLES & LEAKAGE RISK")
    print(f"  * Nearest Neighbor Distance Percentiles (QuantileTransformer standard normal space):")
    print(f"      - 1st Percentile:  {near_dup['p1']:.4f}")
    print(f"      - 5th Percentile:  {near_dup['p5']:.4f}")
    print(f"      - 25th Percentile: {near_dup['p25']:.4f}")
    print(f"      - 50th Percentile: {near_dup['p50']:.4f}")
    print(f"      - 75th Percentile: {near_dup['p75']:.4f}")
    print(f"  * Bottom 1% Closest Samples:")
    print(f"      - Samples with NN distance <= P1: {near_dup['n_samples_in_bottom_1pct']}")
    print(f"      - Unique Sample Pairs:            {near_dup['n_unique_pairs']}")
    print(f"      - Label Consistency Ratio:        {near_dup['prop_same_label']*100:.2f}% ({near_dup['n_same_label']}/{near_dup['n_unique_pairs']})")
    print(f"      - Pairs with Both Bankrupt (1-1): {near_dup['n_both_bankrupt']}")

    b_info = near_dup["bankrupt_nn"]
    print("\n[5] BANKRUPT-SPECIFIC NEAR-NEIGHBOR ANALYSIS (220 SAMPLES)")
    print(f"  * Total Bankrupt Samples:                     {b_info['n_bankrupt']}")
    print(f"  * Nearest Neighbor is also Bankrupt:          {b_info['nn_is_bankrupt_count']} / {b_info['n_bankrupt']} ({b_info['nn_is_bankrupt_ratio']*100:.2f}%)")
    print(f"  * Baseline Random Expectation:                {b_info['expected_random_ratio']*100:.2f}% (Observed is {b_info['nn_is_bankrupt_ratio'] / b_info['expected_random_ratio']:.1f}x higher)")
    print(f"  * Nearest Neighbor Distance Percentiles Comparison:")
    print(f"      -  5th Percentile: Bankrupt = {b_info['p5']:.4f}  vs  All Samples = {near_dup['p5']:.4f}")
    print(f"      - 25th Percentile: Bankrupt = {b_info['p25']:.4f}  vs  All Samples = {near_dup['p25']:.4f}")
    print(f"      - 50th Percentile: Bankrupt = {b_info['p50']:.4f}  vs  All Samples = {near_dup['p50']:.4f}")
    print(f"      - 75th Percentile: Bankrupt = {b_info['p75']:.4f}  vs  All Samples = {near_dup['p75']:.4f}")
    print(f"  * Bankrupt Samples in Overall Bottom 5% Dist: {b_info['in_overall_bottom_5pct_count']} / {b_info['n_bankrupt']}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Data Audit for Bankruptcy Prediction Dataset")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to input CSV data")
    parser.add_argument("--target", type=str, default=TARGET_COL, help="Target column name")
    args = parser.parse_args()

    data_path = Path(args.data)
    target_col = args.target

    print(f"Starting Data Audit on {data_path} (Target: '{target_col}') ...")
    df = load_and_clean_data(data_path)

    # 1. Basic Info
    basic = audit_basic_info(df, target_col)

    # 2. Column Anomalies
    anomalies = audit_column_anomalies(df, basic["features"])

    # 3. Univariate Signals
    auc_df = audit_univariate_signals(df, basic["features"], target_col)

    # 4. Near-Duplicate Samples
    near_dup = audit_near_duplicates(df, basic["features"], target_col)

    # 5. Visualizations
    generate_figures(df, auc_df, target_col)

    # Print summary report to console
    print_audit_report(basic, anomalies, auc_df, near_dup)
    print("\nAudit completed successfully. Outputs saved to:")
    print(f"  - Tables:  {TABLES_DIR}")
    print(f"  - Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
