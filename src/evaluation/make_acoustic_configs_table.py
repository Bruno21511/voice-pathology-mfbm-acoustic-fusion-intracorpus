# -*- coding: utf-8 -*-
import pandas as pd


def make_acoustic_configs_table(tidy_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Generates a compact summary table with custom transposed layout.

    The rows represent the training/testing datasets and metrics (Accuracy and AUC), 
    while the columns compare the feature setups ('acoustic' vs 'acoustic_balanced_weights').
    Accuracy is formatted as a percentage and AUC as a decimal, both with standard deviation.

    Parameters
    ----------
    tidy_metrics_df : pd.DataFrame
        The original tidy DataFrame containing all features and metrics.
        Required columns: train_corpus, test_corpus, metric, features, value, std

    Returns
    -------
    pd.DataFrame
        A formatted summary table with a multi-index (train_corpus, test_corpus, metric)
        and feature configurations as columns.
    """
    df = tidy_metrics_df.copy()

    # 1. Filter for target feature sets and metrics
    target_features = ["acoustic", "acoustic_balanced_weights"]
    df = df[df["features"].isin(target_features)]
    df = df[df["metric"].isin(["accuracy", "auc"])]

    # 2. Apply custom formatting depending on the metric type
    def format_row(row: pd.Series) -> str:
        if row["metric"] == "accuracy":
            # Format accuracy as percentage (e.g., 83.6% ± 1.0%)
            return f"{row['value'] * 100:.1f}% ± {row['std'] * 100:.1f}%"
        else:
            # Format AUC as raw standard decimal (e.g., 0.879 ± 0.006)
            return f"{row['value']:.3f} ± {row['std']:.3f}"

    df["formatted_value"] = df.apply(format_row, axis=1)

    # 3. Pivot the table: features go to columns, train/test corpus and metric go to index
    summary_df = df.pivot(
        index=["train_corpus", "test_corpus", "metric"],
        columns="features",
        values="formatted_value",
    )

    # 4. Reset index temporarily to enforce categorical sorting rules
    summary_df = summary_df.reset_index()

    # Determinar a ordem dos corpora de treino dinamicamente
    unique_train = list(summary_df["train_corpus"].unique())
    if "mysMEEI" in unique_train:
        unique_train.remove("mysMEEI")
        unique_train.insert(0, "mysMEEI")

    summary_df["train_corpus"] = pd.Categorical(
        summary_df["train_corpus"], categories=unique_train, ordered=True
    )
    summary_df["test_corpus"] = pd.Categorical(
        summary_df["test_corpus"], categories=["mysMEEI", "myUSP"], ordered=True
    )
    # Enforce accuracy to always appear before auc for each corpus
    summary_df["metric"] = pd.Categorical(
        summary_df["metric"], categories=["accuracy", "auc"], ordered=True
    )

    summary_df = summary_df.sort_values(
        by=["train_corpus", "test_corpus", "metric"]
    ).reset_index(drop=True)

    # 5. Set 'train_corpus', 'test_corpus' and 'metric' as MultiIndex to remove row numbers
    summary_df.set_index(["train_corpus", "test_corpus", "metric"], inplace=True)

    # Clean up column axis name label
    summary_df.columns.name = None

    # Reorder columns to guarantee 'acoustic' comes before 'acoustic_balanced_weights'
    expected_columns = ["acoustic", "acoustic_balanced_weights"]
    existing_columns = [col for col in expected_columns if col in summary_df.columns]
    summary_df = summary_df[existing_columns]

    return summary_df