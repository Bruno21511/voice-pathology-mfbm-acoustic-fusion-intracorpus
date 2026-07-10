# -*- coding: utf-8 -*-
import pandas as pd


def make_mean_by_mode_table(
    tidy_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generates a summary table comparing the average of feature families.

    Groups features into families, handles dynamic separation names (e.g., sep_05_raw,
    sep_10_a01, sep_xx_a02), calculates the simple arithmetic mean of their
    values for accuracy and auc, and formats them into a transposed layout
    including training and testing corpora.

    Parameters
    ----------
    tidy_metrics_df : pd.DataFrame
        The original tidy DataFrame containing all metrics.
        Required columns: test_corpus, train_corpus, metric, features, value

    Returns
    -------
    pd.DataFrame
        A summary table with dataset/metric as rows and feature families as columns.
    """
    df = tidy_metrics_df.copy()

    # 1. Filter for target metrics only
    df = df[df["metric"].isin(["accuracy", "auc"])]

    # 2. Define a flexible helper function to catch substrings anywhere in the name
    def categorize_family(feature_name: str) -> str:
        # Avoid breaking with edge cases
        if not isinstance(feature_name, str):
            return "exclude"

        if feature_name in ["acoustic", "acoustic_balanced_weights"]:
            return "acoustic"
        elif feature_name.startswith("spectral_"):
            if feature_name == "spectral_paper":
                return "exclude"
            return "spectral"
        elif feature_name.startswith("combined_"):
            if feature_name == "combined_paper":
                return "exclude"
            return "combined"
        elif "sep_" in feature_name:
            if "_raw" in feature_name:
                return "sep_raw"
            elif "_a01" in feature_name or "_a1" in feature_name:
                return "sep_a1"
            elif "_a02" in feature_name or "_a2" in feature_name:
                return "sep_a2"
        return "exclude"

    # Map each row to its respective family and drop excluded features
    df["family"] = df["features"].apply(categorize_family)
    df = df[df["family"] != "exclude"]

    # 3. Calculate the simple arithmetic mean for each group (Incluindo train_corpus)
    grouped = (
        df.groupby(["train_corpus", "test_corpus", "metric", "family"])["value"]
        .mean()
        .reset_index()
    )

    # 4. Apply formatting (Percentage for accuracy, standard decimal for AUC)
    def format_cell(row: pd.Series) -> str:
        if row["metric"] == "accuracy":
            return f"{row['value'] * 100:.1f}%"
        else:
            return f"{row['value']:.3f}"

    grouped["formatted_value"] = grouped.apply(format_cell, axis=1)

    # 5. Pivot the table to move families into columns (Incluindo train_corpus no índice)
    summary_df = grouped.pivot(
        index=["train_corpus", "test_corpus", "metric"],
        columns="family",
        values="formatted_value",
    )

    # 6. Enforce categorical sorting rules for rows
    summary_df = summary_df.reset_index()
    
    # Adiciona categorias para ordenação do train_corpus se necessário
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
    summary_df["metric"] = pd.Categorical(
        summary_df["metric"], categories=["accuracy", "auc"], ordered=True
    )
    
    summary_df = summary_df.sort_values(
        by=["train_corpus", "test_corpus", "metric"]
    ).reset_index(drop=True)

    # 7. Set MultiIndex to clean up visual layout and row numbers
    summary_df.set_index(["train_corpus", "test_corpus", "metric"], inplace=True)
    summary_df.columns.name = None

    # 8. Reorder columns to match your exact requested sequence
    expected_columns = [
        "acoustic",
        "spectral",
        "combined",
        "sep_raw",
        "sep_a1",
        "sep_a2",
    ]
    existing_columns = [col for col in expected_columns if col in summary_df.columns]
    summary_df = summary_df[existing_columns]

    return summary_df