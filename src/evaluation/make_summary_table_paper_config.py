# -*- coding: utf-8 -*-
import pandas as pd


def make_summary_table_paper_config(tidy_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot tidy metrics DataFrame into a wide summary table.

    Parameters
    ----------
    tidy_metrics_df : pd.DataFrame
        Long-format metrics table as produced by flatten_results.

    Returns
    -------
    pd.DataFrame
        Wide-format table containing train/test corpus,
        feature set and metrics formatted as 'mean ± std'.
    """

    metrics_to_show = ["accuracy", "auc"]

    feature_sets = [
        "acoustic",
        "spectral_paper",
        "combined_paper"
    ]

    df = tidy_metrics_df[
        (tidy_metrics_df["metric"].isin(metrics_to_show)) &
        (tidy_metrics_df["features"].isin(feature_sets))
    ].copy()


    df["formatted"] = df.apply(
        lambda r: (
            f"{r['value']*100:.1f} ± {r['std']*100:.1f}"
            if r["metric"] == "accuracy"
            else f"{r['value']:.3f} ± {r['std']:.3f}"
        ),
        axis=1,
    )


    table = df.pivot(
        index=[
            "train_corpus",
            "test_corpus",
            "features"
        ],
        columns="metric",
        values="formatted",
    )


    # Preserve metric order
    table = table[metrics_to_show]


    table.columns = [
        "Accuracy",
        "AUC"
    ]


    table = table.reset_index()
    
    # Preserve corpus ordering
    corpus_order = tidy_metrics_df["train_corpus"].unique()

    table["train_corpus"] = pd.Categorical(
        table["train_corpus"],
        categories=corpus_order,
        ordered=True
    )

    table["test_corpus"] = pd.Categorical(
        table["test_corpus"],
        categories=corpus_order,
        ordered=True
    )


    # Preserve feature ordering
    table["features"] = pd.Categorical(
        table["features"],
        categories=feature_sets,
        ordered=True
    )


    table = table.sort_values(
        [
            "train_corpus",
            "test_corpus",
            "features"
        ]
    )

    return table