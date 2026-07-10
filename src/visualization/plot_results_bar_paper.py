# -*- coding: utf-8 -*-
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_results_bar_paper(
    tidy_metrics,
    metric="accuracy",
    save_path=None,
    dpi=300,
    y_min=None,
):
    """
    Plot grouped bar charts comparing paper configurations across corpora.

    Parameters
    ----------
    tidy_metrics : pd.DataFrame
        Tidy metrics dataframe produced by flatten_results.

        Required columns:
            - test_corpus
            - train_corpus  # <-- Necessário para a validação dinâmica
            - features
            - metric
            - value
            - std

    metric : {"accuracy", "auc"}
        Metric to plot.

    save_path : str or None
        If provided, saves the figure.

    dpi : int
        Figure resolution.

    y_min : float or None
        Lower y-axis limit.
        If None, automatically estimated.
    """

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    feature_order = [
        "acoustic",
        "spectral_paper",
        "combined_paper",
    ]

    feature_labels = {
        "acoustic": "Acoustic",
        "spectral_paper": "Spectral",
        "combined_paper": "Combined",
    }

    # ---------------------------------------------------------
    # Validate metric
    # ---------------------------------------------------------

    if metric not in ["accuracy", "auc"]:
        raise ValueError(
            "metric must be either 'accuracy' or 'auc'"
        )

    # ---------------------------------------------------------
    # Filter selected metric and configurations
    # ---------------------------------------------------------

    df = tidy_metrics[
        (tidy_metrics["metric"] == metric) &
        (tidy_metrics["features"].isin(feature_order))
    ].copy()

    if df.empty:
        raise ValueError(
            f"No data found for metric '{metric}'"
        )

    # ---------------------------------------------------------
    # Corpus ordering & Dynamic X-Label Generation
    # ---------------------------------------------------------

    corpora = list(df["test_corpus"].unique())

    # Force mysMEEI first if available
    if "mysMEEI" in corpora:
        corpora.remove("mysMEEI")
        corpora.insert(0, "mysMEEI")

    # Criar rótulos customizados para o eixo X com base no train_corpus
    xtick_labels = []
    for corpus in corpora:
        sub_df = df[df["test_corpus"] == corpus]
        
        if not sub_df.empty and "train_corpus" in sub_df.columns:
            train_corpus = sub_df["train_corpus"].iloc[0]
            if train_corpus == corpus:
                xtick_labels.append(corpus)
            else:
                xtick_labels.append(f"Trained in {train_corpus}\nTested in {corpus}")
        else:
            xtick_labels.append(corpus)

    # ---------------------------------------------------------
    # Build matrices
    # ---------------------------------------------------------

    values = []
    errors = []

    for feat in feature_order:
        value_row = []
        error_row = []

        for corpus in corpora:
            row = df[
                (df["features"] == feat) &
                (df["test_corpus"] == corpus)
            ]

            if row.empty:
                raise ValueError(
                    f"Missing result: {corpus} - {feat}"
                )

            value = row["value"].iloc[0]
            std = row["std"].iloc[0]

            if metric == "accuracy":
                value *= 100
                std *= 100

            value_row.append(value)
            error_row.append(std)

        values.append(value_row)
        errors.append(error_row)

    values = np.asarray(values)
    errors = np.asarray(errors)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    positions = np.arange(len(corpora))
    width = 0.22

    colors = [
        "darkorange",
        "lightcyan",
        "darkgreen",
    ]

    hatches = [
        "//",
        "\\\\",
        "oo",
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    for ii, feat in enumerate(feature_order):
        pos = positions + (ii - 1) * width

        ax.bar(
            pos,
            values[ii],
            width,
            yerr=errors[ii],
            capsize=5,
            color=colors[ii],
            edgecolor="black",
            hatch=hatches[ii],
            label=feature_labels[feat],
        )

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------

    ax.set_xticks(positions)
    # Aplicar os novos rótulos construídos dinamicamente
    ax.set_xticklabels(xtick_labels, fontsize=10, fontweight="bold")

    if metric == "accuracy":
        if y_min is None:
            y_min = np.floor(values.min() - 5)

        ax.set_ylim(y_min, 100)
        ax.set_ylabel("Accuracy [%]", fontsize=11)
        ax.set_title("Classification Accuracy", fontsize=13, fontweight="bold")

    elif metric == "auc":
        if y_min is None:
            y_min = np.floor(values.min() * 100) / 100 - 0.02

        ax.set_ylim(y_min, 1.0)
        ax.set_ylabel("AUC", fontsize=11)
        ax.set_title("Area Under the ROC Curve", fontsize=13, fontweight="bold")

    ax.grid(
        which="major",
        axis="y",
        linewidth=0.6,
        color="black",
        alpha=0.3,
    )

    ax.grid(
        which="minor",
        axis="y",
        linewidth=0.3,
        color="black",
        alpha=0.2,
    )

    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.legend(loc="best")

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )
        
        logger.info(
            "%s bar plot saved to %s",
            metric,
            save_path
        )

    plt.show()