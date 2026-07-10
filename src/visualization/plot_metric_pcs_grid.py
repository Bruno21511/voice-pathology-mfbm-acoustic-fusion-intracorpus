# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_metric_pcs_grid(
    tidy_metrics_df: pd.DataFrame,
    metric_name: str = "accuracy",
    figsize: tuple[float, float] = (14, 6),
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> None:
    """Creates a 1x2 grid plotting a specific metric across Spectral PCs for both datasets.

    Extracting PC numbers dynamically from the tidy features column and plots separate
    curves for Spectral, Combined, Sep_raw, Sep_a1, and Sep_a2 configurations.
    """
    df = tidy_metrics_df.copy()

    # 1. Filter for the requested metric
    df = df[df["metric"] == metric_name]

    if df.empty:
        print(f"Warning: No data found for metric '{metric_name}'")
        return

    # 2. Extract the PC number from the feature string dynamically
    def extract_pc(feature_name):
        if not isinstance(feature_name, str):
            return -1
        for part in feature_name.split("_"):
            if part.isdigit():
                return int(part)
        return -1

    df["spec_pca"] = df["features"].apply(extract_pc)
    # Focus only on the valid experimental range (2 to 15 PCs)
    df = df[df["spec_pca"].between(2, 15)]

    # 3. Handle scale and axis formatting if metric is accuracy
    is_accuracy = metric_name.lower() == "accuracy"
    if is_accuracy:
        df["value"] = df["value"] * 100
        ylabel = "Accuracy (%)"
    else:
        ylabel = metric_name.upper()

    # Define the unique datasets and setup the 1x2 grid subplot
    datasets = ["mysMEEI", "myUSP"]
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # Global X axis styling parameters
    global_xmin = df["spec_pca"].min() - 0.5
    global_xmax = df["spec_pca"].max() + 0.5

    # Guardar handles para a legenda partilhada no final
    handles, labels = [], []

    # 4. Loop through datasets to build the comparison panels
    for idx, dataset_name in enumerate(datasets):
        ax = axes[idx]
        ds_df = df[df["test_corpus"] == dataset_name]

        # Configuração dinâmica do título baseando-se no train_corpus encontrado no dataframe
        if not ds_df.empty and "train_corpus" in ds_df.columns:
            train_corpus = ds_df["train_corpus"].iloc[0]
            if train_corpus == dataset_name:
                title_text = f"{ylabel} Analysis — {dataset_name}"
            else:
                title_text = f"{ylabel} Analysis — Trained in {train_corpus}, Tested in {dataset_name}"
        else:
            title_text = f"{ylabel} Analysis — {dataset_name}"

        # Define mapping definitions to filter tidy rows into line plots
        curves_definition = [
            {
                "label": "Spectral",
                "condition": lambda name: name.startswith("spectral_"),
                "style": {
                    "marker": "o",
                    "linewidth": 3,
                    "linestyle": "--",
                },
            },
            {
                "label": "Combined",
                "condition": lambda name: name.startswith("combined_"),
                "style": {"marker": "s", "linewidth": 2},
            },
            {
                "label": "Separate (raw acoustic)",
                "condition": lambda name: "sep_" in name
                and "_raw" in name,
                "style": {"marker": "^", "linewidth": 2},
            },
            {
                "label": "Separate (acoustic PCA=1)",
                "condition": lambda name: "sep_" in name
                and ("_a01" in name or "_a1" in name),
                "style": {"marker": "v", "linewidth": 2},
            },
            {
                "label": "Separate (acoustic PCA=2)",
                "condition": lambda name: "sep_" in name
                and ("_a02" in name or "_a2" in name),
                "style": {"marker": "d", "linewidth": 2},
            },
        ]

        # Plot each line map dynamically
        for curve in curves_definition:
            mask = ds_df["features"].apply(curve["condition"])
            curve_df = ds_df[mask].sort_values("spec_pca")

            if not curve_df.empty:
                line = ax.plot(
                    curve_df["spec_pca"],
                    curve_df["value"],
                    label=curve["label"],
                    **curve["style"],
                )
                # Guarda referências para a legenda apenas no primeiro subplot
                if idx == 0:
                    handles.append(line[0])
                    labels.append(curve["label"])

        # 5. Apply individual subplot cosmetics
        ax.set_xlim(global_xmin, global_xmax)
        ax.set_xticks(
            range(2, 16)
        )  # Explicit ticks for all PCs from 2 to 15
        ax.set_xlabel("Number of Spectral PCA Components")
        ax.set_ylabel(ylabel)
        ax.set_title(title_text, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

    # 6. Create unique global legend below the plots
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=10,
    )

    plt.tight_layout()

    # 7. Save figure handler
    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )
        logger.info(
            "%s results plot successfully saved to: %s",
            metric_name,
            save_path,
        )

    plt.show()