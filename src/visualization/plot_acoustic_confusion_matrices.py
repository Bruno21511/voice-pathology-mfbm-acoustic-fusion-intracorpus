# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import logging

logger = logging.getLogger(__name__)


def plot_acoustic_confusion_matrices(
    tidy_cm_df: pd.DataFrame,
    test_corpus: str,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> None:
    """
    Plot side-by-side confusion matrices for the acoustic baseline and the
    class-balanced acoustic classifier.

    Cell colours represent the normalised confusion matrix, while the numbers
    correspond to the (average) confusion matrix counts.

    Parameters
    ----------
    tidy_cm_df : pd.DataFrame
        Tidy dataframe containing confusion matrix entries.
    test_corpus : str
        Test corpus to display (e.g. "mysMEEI").
    save_path : str or Path or None, optional
        If provided, saves the figure.
    dpi : int, optional
        Figure resolution when saving.
    """

    class_names = [
        "Healthy",
        "Pathological",
    ]

    # ---------------------------------------------------------
    # Select results for the requested corpus
    # ---------------------------------------------------------

    corpus_df = tidy_cm_df[
        tidy_cm_df["test_corpus"] == test_corpus
    ]

    if corpus_df.empty:
        raise ValueError(
            f"No results found for test corpus '{test_corpus}'."
        )

    # ---------------------------------------------------------
    # Reconstruct confusion matrices
    # ---------------------------------------------------------

    def get_matrices(
        feature_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:

        sub_df = corpus_df[
            corpus_df["features"] == feature_name
        ]

        if sub_df.empty:
            raise ValueError(
                f"No confusion matrix found for "
                f"'{feature_name}'."
            )

        cm = (
            sub_df.pivot(
                index="true_index",
                columns="pred_index",
                values="count",
            )
            .reindex(
                index=[0, 1],
                columns=[0, 1],
                fill_value=0,
            )
            .to_numpy()
        )

        cm_norm = (
            sub_df.pivot(
                index="true_index",
                columns="pred_index",
                values="norm_value",
            )
            .reindex(
                index=[0, 1],
                columns=[0, 1],
                fill_value=0,
            )
            .to_numpy()
        )

        return cm, cm_norm


    cm_acoustic, cmn_acoustic = get_matrices(
        "acoustic"
    )

    cm_balanced, cmn_balanced = get_matrices(
        "acoustic_balanced_weights"
    )

    # ---------------------------------------------------------
    # Figure layout & Dynamic Title Configuration
    # ---------------------------------------------------------

    # Descobre o train_corpus automaticamente a partir do dataframe filtrado
    train_corpus = corpus_df["train_corpus"].iloc[0]

    if train_corpus == test_corpus:
        title_text = f"Confusion Matrices - {test_corpus}"
    else:
        title_text = f"Confusion Matrices - trained in {train_corpus}, tested in {test_corpus}"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 3.5),
        sharey=True,
    )

    fig.suptitle(
        title_text,
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )

    # ---------------------------------------------------------
    # Plot helper
    # ---------------------------------------------------------

    def plot_matrix(
        ax,
        cm,
        cm_norm,
        title,
    ):

        sns.heatmap(
            cm_norm,
            ax=ax,
            annot=False,
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
        )

        ax.set_title(
            title,
            fontsize=12,
        )

        ax.set_xlabel(
            "Predicted Label"
        )

        # Overlay count values
        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                colour = (
                    "white"
                    if cm_norm[i, j] > 0.5
                    else "black"
                )

                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{cm[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=colour,
                )

    # ---------------------------------------------------------
    # Acoustic
    # ---------------------------------------------------------

    plot_matrix(
        axes[0],
        cm_acoustic,
        cmn_acoustic,
        "Acoustic",
    )

    axes[0].set_ylabel(
        "True Label"
    )

    # ---------------------------------------------------------
    # Acoustic + balanced weights
    # ---------------------------------------------------------

    plot_matrix(
        axes[1],
        cm_balanced,
        cmn_balanced,
        "Acoustic (Balanced Weights)",
    )

    axes[1].set_ylabel("")


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
            "Acoustic configs confusion matrices of %s plot saved to %s",
            test_corpus,
            save_path
        )

    plt.show()