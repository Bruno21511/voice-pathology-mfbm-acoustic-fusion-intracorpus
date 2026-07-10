# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


def _plot_duration(
    df: pd.DataFrame,
    dataframe_name: str,
    show_plot: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Plot signal duration distribution by sample, grouped by class.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing:
            - signal
            - fs
            - group

    dataframe_name : str
        Name of the dataframe/dataset used in the plot title.

    show_plot : bool
        If True, displays the plot.

    save_path : str or None
        If provided, saves the figure to this path.

    Returns
    -------
    None
    """

    # Compute signal durations
    durations = [
        len(row["signal"]) / row["fs"]
        for _, row in df.iterrows()
    ]

    # Temporary dataframe for plotting
    temp_df = df.copy()
    temp_df["duration_tmp"] = durations


    # Create figure
    plt.figure(figsize=(12, 6))

    groups = temp_df["group"].unique()

    colors = plt.cm.viridis(
        np.linspace(0, 1, len(groups))
    )


    for i, group in enumerate(groups):

        mask = temp_df["group"] == group

        plt.scatter(
            temp_df.index[mask],
            temp_df.loc[mask, "duration_tmp"],
            label=group,
            alpha=0.7,
            s=30,
            color=colors[i]
        )


    # Plot styling
    plt.title(
        f"Signal Duration Distribution - {dataframe_name}",
        fontsize=14
    )

    plt.xlabel(
        "Sample Index",
        fontsize=12
    )

    plt.ylabel(
        "Duration (seconds)",
        fontsize=12
    )

    plt.legend(
        title="Class",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.6
    )

    plt.tight_layout()


    # Save figure if requested
    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        logger.info(
            "Duration plot saved to %s",
            save_path
        )


    # Display plot if requested
    if show_plot:
        plt.show()

    else:
        plt.close()


def _check_short_files(
    df: pd.DataFrame,
    threshold_seconds: float = 3
) -> list[str]:

    """
    Identify files shorter than a duration threshold.

    Returns
    -------
    list[str]
        File names below the threshold.
    """

    if "duration" not in df.columns:

        df = df.copy()

        df["duration"] = df.apply(
            lambda row: len(row["signal"]) / row["fs"],
            axis=1
        )


    short_files = df[
        df["duration"] < threshold_seconds
    ]


    if short_files.empty:

        logger.info(
            "No files shorter than %.2f seconds found.",
            threshold_seconds
        )

        return []


    logger.info(
        "Found %d files shorter than %.2f seconds.",
        len(short_files),
        threshold_seconds
    )


    return short_files["file"].tolist()


def remove_short_files(
    datasets: dict[str, pd.DataFrame],
    threshold_seconds: list[float],
    save_dir: Optional[str] = None,
    show_plot: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Remove files shorter than a defined duration threshold from multiple datasets.

    For each dataset:
        - checks short files
        - saves duration plot
        - removes short files

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Dictionary containing datasets.

    threshold_seconds : list[float]
        Duration threshold in seconds for each dataset.
        Must have the same number of elements as datasets.

    save_dir : str
        Directory where duration plots are saved.

    show_plot : bool
        If True, displays plots.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Processed datasets.
    """

    # ------------------------------------------------------------------
    # Dynamic Threshold Expansion
    # ------------------------------------------------------------------
    # If a single float/int was passed instead of a list, turn it into a list
    if isinstance(threshold_seconds, (int, float)):
        threshold_seconds = [float(threshold_seconds)]
    else:
        threshold_seconds = [float(t) for t in threshold_seconds]

    # If we have fewer thresholds than datasets, expand using the last available threshold
    if 0 < len(threshold_seconds) < len(datasets):
        last_threshold = threshold_seconds[-1]
        padding_needed = len(datasets) - len(threshold_seconds)
        threshold_seconds.extend([last_threshold] * padding_needed)
        logger.info(
            "Expanded thresholds to match dataset count: %s",
            threshold_seconds,
        )
    elif len(threshold_seconds) != len(datasets):
        # This handles the edge case where threshold_seconds might be empty
        logger.warning(
            "Number of thresholds (%d) does not match number of datasets (%d).",
            len(threshold_seconds),
            len(datasets),
        )


    processed_datasets = {}

    Path(save_dir).mkdir(
        parents=True,
        exist_ok=True
    )


    for i, (dataset_name, df) in enumerate(datasets.items()):


        # Protect against missing thresholds
        if i >= len(threshold_seconds):

            logger.warning(
                "Missing threshold for %s. "
                "Skipping short-file removal.",
                dataset_name
            )

            processed_datasets[dataset_name] = df

            continue


        threshold = threshold_seconds[i]


        # Find short files
        short_files = _check_short_files(
            df,
            threshold_seconds=threshold
        )


        # Save duration plot
        plot_path = Path(save_dir) / (
            f"01_{dataset_name}_files_duration.png"
        )


        _plot_duration(
            df,
            dataframe_name=dataset_name,
            show_plot=show_plot,
            save_path=str(plot_path)
        )


        # Remove short files
        if short_files:

            df = df[
                ~df["file"].isin(short_files)
            ].reset_index(drop=True)


            logger.info(
                "%s: removed %d short files.",
                dataset_name,
                len(short_files)
            )


        processed_datasets[dataset_name] = df


    return processed_datasets
    
    
"""    
# Note:
# the sMEEI subset contains one sample with a significantly shorter duration
# than the remaining recordings. This sample is excluded before the
# equal-duration preprocessing step, as retaining it would impose a
# considerable reduction in the available signal duration for all samples
# after truncation.


datasets = remove_short_files(
    datasets,
    threshold_seconds=config["audio"]["min_duration"],
    save_dir=str(
        PROJECT_ROOT /
        config["results"]["figures_dir"]
    ),
    show_plot=True
)
"""