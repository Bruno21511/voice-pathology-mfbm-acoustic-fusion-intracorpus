# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)


def plot_combined_violin(
    tidy_metrics_df: pd.DataFrame,
    metric_name: str = "accuracy",
    custom_config_sMEEI: str = "combined",  # "spectral", "combined", "sep_raw", "sep_a01", "sep_a02"
    custom_config_USP: str = "combined",  # "spectral", "combined", "sep_raw", "sep_a01", "sep_a02"
    figsize: tuple[float, float] = (14, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """Creates a 6-violin grid comparing architectural families and custom sub-families
    for mysMEEI and myUSP datasets based on tidy metrics.
    """
    df = tidy_metrics_df.copy()

    # 1. Base filters: target metric and discard all paper baselines
    df = df[df["metric"] == metric_name]
    df = df[~df["features"].str.contains("_paper", na=False)]

    if df.empty:
        print(f"Warning: No data found for metric '{metric_name}'")
        return

    # Helper to scale accuracy to percentage if needed
    is_acc: bool = metric_name.lower() == "accuracy"
    scale_factor: float = 100 if is_acc else 1.0

    # Helper function to map generic input strings to exact regex filters
    def get_custom_mask(
        features_series: pd.Series, config_name: str
    ) -> pd.Series:
        if config_name == "spectral":
            return features_series.str.match(r"^spectral_\d+$")
        elif config_name == "combined":
            return features_series.str.match(r"^combined_\d+$")
        elif config_name == "sep_raw":
            return features_series.str.match(r"^sep_\d+_raw$")
        elif config_name == "sep_a01":
            return features_series.str.match(r"^sep_\d+_a0?1$")
        elif config_name == "sep_a02":
            return features_series.str.match(r"^sep_\d+_a0?2$")
        else:
            # Fallback direct match case sensitive
            return features_series == config_name

    # =====================================================
    # Data Filtering & Dynamic Corpus Label Logic
    # =====================================================

    # --- MEEI Datasets ---
    meei_df = df[df["test_corpus"] == "mysMEEI"]

    # Determinar texto para o grupo MEEI
    if not meei_df.empty and "train_corpus" in meei_df.columns:
        train_meei = meei_df["train_corpus"].iloc[0]
        label_meei = (
            "mysMEEI"
            if train_meei == "mysMEEI"
            else f"Trained in {train_meei},\ntested in mysMEEI"
        )
    else:
        label_meei = "mysMEEI"

    # V1: Spectral Family (starts with spectral_ followed by digits)
    v1_mask = meei_df["features"].str.match(r"^spectral_\d+$")
    v1_data: np.ndarray = (
        (meei_df[v1_mask]["value"] * scale_factor).dropna().to_numpy()
    )

    # V2: All Combined + All Separate features grouped together
    v2_mask = meei_df["features"].str.startswith("combined_") | meei_df[
        "features"
    ].str.startswith("sep_")
    v2_data: np.ndarray = (
        (meei_df[v2_mask]["value"] * scale_factor).dropna().to_numpy()
    )

    # V3: Custom sub-family matching parameter inputs
    v3_mask = get_custom_mask(meei_df["features"], custom_config_sMEEI)
    v3_data: np.ndarray = (
        (meei_df[v3_mask]["value"] * scale_factor).dropna().to_numpy()
    )

    # --- USP Datasets ---
    usp_df = df[df["test_corpus"] == "myUSP"]

    # Determinar texto para o grupo USP
    if not usp_df.empty and "train_corpus" in usp_df.columns:
        train_usp = usp_df["train_corpus"].iloc[0]
        label_usp = (
            "myUSP"
            if train_usp == "myUSP"
            else f"Trained in {train_usp},\ntested in myUSP"
        )
    else:
        label_usp = "myUSP"

    # V4: Spectral Family
    v4_mask = usp_df["features"].str.match(r"^spectral_\d+$")
    v4_data: np.ndarray = (
        (usp_df[v4_mask]["value"] * scale_factor).dropna().to_numpy()
    )

    # V5: All Combined + All Separate features grouped together
    v5_mask = usp_df["features"].str.startswith("combined_") | usp_df[
        "features"
    ].str.startswith("sep_")
    v5_data: np.ndarray = (
        (usp_df[v5_mask]["value"] * scale_factor).dropna().to_numpy()
    )

    # V6: Custom sub-family matching parameter inputs
    v6_mask = get_custom_mask(usp_df["features"], custom_config_USP)
    v6_data: np.ndarray = (
        (usp_df[v6_mask]["value"] * scale_factor).dropna().to_numpy()
    )

    data: list[np.ndarray] = [
        v1_data,
        v2_data,
        v3_data,
        v4_data,
        v5_data,
        v6_data,
    ]

    # =====================================================
    # Geometry and Positions
    # =====================================================
    gap: float = 1.2
    inner: float = 0.6

    pos_meei: list[float] = [1, 1 + inner, 1 + 2 * inner]
    pos_usp: list[float] = [
        pos_meei[-1] + gap,
        pos_meei[-1] + gap + inner,
        pos_meei[-1] + gap + 2 * inner,
    ]
    positions: list[float] = pos_meei + pos_usp

    # Colors for the 3 functional types (Spectral, Mixed, Custom Selection)
    colors: list[str] = ["#185FA5", "#0F6E56", "#993C1D"] * 2
    hatches: list[str] = ["", "", "", "///", "///", "///"]

    # =====================================================
    # Plotting
    # =====================================================
    fig, ax = plt.subplots(figsize=figsize)

    # Filter out empty arrays from building violin structural paths to prevent layout errors
    vp = ax.violinplot(
        data,
        positions=positions,
        widths=0.45,
        showmedians=True,
        showextrema=True,
        bw_method=0.3,
    )

    # Styling violin bodies
    for body, color, hatch in zip(vp["bodies"], colors, hatches):
        body.set_facecolor(color)
        body.set_alpha(0.65)
        body.set_hatch(hatch)
        body.set_edgecolor("white" if not hatch else "#555555")
        body.set_linewidth(0.8)

    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if part in vp:
            vp[part].set_edgecolor("#444444")
            vp[part].set_linewidth(1.2)

    # Overlay Points with Jitter
    for pos, vals, color in zip(positions, data, colors):
        if len(vals) == 0:
            continue
        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            color="white",
            edgecolors=color,
            linewidths=0.8,
            s=25,
            zorder=3,
            alpha=0.9,
        )

    # =====================================================
    # Labels and Cosmetics
    # =====================================================
    ylabel: str = "Accuracy (%)" if is_acc else metric_name.upper()
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"{ylabel} Distribution Grouped by Architectural Families",
        fontsize=12,
    )

    # Beautiful Labels for X ticks reflecting custom settings
    tick_labels: list[str] = [
        "Spectral\n(All PCs)",
        "Comb + Sep\n(All)",
        f"Custom Highlight\n({custom_config_sMEEI})",
        "Spectral\n(All PCs)",
        "Comb + Sep\n(All)",
        f"Custom Highlight\n({custom_config_USP})",
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=9)

    # Corpus Grouping Titles (MEEI vs USP) - Now using dynamic texts
    y_min, y_max = ax.get_ylim()
    offset: float = (y_max - y_min) * 0.08

    ax.text(
        np.mean(pos_meei),
        y_min - offset,
        label_meei,
        ha="center",
        va="top",
        fontsize=11,  # Reduced from 12 to acommodate 2 lines if needed
        fontweight="bold",
        color="#333333",
    )
    ax.text(
        np.mean(pos_usp),
        y_min - offset,
        label_usp,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#333333",
    )

    # Separation Grid Line
    ax.axvline(
        (pos_meei[-1] + pos_usp[0]) / 2,
        color="#aaaaaa",
        linestyle="--",
        linewidth=1,
    )

    # Legend Customization
    legend_elements: list[Patch] = [
        Patch(
            facecolor="#185FA5",
            alpha=0.65,
            label="Spectral Family (2-15 PCs)",
        ),
        Patch(
            facecolor="#0F6E56",
            alpha=0.65,
            label="Mixed Approaches (All Comb/Sep)",
        ),
        Patch(
            facecolor="#993C1D",
            alpha=0.65,
            label="Custom Selection (All PCs)",
        ),
        Patch(
            facecolor="white",
            alpha=0.65,
            hatch="///",
            edgecolor="gray",
            label="myUSP Corpus (Hatched)",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(
            "Violin plot successfully saved to: %s", save_path
        )

    plt.show()