# -*- coding: utf-8 -*-
import pandas as pd


def make_pc_analysis_tables(
    tidy_metrics_df: pd.DataFrame, target_feature_sMEEI: str, target_feature_USP: str
) -> dict:
    """Generates two summary tables (one per dataset) analyzing accuracy metrics per PC.

    Calculates spectral accuracy, mean combined accuracy, best combined accuracy,
    and a custom targeted feature accuracy for PCs ranging from 2 to 15.
    Returns percentages formatted to 1 decimal place.

    Parameters
    ----------
    tidy_metrics_df : pd.DataFrame
        The original tidy DataFrame containing all metrics.
    target_feature_sMEEI : str
        The specific feature family to display in the 5th column for mysMEEI (e.g., 'combined', 'sep_raw', 'sep_a1').
    target_feature_USP : str
        The specific feature family to display in the 5th column for myUSP.

    Returns
    -------
    dict
        A dictionary containing two DataFrames: 'mysMEEI' and 'myUSP'.
    """
    # 1. Base filter: We only care about accuracy
    df = tidy_metrics_df.copy()
    df = df[df["metric"] == "accuracy"]

    # 2. Extract the PC number from the feature string
    def extract_pc(feature_name: str) -> int:
        if not isinstance(feature_name, str):
            return -1
        parts = feature_name.split("_")
        for part in parts:
            if part.isdigit():
                return int(part)
        return -1

    df["spec_pca"] = df["features"].apply(extract_pc)
    df = df[df["spec_pca"].between(2, 15)]

    datasets = ["mysMEEI", "myUSP"]
    results = {}

    for dataset in datasets:
        target_feat = (
            target_feature_sMEEI if dataset == "mysMEEI" else target_feature_USP
        )
        custom_col_name = f"{target_feat}_acc"

        ds_df = df[df["test_corpus"] == dataset]
        table_rows = []

        # Get unique valid PCs present in data
        valid_pcs = sorted(ds_df["spec_pca"].unique())

        for pc in valid_pcs:
            pc_data = ds_df[ds_df["spec_pca"] == pc]
            pc_str = f"{pc:02d}"  # Zero-padded string (e.g., '02', '05')

            # a) Extract spectral accuracy
            spec_row = pc_data[pc_data["features"] == f"spectral_{pc_str}"]
            spectral_acc = spec_row["value"].values[0] if not spec_row.empty else None

            # b) Gather the 4 specific models for combination analysis
            comb_names = [
                f"combined_{pc_str}",
                f"sep_{pc_str}_raw",
                f"sep_{pc_str}_a01",
                f"sep_{pc_str}_a02",
            ]
            comb_data = pc_data[pc_data["features"].isin(comb_names)]

            if not comb_data.empty:
                mean_comb_acc = comb_data["value"].mean()
                best_comb_acc = comb_data["value"].max()
            else:
                mean_comb_acc, best_comb_acc = None, None

            # c) Extract the targeted 5th column feature dynamically
            custom_acc = None
            if target_feat == "combined":
                target_name = f"combined_{pc_str}"
                custom_row = pc_data[pc_data["features"] == target_name]
                if not custom_row.empty:
                    custom_acc = custom_row["value"].values[0]
            else:
                # Handle sep_raw, sep_a1, sep_a2 variations flexibly
                suffix = target_feat.replace("sep_", "")
                if suffix == "a1":
                    suffix = "a01"
                elif suffix == "a2":
                    suffix = "a02"
                
                # Look for exact matching names like 'sep_02_raw' or 'sep_05_a01'
                target_name = f"sep_{pc_str}_{suffix}"
                custom_row = pc_data[pc_data["features"] == target_name]
                if not custom_row.empty:
                    custom_acc = custom_row["value"].values[0]

            # Format outputs safely as percentage text strings
            def to_pct(val):
                return f"{val * 100:.1f}%" if val is not None else "N/A"

            table_rows.append(
                {
                    "spec_pca": pc,
                    "spectral_acc": to_pct(spectral_acc),
                    "mean_comb_acc": to_pct(mean_comb_acc),
                    "best_comb_acc": to_pct(best_comb_acc),
                    custom_col_name: to_pct(custom_acc),
                }
            )

        # Build final DataFrame for the active dataset loop
        final_ds_df = pd.DataFrame(table_rows)
        if not final_ds_df.empty:
            final_ds_df.set_index("spec_pca", inplace=True)
        results[dataset] = final_ds_df

    return results