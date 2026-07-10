"""
main.py

Intracorpus analysis and experiment pipeline for the voice pathology
classification project (sMEEI and USP corpora).

Loads the precomputed MFBM/acoustic features (produced by
build_datasets.py), merges pathology classes, runs the intracorpus
experiment grid across feature-set / PCA configurations, and exports
all resulting tables and figures.

Runs the same steps as the original notebook, but as a single command:

    python main.py
    python main.py --config config.yaml

Should be placed at the repository root (same level the notebook ran
from), since it assumes the same directory layout (config.yaml, src/,
data/processed/, results/, etc.).
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, does NOT open windows

import random
import numpy as np

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Project root directory (where this script is located)
PROJECT_ROOT = Path(__file__).resolve().parent

# Force project root into path to ensure local src modules are found
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Data
from src.data.import_dataframe import import_dataframe
from src.data.merge_pathology_classes import merge_pathology_classes

# --- Analysis
from src.analysis.compute_band_statistics import compute_band_statistics
from src.analysis.aggregate_band_statistics_per_class import aggregate_band_statistics_per_class
from src.analysis.build_experiment_config_names import build_experiment_config_names
from src.analysis.build_Xy import build_Xy  # noqa: F401 (kept: used in original notebook)
from src.analysis.run_intracorpus import run_intracorpus
from src.analysis.run_intracorpus_grid import run_intracorpus_grid
from src.analysis.flatten_results import flatten_results

# --- Visualization
from src.visualization.plot_mfbm_statistics_per_class import plot_mfbm_statistics_per_class
from src.visualization.plot_acoustic_features import plot_acoustic_features
from src.visualization.plot_results_bar_paper import plot_results_bar_paper
from src.visualization.plot_acoustic_confusion_matrices import plot_acoustic_confusion_matrices
from src.visualization.plot_metric_pcs_grid import plot_metric_pcs_grid
from src.visualization.plot_combined_violin import plot_combined_violin

# --- Evaluation
from src.evaluation.make_summary_table_paper_config import make_summary_table_paper_config
from src.evaluation.make_acoustic_configs_table import make_acoustic_configs_table
from src.evaluation.make_mean_by_mode_table import make_mean_by_mode_table
from src.evaluation.make_pc_analysis_tables import make_pc_analysis_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the intracorpus analysis and experiment pipeline (sMEEI/USP)."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """Loads the YAML configuration file."""
    logger.info("Loading configuration from: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_datasets(config: dict) -> dict:
    """Loads the precomputed MFBM/acoustic feature dataframes for all configured corpora."""
    corpora_names = config["data"]["corpora"]
    logger.info("Loading processed datasets: %s", corpora_names)

    datasets = {}
    for name in corpora_names:
        datasets[name] = import_dataframe(
            name, input_root=str(PROJECT_ROOT / "data" / "processed")
        )

    for name, df in datasets.items():
        logger.info("Inspecting %s dataframe structure: %s", name, list(df.columns))

    return datasets


def build_class_profiles(datasets: dict, config: dict, figures_dir: Path) -> dict:
    """Computes per-band statistics, merges pathology classes, and plots profiles per corpus."""
    merge_cfg = config["class_merging"]

    for dataset_name, df in datasets.items():
        logger.info("Processing profile for corpus: %s", dataset_name)

        # 1. Compute per-band mean and standard deviation statistics
        df = compute_band_statistics(df)

        # 2. Merge Reinke's edema and vocal nodules into a single 'physio' class
        df = merge_pathology_classes(
            df,
            classes_to_merge=merge_cfg["groups_to_merge"],
            merged_label=merge_cfg["merged_label"],
        )

        # Re-save the processed dataframe back into the dictionary so that
        # the next steps of the pipeline (like ML) use the updated data
        datasets[dataset_name] = df

        # 3. Aggregate statistics and generate profiles per class
        mean_dict, std_dict = aggregate_band_statistics_per_class(df)
        output_image_path = figures_dir / f"02_{dataset_name}_mean_std_per_class.png"
        plot_mfbm_statistics_per_class(mean_dict, std_dict, save_path=str(output_image_path))

    return datasets


def plot_acoustic_feature_distributions(datasets: dict, figures_dir: Path) -> None:
    """Plots acoustic features per class for all configured corpora."""
    for dataset_name, df in datasets.items():
        save_path = figures_dir / f"03_{dataset_name}_acoustic_features_per_class.png"
        plot_acoustic_features(
            df,
            log_transform=[],
            save_path=str(save_path),
        )


def build_used_configs(config: dict) -> list:
    """Builds the list of feature-set / PCA experiment configurations to evaluate.

    Paper configuration:
        PCA is applied independently to:
          - mean_MFBM bands 1-6  (2 PCs)
          - mean_MFBM bands 7-12 (1 PC)
          - std_MFBM  bands 1-6  (2 PCs)
          - std_MFBM  bands 7-12 (1 PC)
        Acoustic features are used without PCA.

    Experimental configurations (feature-set names generated below):
        acoustic
            Acoustic features only, no PCA.
        acoustic_balanced_weights
            Acoustic features only, no PCA, SVM with balanced class weights.
        spectral_paper
            Paper spectral configuration (spectral features only, 6 dimensions).
        combined_paper
            Paper combined configuration (spectral + acoustic features, 9 dimensions).
        spectral_xx
            PCA applied only to spectral features, xx retained spectral PCs.
        combined_xx
            Joint PCA applied to spectral and acoustic features, xx retained PCs.
        sep_xx_raw
            PCA applied only to spectral features, acoustic features kept in the
            original space.
        sep_xx_a01
            PCA applied separately to spectral features, acoustic features
            reduced to 1 PC.
        sep_xx_a02
            PCA applied separately to spectral features, acoustic features
            reduced to 2 PCs.
    """
    pca_cfg = config["pca"]["retained_components"]
    return build_experiment_config_names(pca_cfg["min"], pca_cfg["max"])


def run_experiments(datasets: dict, config: dict, used_configs: list, metrics_dir: Path):
    """Runs the full intracorpus experiment grid for all corpora and exports tidy results."""
    corpora_names = config["data"]["corpora"]

    logger.info("Running experiment grid for all corpora")
    all_results = []
    for name in corpora_names:
        results = run_intracorpus_grid(
            run_intracorpus,
            datasets,
            train_corpus=name,
            feature_sets=used_configs,
            num_iters=config["experiments"]["num_iters"],
            print_report=False,
        )
        all_results.extend(results)

    tidy_metrics, tidy_cm = flatten_results(all_results)

    tidy_metrics.to_csv(metrics_dir / "01_intracorpus_tidy_metrics.csv", index=False, sep=";")
    tidy_cm.to_csv(metrics_dir / "01_intracorpus_tidy_cm.csv", index=False, sep=";")

    return tidy_metrics, tidy_cm


def analyse_paper_config(tidy_metrics, metrics_dir: Path, figures_dir: Path) -> None:
    """Builds and exports the summary table and plots for the paper's reference configuration."""
    table_paper = make_summary_table_paper_config(tidy_metrics)

    logger.info("Paper configurations results:\n%s", table_paper)
    table_paper.to_csv(metrics_dir / "02_summarized_intracorpus_results_paper.csv", index=False, sep=";")

    plot_results_bar_paper(
        tidy_metrics,
        metric="accuracy",
        save_path=str(figures_dir / "04_accuracy_paper_config.png"),
    )
    plot_results_bar_paper(
        tidy_metrics,
        metric="auc",
        save_path=str(figures_dir / "04_auc_paper_config.png"),
    )


def analyse_pca_component_configs(tidy_metrics, config: dict, metrics_dir: Path, figures_dir: Path) -> None:
    """Analyses results across PCA component configurations: grids, tables, and violin plots."""
    corpora_names = config["data"]["corpora"]

    for metric_name in ("accuracy", "auc"):
        save_path = figures_dir / f"05_{metric_name}_results.png"
        plot_metric_pcs_grid(tidy_metrics, metric_name=metric_name, save_path=str(save_path))

    mean_by_mode_table = make_mean_by_mode_table(tidy_metrics)
    logger.info("Metrics mean value by mode results:\n%s", mean_by_mode_table)
    mean_by_mode_table.to_csv(metrics_dir / "03_intracorpus_mean_by_mode.csv", index=True, sep=";")

    # Results per config (components): spectral, mean combined, best combined, best combined config
    tables_per_nr_pcs = make_pc_analysis_tables(
        tidy_metrics, target_feature_sMEEI="sep_a01", target_feature_USP="combined"
    )
    for name in corpora_names:
        logger.info("%s analysis table:\n%s", name.upper(), tables_per_nr_pcs[name])
        filename = f"04_intracorpus_results_per_spectral_pcs_{name}.csv"
        tables_per_nr_pcs[name].to_csv(metrics_dir / filename, index=True, sep=";")

    plot_combined_violin(
        tidy_metrics,
        metric_name="accuracy",
        custom_config_sMEEI="sep_raw",
        custom_config_USP="combined",
        save_path=str(figures_dir / "06_Accuracy_results.png"),
    )
    plot_combined_violin(
        tidy_metrics,
        metric_name="auc",
        custom_config_sMEEI="sep_raw",
        custom_config_USP="combined",
        save_path=str(figures_dir / "06_AUC_results.png"),
    )


def analyse_acoustic_results(tidy_metrics, tidy_cm, config: dict, metrics_dir: Path, figures_dir: Path) -> None:
    """Builds and exports the acoustic-only configuration results table and confusion matrices."""
    corpora_names = config["data"]["corpora"]

    table_acoustic_config = make_acoustic_configs_table(tidy_metrics)
    logger.info("Acoustic configurations results:\n%s", table_acoustic_config)
    table_acoustic_config.to_csv(
        metrics_dir / "05_summarized_intracorpus_acoustic_results.csv", index=False, sep=";"
    )

    for name in corpora_names:
        save_path = figures_dir / f"07_{name}_acoustic_configs_confusion_matrices.png"
        plot_acoustic_confusion_matrices(
            tidy_cm_df=tidy_cm,
            test_corpus=name,
            save_path=str(save_path),
        )


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    figures_dir = PROJECT_ROOT / config["results"]["figures_dir"]
    metrics_dir = PROJECT_ROOT / config["results"]["metrics_dir"]

    datasets = load_datasets(config)
    datasets = build_class_profiles(datasets, config, figures_dir)
    plot_acoustic_feature_distributions(datasets, figures_dir)

    used_configs = build_used_configs(config)
    tidy_metrics, tidy_cm = run_experiments(datasets, config, used_configs, metrics_dir)

    analyse_paper_config(tidy_metrics, metrics_dir, figures_dir)
    analyse_pca_component_configs(tidy_metrics, config, metrics_dir, figures_dir)
    analyse_acoustic_results(tidy_metrics, tidy_cm, config, metrics_dir, figures_dir)

    logger.info("Analysis and experiment pipeline complete. Done.")


if __name__ == "__main__":
    main()