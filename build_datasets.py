"""
build_datasets.py

Dataset construction pipeline for the voice pathology classification
project (sMEEI and USP corpora).

Runs the same steps as the original notebook, but as a single command:

    python build_datasets.py
    python build_datasets.py --config config.yaml --corpora-root ../corpora

Should be placed at the repository root (same level the notebook ran
from), since it assumes the same directory layout (config.yaml, src/,
corpora/ one level above the repo, etc.).
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
from src.data.data_loader import data_loader
from src.data.remove_group_classes import remove_group_classes
from src.data.preprocessing import preprocessing
from src.data.remove_short_files import remove_short_files  # noqa: F401 (kept: used in original notebook)
from src.data.equalize_dataframe_signal_lengths import equalize_dataframe_signal_lengths
from src.data.export_dataframe import export_dataframe

# --- Features
from src.features.get_MFBM import get_MFBM
from src.features.extract_voice_features import extract_voice_features
from src.features.analyse_acoustic_feature_distributions import (  # noqa: F401
    analyse_acoustic_feature_distributions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build sMEEI/USP datasets from raw audio corpora, extracting MFBM and acoustic features."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--corpora-root", type=str, default=None, help="Path to shared corpora root")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """Loads the YAML configuration file."""
    logger.info("Loading configuration from: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_datasets(config: dict, corpora_root: Path) -> dict:
    """Loads the corpora (sMEEI, USP) and removes unwanted classes.

    sMEEI: 4 classes -> Control, Edema, Nodules, UVFP
    USP:   4 classes -> Control, Edema, Nodules, Neuro
    """
    logger.info("Loading datasets: %s", config["data"]["corpora"])
    datasets = {}
    for corpus_name in config["data"]["corpora"]:
        df = data_loader(dataset_name=corpus_name, data_root=str(corpora_root))
        datasets[corpus_name] = df

    logger.info("Removing 'neuro' and 'UVFP' classes")
    datasets = remove_group_classes(datasets, remove_groups=["neuro", "UVFP"])
    return datasets




def preprocess_datasets(datasets: dict, config: dict) -> dict:
    """Normalization, DC removal, silence trimming, and duration equalization."""
    normalize = config["audio"]["normalize"]
    dc_remove = config["audio"]["dc_remove"]
    trim_signal = config["audio"]["trim_signal"]

    logger.info(
        "Applying preprocessing (normalize=%s, dc_remove=%s, trim_signal=%s)",
        normalize, dc_remove, trim_signal,
    )
    for corpus_name, df in datasets.items():
        datasets[corpus_name] = preprocessing(
            df,
            normalize=normalize,
            dc_remove=dc_remove,
            trim_signal=trim_signal,
        )

    logger.info("Equalizing signal durations across datasets")
    datasets = equalize_dataframe_signal_lengths(
        datasets,
        preserve=config["audio"]["preserve"],
    )
    return datasets


def extract_mfbm(datasets: dict, config: dict) -> dict:
    """Extracts MFBM values for all signals.

    Note: the first and last 2 window values are discarded to mitigate
    eventual signal transition instabilities. Hann window by default.
    """
    feature_cfg = {
        "tamanho_in": config["features"]["frame_size_ms"],
        "passo_in": config["features"]["hop_size_ms"],
        "n_fft": config["features"]["n_fft"],
        "n_filters": config["features"]["n_filters"],
        "fmax": config["features"]["fmax"],
        "sobrep": config["features"]["overlap"],
        "window_type": config["features"]["window_type"],
        "edge_trim_frames": config["features"]["edge_trim_frames"],
    }

    logger.info("Extracting Mel-Frequency Band Magnitude (MFBM) features")
    for dataset_name, df in datasets.items():
        save_path = (
            PROJECT_ROOT / config["results"]["figures_dir"] / f"01_{dataset_name}_mel_filterbank.png"
        )
        datasets[dataset_name] = get_MFBM(
            df,
            **feature_cfg,
            print_filters=True,
            save_path=str(save_path),
        )

    # Note: a small amplitude difference can be observed between the Mel
    # filterbank outputs of the two datasets. This is caused by the
    # different sampling frequencies, which result in different FFT
    # frequency resolutions and therefore a different number of spectral
    # bins contributing to each Mel band. Since each Mel filter is
    # normalized to unit area, the filterbank output represents average
    # energy per band rather than total accumulated energy, hence the
    # observed amplitude difference.
    return datasets


def extract_acoustic_features(datasets: dict, config: dict) -> dict:
    """Extracts Jitter, Shimmer and HNR."""
    acoustic_cfg = config["acoustic_features"]
    logger.info("Extracting acoustic features (Jitter, Shimmer, HNR)")
    for dataset_name, df in datasets.items():
        datasets[dataset_name] = extract_voice_features(df, **acoustic_cfg, print_report=True)
    return datasets


def apply_log_transform(datasets: dict) -> dict:
    """Applies a log-transform to Jitter and Shimmer (HNR is already approximately gaussian)."""

    def to_log(x):
        # 1e-9 as a safety margin for zeros
        return np.log10(100 * x + 1e-9)

    logger.info("Applying log-transform to Jitter/Shimmer (USP and sMEEI)")
    for dataset_name in ("myUSP", "mysMEEI"):
        datasets[dataset_name]["localjitter"] = datasets[dataset_name]["localjitter"].apply(to_log)
        datasets[dataset_name]["localshimmer"] = datasets[dataset_name]["localshimmer"].apply(to_log)
    return datasets


def export_datasets(datasets: dict, config: dict) -> None:
    """Exports the processed dataframes to data/processed/."""
    output_root = PROJECT_ROOT / config["data"].get("processed_dir", "data/processed")
    logger.info("Exporting processed datasets to: %s", output_root)
    drop_columns = ["signal", "mfbm", "path", "fs", "meanf0", "stddevf0"]
    for dataset_name, df in datasets.items():
        export_dataframe(
            df,
            dataset_name,
            output_root=str(output_root),
            drop_columns=drop_columns,
        )


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    corpora_root = Path(args.corpora_root) if args.corpora_root else PROJECT_ROOT.parent / "corpora"

    datasets = load_datasets(config, corpora_root)
    datasets = preprocess_datasets(datasets, config)
    datasets = extract_mfbm(datasets, config)
    datasets = extract_acoustic_features(datasets, config)
    datasets = apply_log_transform(datasets)
    export_datasets(datasets, config)

    logger.info("Dataset construction complete. Done.")


if __name__ == "__main__":
    main()