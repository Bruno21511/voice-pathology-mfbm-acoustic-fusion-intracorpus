# src/analysis/run_experiment_grid.py

from typing import Callable, Dict, List, Optional

import pandas as pd

from src.analysis.build_Xy import build_Xy
from src.analysis.experiment_types import (
    ExperimentConfig,
    ExperimentResult,
)


def run_intracorpus_grid(
    run_function: Callable,
    datasets: Dict[str, pd.DataFrame],
    train_corpus: str,
    feature_sets: List[str],
    num_iters: int = 100,
    print_report: bool = False,
) -> List[ExperimentResult]:
    """Run all feature-set experiments for a single corpus (intracorpus evaluation).

    The corpus is used both for training and testing, with the actual
    train/test split performed internally by `run_function` (e.g. via
    repeated stratified k-fold cross-validation).

    Parameters
    ----------
    run_function : Callable
        The execution function used to run each individual experiment
        (e.g., `run_intracorpus`). Expected signature:
        `run_function(X, y, config, print_report=False) -> ExperimentResult`.
    datasets : dict[str, pd.DataFrame]
        Dictionary containing all available corpora.
    train_corpus : str
        Name of the corpus used for both training and testing.
    feature_sets : list of str
        Feature configurations to evaluate.
    num_iters : int, optional
        Number of repeated cross-validation iterations.
    print_report : bool, optional
        Whether to print experiment summaries.

    Returns
    -------
    list of ExperimentResult
        Results for all evaluated feature configurations.
    """
    df = datasets[train_corpus]
    X, y = build_Xy(df)

    all_results = []
    for feat in feature_sets:
        cfg = ExperimentConfig(
            train_corpus=train_corpus,
            test_corpus=train_corpus,
            feature_set=feat,
            num_iters=num_iters,
            class_weight="balanced" if "balanced" in feat else None,
        )
        result = run_function(
            X,
            y,
            config=cfg,
            print_report=print_report,
        )
        all_results.append(result)
    return all_results