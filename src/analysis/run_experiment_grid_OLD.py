# src/analysis/run_experiment_grid.py

from typing import Callable, Dict, List, Optional

import pandas as pd

from src.analysis.build_Xy import build_Xy
from src.analysis.experiment_types import (
    ExperimentConfig,
    ExperimentResult,
)


def run_experiment_grid(
    run_function: Callable,  # <--- Recebe a função como parâmetro de entrada
    datasets: Dict[str, pd.DataFrame],
    train_corpus: str,
    feature_sets: List[str],
    test_corpus: Optional[str] = None,
    num_iters: int = 100,
    print_report: bool = False,
) -> List[ExperimentResult]:
    """Run all feature-set experiments for one train/test corpus pair.

    Parameters
    ----------
    run_function : Callable
        The execution function used to run each individual experiment
        (e.g., `run_intracorpus` or `run_crosscorpus`).
    datasets : dict[str, pd.DataFrame]
        Dictionary containing all available corpora.
    train_corpus : str
        Name of the corpus used for training.
    feature_sets : list of str
        Feature configurations to evaluate.
    test_corpus : str, optional
        Name of the corpus used for testing.
        If None, the training corpus is also used for testing
        (intra-corpus evaluation).
    num_iters : int, optional
        Number of repeated cross-validation iterations.
    print_report : bool, optional
        Whether to print experiment summaries.

    Returns
    -------
    list of ExperimentResult
        Results for all evaluated feature configurations.
    """

    if test_corpus is None:
        test_corpus = train_corpus

    train_df = datasets[train_corpus]
    test_df = datasets[test_corpus]

    X_train, y_train = build_Xy(train_df)

    # Future-proof: these will be used for inter-corpus evaluation
    X_test, y_test = build_Xy(test_df)

    all_results = []

    for feat in feature_sets:

        cfg = ExperimentConfig(
            train_corpus=train_corpus,
            test_corpus=test_corpus,
            feature_set=feat,
            num_iters=num_iters,            
            class_weight="balanced" if "balanced" in feat else None,
        )

        result = run_function(
            X_train,
            y_train,
            config=cfg,
            print_report=print_report,
        )

        all_results.append(result)

    return all_results