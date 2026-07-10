# src/analysis/experiment_types.py
"""
Typed data structures representing configurations and results of
classification experiments.

These dataclasses replace loose dictionaries, ensuring that each
experiment always contains the expected fields (failing early if
something is missing). 
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ExperimentConfig:
    """
    Configuration for a single classification experiment.

    Parameters
    ----------
    train_corpus : str
        Identifies the corpus used for training.
    test_corpus : str
        Identifies the corpus used for testing.
    feature_set : str
        Feature set identifier (e.g., "acoustic", "spectral_05",
        "sep_05_a02"). Controls which features are selected and how
        PCA is applied inside the experiment pipeline.
    n_splits : int, optional
        Number of folds in stratified cross-validation (default: 5).
    num_iters : int, optional
        Number of cross-validation repetitions. Each repetition uses a
        different random state to reduce variance in the estimates
        (default: 100).
    class_weight : str or None, optional
        Class weighting strategy passed to the SVM classifier.
        Use "balanced" to compensate for class imbalance, or None for
        uniform weights (default: None).
    n_spectral : int, optional
        Number of spectral feature columns in the feature matrix,
        corresponding to the MFBM bands (default: 24).
    acoustic_cols : tuple of int, optional
        Column indices of the acoustic features (jitter, shimmer, HNR)
        in the feature matrix (default: (24, 25, 26)).
    """
    train_corpus: str
    test_corpus: str

    feature_set: str

    n_splits: int = 5
    num_iters: int = 100

    class_weight: Optional[str] = None  # None or "balanced"

    n_spectral: int = 24
    acoustic_cols: list = field(default_factory=lambda: [24, 25, 26])
    
    @property
    def spectral_cols(self) -> list[int]:
        """Column indices of the spectral (MFBM) features."""
        return list(range(self.n_spectral))


@dataclass
class ExperimentResult:
    """
    Aggregated result of a classification experiment after all
    cross validation repetitions.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration that produced this result — maintains traceability
        between results and the parameters used.
    accuracy_mean : float
        Mean accuracy across repetitions.
    accuracy_std : float
        Standard deviation of accuracy across repetitions.
    precision : float
        Mean precision (positive class).
    recall : float
        Mean recall (positive class).
    f1 : float
        Mean F1‑score (positive class).
    auc : float
        Mean ROC AUC across repetitions. Set to 0.0 for multi‑class
        problems where AUC is not computed.
    confusion_matrix : np.ndarray
        Confusion matrix averaged across repetitions (non‑normalised).
    confusion_matrix_norm : np.ndarray
        Confusion matrix normalised per row (true label), averaged
        across repetitions.
    """
    config: ExperimentConfig
    accuracy_mean: float
    accuracy_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float
    auc_mean: float
    auc_std: float
    confusion_matrix: np.ndarray
    confusion_matrix_norm: np.ndarray
