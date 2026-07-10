# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import pytest

# Only importing what actually exists now
from src.analysis.flatten_results import flatten_results


# -------------------------------------------------------------------------
# 1. FIXTURES AND MOCKS
# -------------------------------------------------------------------------
@pytest.fixture
def mock_tidy_cm_df():
    """Generates a mock DataFrame simulating the long tidy format of 
    confusion matrices (7 columns) across multiple corpora and feature sets.
    """
    rows = [
        {
            "train_corpus": "mysMEEI",
            "test_corpus": "mysMEEI",
            "features": "acoustic",
            "true_index": 0,
            "pred_index": 0,
            "count": 28.05,
            "norm_value": 0.779167,
        },
        {
            "train_corpus": "mysMEEI",
            "test_corpus": "mysMEEI",
            "features": "acoustic",
            "true_index": 0,
            "pred_index": 1,
            "count": 7.95,
            "norm_value": 0.220833,
        },
        {
            "train_corpus": "mysMEEI",
            "test_corpus": "mysMEEI",
            "features": "acoustic",
            "true_index": 1,
            "pred_index": 0,
            "count": 7.00,
            "norm_value": 0.125000,
        },
        {
            "train_corpus": "mysMEEI",
            "test_corpus": "mysMEEI",
            "features": "acoustic",
            "true_index": 1,
            "pred_index": 1,
            "count": 49.00,
            "norm_value": 0.875000,
        }
    ]
    return pd.DataFrame(rows)


@dataclass
class MockConfig:
    task_name: str
    feature_set: str
    train_corpus: str = "mysMEEI"
    test_corpus: str = "mysMEEI"


@dataclass
class MockExperimentResult:
    """Mock to simulate the structured ExperimentResult class."""
    config: MockConfig
    confusion_matrix: List[List[int]]
    confusion_matrix_norm: List[List[float]]
    accuracy_mean: float = 0.85
    accuracy_std: float = 0.03
    precision_mean: float = 0.84
    precision_std: float = 0.04
    recall_mean: float = 0.83
    recall_std: float = 0.05
    f1_mean: float = 0.83
    f1_std: float = 0.05
    auc_mean: float = 0.90
    auc_std: float = 0.02


# -------------------------------------------------------------------------
# 2. TESTS FOR RESULT FLATTENING (flatten_results)
# -------------------------------------------------------------------------
def test_flatten_results_matrices_and_scalars_with_corpora():
    """Ensures that flatten_results extracts scalar metrics including the new

    corpus metadata columns (train_corpus, test_corpus).
    """
    config = MockConfig(
        task_name="Binary_Task",
        feature_set="acoustic",
        train_corpus="mysMEEI",
        test_corpus="mysMEEI"
    )

    cm_mock = [[10, 2], [1, 15]]
    cm_norm_mock = [[0.833, 0.167], [0.062, 0.938]]

    result_obj = MockExperimentResult(
        config=config,
        confusion_matrix=cm_mock,
        confusion_matrix_norm=cm_norm_mock,
    )

    # Execute flattening
    df_metrics, df_cm = flatten_results([result_obj])

    # Assertions for new mandatory corpus columns in Metrics DataFrame
    assert "train_corpus" in df_metrics.columns
    assert "test_corpus" in df_metrics.columns
    assert df_metrics["train_corpus"].iloc[0] == "mysMEEI"
    assert df_metrics["test_corpus"].iloc[0] == "mysMEEI"

    # Check if metrics are mapped correctly
    assert set(df_metrics["metric"].unique()).issuperset(
        {"accuracy", "precision", "recall", "f1", "auc"}
    )

    acc_row = df_metrics[df_metrics["metric"] == "accuracy"].iloc[0]
    assert acc_row["value"] == 0.85
    assert acc_row["std"] == 0.03


# -------------------------------------------------------------------------
# 3. TESTS FOR TIDY CONFUSION MATRIX STRUCTURE (tidy_cm)
# -------------------------------------------------------------------------
def test_tidy_cm_column_structure(mock_tidy_cm_df):
    """Verifies that the confusion matrix DataFrame contains all 7 mandatory columns

    and maintains the expected semantic data types.
    """
    expected_columns = {
        "train_corpus",
        "test_corpus",
        "features",
        "true_index",
        "pred_index",
        "count",
        "norm_value",
    }
    
    assert expected_columns.issubset(mock_tidy_cm_df.columns)
    assert pd.api.types.is_integer_dtype(mock_tidy_cm_df["true_index"])
    assert pd.api.types.is_integer_dtype(mock_tidy_cm_df["pred_index"])
    assert pd.api.types.is_numeric_dtype(mock_tidy_cm_df["count"])
    assert pd.api.types.is_numeric_dtype(mock_tidy_cm_df["norm_value"])


def test_tidy_cm_normalization_bounds(mock_tidy_cm_df):
    """Ensures that normalized values inside the tidy confusion matrix 

    strictly reside within the probabilistic range of [0.0, 1.0].
    """
    assert mock_tidy_cm_df["norm_value"].min() >= 0.0
    assert mock_tidy_cm_df["norm_value"].max() <= 1.0


def test_tidy_cm_row_totals_normalization(mock_tidy_cm_df):
    """Validates that the normalized values sum up to approximately 1.0 

    across each true class row (Row-normalized CM).
    """
    filtered_df = mock_tidy_cm_df[
        (mock_tidy_cm_df["train_corpus"] == "mysMEEI")
        & (mock_tidy_cm_df["test_corpus"] == "mysMEEI")
        & (mock_tidy_cm_df["features"] == "acoustic")
    ]

    row_sums = filtered_df.groupby("true_index")["norm_value"].sum()

    for true_class, total_prob in row_sums.items():
        assert pytest.approx(total_prob, abs=1e-4) == 1.0