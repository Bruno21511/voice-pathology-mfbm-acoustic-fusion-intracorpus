# -*- coding: utf-8 -*-
import pytest
import pandas as pd
import numpy as np


# -------------------------------------------------------------------------
# 1. FIXTURE FOR TIDY CONFUSION MATRIX (tidy_cm)
# -------------------------------------------------------------------------
@pytest.fixture
def mock_tidy_cm_df():
    """Generates a mock DataFrame simulating the long tidy format of 
    confusion matrices (7 columns) across multiple corpora and feature sets.
    """
    rows = [
        # mysMEEI -> mysMEEI | acoustic | 2x2 Matrix
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
        },
        # myUSP -> myUSP | sep_15_a02 | 2x2 Matrix
        {
            "train_corpus": "myUSP",
            "test_corpus": "myUSP",
            "features": "sep_15_a02",
            "true_index": 0,
            "pred_index": 0,
            "count": 6.63,
            "norm_value": 0.442000,
        },
        {
            "train_corpus": "myUSP",
            "test_corpus": "myUSP",
            "features": "sep_15_a02",
            "true_index": 0,
            "pred_index": 1,
            "count": 8.37,
            "norm_value": 0.558000,
        },
    ]
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# 2. UNIT TESTS FOR TIDY CM STRUCTURE AND INTEGRITY
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
    
    # Check if all required columns are present
    assert expected_columns.issubset(mock_tidy_cm_df.columns)
    
    # Ensure index coordinates are stored as integers for matrix indexing mapping
    assert pd.api.types.is_integer_dtype(mock_tidy_cm_df["true_index"])
    assert pd.api.types.is_integer_dtype(mock_tidy_cm_df["pred_index"])
    
    # Ensure quantitative targets are float/numeric due to cross-validation averaging
    assert pd.api.types.is_numeric_dtype(mock_tidy_cm_df["count"])
    assert pd.api.types.is_numeric_dtype(mock_tidy_cm_df["norm_value"])


def test_tidy_cm_normalization_bounds(mock_tidy_cm_df):
    """Ensures that normalized values inside the tidy confusion matrix 

    strictly reside within the probabilistic range of [0.0, 1.0].
    """
    assert mock_tidy_cm_df["norm_value"].min() >= 0.0
    assert mock_tidy_cm_df["norm_value"].max() <= 1.0


def test_tidy_cm_row_totals_normalization(mock_tidy_cm_df):
    """Validates that for any given experimental setup, the normalized values 

    sum up to approximately 1.0 across each true class row (Row-normalized CM).
    """
    # Filter for a single specific configuration scenario
    filtered_df = mock_tidy_cm_df[
        (mock_tidy_cm_df["train_corpus"] == "mysMEEI")
        & (mock_tidy_cm_df["test_corpus"] == "mysMEEI")
        & (mock_tidy_cm_df["features"] == "acoustic")
    ]

    # Group by true class and sum up the normalized probabilities
    row_sums = filtered_df.groupby("true_index")["norm_value"].sum()

    # Every row in a row-normalized confusion matrix must sum to 1.0
    for true_class, total_prob in row_sums.items():
        assert pytest.approx(total_prob, abs=1e-4) == 1.0


def test_tidy_cm_data_filtering_and_values(mock_tidy_cm_df):
    """Confirms that querying specific coordinate cells yields correct values,

    ensuring structural mapping wasn't scrambled during flattening pipelines.
    """
    # Locate the False Positive cell (True Class 0, Predicted Class 1) for acoustic features
    fp_cell = mock_tidy_cm_df[
        (mock_tidy_cm_df["train_corpus"] == "mysMEEI")
        & (mock_tidy_cm_df["features"] == "acoustic")
        & (mock_tidy_cm_df["true_index"] == 0)
        & (mock_tidy_cm_df["pred_index"] == 1)
    ].iloc[0]

    assert fp_cell["count"] == 7.95
    assert fp_cell["norm_value"] == 0.220833