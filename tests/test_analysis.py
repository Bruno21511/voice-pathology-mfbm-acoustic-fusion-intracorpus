# -*- coding: utf-8 -*-
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.analysis.experiment_types import ExperimentConfig, ExperimentResult
from src.analysis.apply_feature_selection import apply_feature_selection
from src.analysis.run_intracorpus import run_intracorpus
from src.analysis.run_intracorpus_grid import run_intracorpus_grid


# =====================================================================
# 1. GRID RUNNER TESTS (ORCHESTRATION & COMBINATORICS)
# =====================================================================
@patch("src.analysis.run_intracorpus_grid.build_Xy")
def test_run_experiment_grid_combinatorics(mock_build_Xy):
    """Verifies that the grid orchestrator correctly parses datasets dictionary,

    invokes build_Xy for training/testing sets, and triggers the run_function callback
    exactly once per requested feature configuration.
    """
    # 1. Setup mock dataframes representing individual corpora
    df_mysmeei = pd.DataFrame({"dummy_col": [1, 2, 3]})
    df_myusp = pd.DataFrame({"dummy_col": [4, 5, 6]})
    
    datasets_mock = {
        "mysMEEI": df_mysmeei,
        "myUSP": df_myusp
    }
    feature_sets_mock = ["acoustic", "spectral", "all"]

    # Mock build_Xy to return dummy X and y arrays
    X_dummy, y_dummy = np.array([[1]]), np.array([0])
    mock_build_Xy.return_value = (X_dummy, y_dummy)

    # 2. Setup mock run_function callback
    mock_run_function = MagicMock()
    mock_result_obj = MagicMock(spec=ExperimentResult)
    mock_run_function.return_value = mock_result_obj

    # 3. Execute Grid Orchestrator (Intra-corpus configuration)
    results = run_intracorpus_grid(
        run_function=mock_run_function,
        datasets=datasets_mock,
        train_corpus="mysMEEI",
        feature_sets=feature_sets_mock,
        num_iters=10,
    )

    # 4. Assertions
    # build_Xy deve ser chamado apenas 1 vez pois treino e teste usam o mesmo corpus (intra-corpus)
    assert mock_build_Xy.call_count == 1
    
    # run_function should be called exactly once per feature set (3 sets = 3 calls)
    assert mock_run_function.call_count == 3
    assert len(results) == 3
    
    # Inspect arguments passed to the callback function
    first_call_args, first_call_kwargs = mock_run_function.call_args_list[0]
    passed_config = first_call_kwargs["config"]
    assert isinstance(passed_config, ExperimentConfig)
    assert passed_config.train_corpus == "mysMEEI"
    assert passed_config.test_corpus == "mysMEEI"
    assert passed_config.feature_set == "acoustic"


# =====================================================================
# 2. SUPPORT FUNCTIONS TESTS (FEATURE SELECTION)
# =====================================================================
def test_apply_feature_selection_acoustic():
    """Ensure the acoustic feature selector extracts only the correct column indices."""
    X = np.arange(30).reshape(1, 30)
    train_idx = np.array([0])
    test_idx = np.array([0])

    acoustic_indices = [24, 25, 26]
    spectral_indices = list(range(24))

    X_tr, X_te = apply_feature_selection(
        X, train_idx, test_idx, "acoustic", 
        acoustic_cols=acoustic_indices, 
        spectral_cols=spectral_indices
    )

    np.testing.assert_array_equal(X_tr, [[24, 25, 26]])
    np.testing.assert_array_equal(X_te, [[24, 25, 26]])


def test_apply_feature_selection_combined_paper():
    """Ensure the combined_paper configuration applies block-wise PCA and appends acoustic features."""
    # Create 1 sample with 30 features
    X = np.random.rand(5, 30)
    train_idx = np.array([0, 1, 2, 3])
    test_idx = np.array([4])

    acoustic_indices = [24, 25, 26]
    spectral_indices = list(range(24))

    X_tr, X_te = apply_feature_selection(
        X, train_idx, test_idx, "combined_paper", 
        acoustic_cols=acoustic_indices, 
        spectral_cols=spectral_indices
    )

    # 4 blocks of PCA components (2 + 1 + 2 + 1 = 6 components) + 3 acoustic features = 9 columns expected
    assert X_tr.shape == (4, 9)
    assert X_te.shape == (1, 9)


def test_apply_feature_selection_spectral_pca():
    """Verify spectral_X mode extracts spectral features, scales them, and reduces to X components."""
    X = np.random.rand(5, 30)
    train_idx = np.array([0, 1, 2, 3])
    test_idx = np.array([4])

    acoustic_indices = [24, 25, 26]
    spectral_indices = list(range(24))

    # Ask for 2 PCA components from spectral features
    X_tr, X_te = apply_feature_selection(
        X, train_idx, test_idx, "spectral_2", 
        acoustic_cols=acoustic_indices, 
        spectral_cols=spectral_indices
    )

    assert X_tr.shape == (4, 2)
    assert X_te.shape == (1, 2)


def test_apply_feature_selection_combined_pca():
    """Verify combined_X mode scales the entire matrix and reduces it down to X components."""
    X = np.random.rand(5, 30)
    train_idx = np.array([0, 1, 2, 3])
    test_idx = np.array([4])

    acoustic_indices = [24, 25, 26]
    spectral_indices = list(range(24))

    # Ask for 4 PCA components from the whole array
    X_tr, X_te = apply_feature_selection(
        X, train_idx, test_idx, "combined_4", 
        acoustic_cols=acoustic_indices, 
        spectral_cols=spectral_indices
    )

    assert X_tr.shape == (4, 4)
    assert X_te.shape == (1, 4)


def test_apply_feature_selection_invalid():
    """Ensure a ValueError is raised with the exact error message when given an unknown key."""
    X = np.zeros((2, 30))
    acoustic_indices = [24, 25, 26]
    spectral_indices = list(range(24))
    
    # Updated regex string to exactly match your 'feature_set not recognised' error message
    with pytest.raises(ValueError, match="feature_set not recognised"):
        apply_feature_selection(
            X, np.array([0]), np.array([1]), "bad_key",
            acoustic_cols=acoustic_indices, 
            spectral_cols=spectral_indices
        )


# =====================================================================
# 3. CORE PIPELINE TESTS (INTRA-CORPUS EXPERIMENT EXECUTION)
# =====================================================================
@patch("src.analysis.run_intracorpus.apply_feature_selection")
@patch("src.analysis.run_intracorpus.SVC")
def test_run_intracorpus_binary_execution(mock_svc_cls, mock_feat_sel):
    """Test the complete intra-corpus classification cross-validation loop using ML mocks."""
    # 1. Setup synthetic binary classification arrays (10 samples)
    X_mock = np.random.rand(10, 27)
    y_mock = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    cfg = ExperimentConfig(
        train_corpus="mysMEEI",
        test_corpus="mysMEEI",
        feature_set="all",
        num_iters=2,  # Run 2 iterations for a fast test run
    )
    # Inject deterministic mock cross-validation properties
    cfg.n_splits = 2

    # 2. Configure Mock behaviors for feature selection and classifier
    mock_feat_sel.return_value = (np.ones((5, 3)), np.ones((5, 3)))

    mock_svc_instance = MagicMock()
    mock_svc_cls.return_value = mock_svc_instance
    mock_svc_instance.fit.return_value = mock_svc_instance

    # Mock predictions and continuous scores needed for evaluation metrics
    mock_svc_instance.predict.side_effect = [
        np.array([0, 0, 0, 1, 1]),  # Iter 0 - Fold 1
        np.array([0, 0, 1, 1, 1]),  # Iter 0 - Fold 2
        np.array([0, 0, 0, 1, 1]),  # Iter 1 - Fold 1
        np.array([0, 0, 1, 1, 1]),  # Iter 1 - Fold 2
    ]

    mock_svc_instance.decision_function.side_effect = [
        np.array([-1.0, -1.0, -1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, 1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, 1.0, 1.0, 1.0]),
    ]

    # 3. Run the intra-corpus validation experiment pipeline
    result = run_intracorpus(X_mock, y_mock, config=cfg, print_report=False)

    # 4. Assertions
    assert isinstance(result, ExperimentResult)
    assert result.config == cfg
    assert result.accuracy_mean == 1.0  # Perfect prediction configuration mapping match
    assert result.auc_mean == 1.0

    # 2 iterations * 2 validation folds = 4 total calls
    assert mock_feat_sel.call_count == 4
    assert mock_svc_instance.fit.call_count == 4