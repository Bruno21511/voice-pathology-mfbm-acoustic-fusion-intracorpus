import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from src.analysis.experiment_types import ExperimentConfig, ExperimentResult
from src.analysis.apply_feature_selection import apply_feature_selection


def run_intracorpus(
    X: np.ndarray,
    y: np.ndarray,
    config: ExperimentConfig,
    print_report: bool = False,
) -> ExperimentResult:
    """Runs a repeated stratified k-fold cross-validation experiment using

    configurations.

    This function handles the complete pipeline for a single experiment task,
    including splitting the data, applying block-based feature selection (PCA),
    training the classifier, evaluating performance, and aggregating confusion
    matrices and metrics across all folds and repetitions.

    Parameters
    ----------
    X : np.ndarray
        The complete feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        The target labels array of shape (n_samples,).
    config : ExperimentConfig
        An instance containing configuration parameters such as the number of
        iterations, selected feature set, and classifier hyperparameters.
    print_report : bool, default=False
        If True, prints a detailed classification report and metrics summary
        to the console after completion.

    Returns
    -------
    ExperimentResult
        An object containing the aggregated evaluation metrics, raw predictions,
        and final confusion matrices for the cross-validation run.
    """
    
    acc = np.zeros(config.num_iters)
    pre = np.zeros(config.num_iters)
    rec = np.zeros(config.num_iters)
    f1s = np.zeros(config.num_iters)
    aucs = np.zeros(config.num_iters)

    conf_total = None
    conf_norm_total = None

    for ii in range(config.num_iters):
        cv = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=ii
        )

        y_true_all = []
        y_pred_all = []
        y_score_all = []

        for train_idx, test_idx in cv.split(X, y):
        
            # Feature selection via blocks
            X_tr, X_te = apply_feature_selection(
                X, 
                train_idx, 
                test_idx, 
                config.feature_set,
                config.acoustic_cols,
                config.spectral_cols
            )

            # Scaling
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_te = scaler.transform(X_te)

            # Model Execution
            clf = SVC(kernel='rbf', class_weight=config.class_weight).fit(X_tr, y[train_idx])
            y_pred = clf.predict(X_te)
            
            y_true_all.append(y[test_idx])
            y_pred_all.append(y_pred)

            y_score_all.append(clf.decision_function(X_te))

        y_true_all = np.hstack(y_true_all)
        y_pred_all = np.hstack(y_pred_all)

        conf = confusion_matrix(y_true_all, y_pred_all)
        acc[ii] = np.mean(y_true_all == y_pred_all)

        pre[ii] = precision_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
        rec[ii] = recall_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
        f1s[ii] = f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
            
        y_score_all = np.hstack(y_score_all)
        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
        aucs[ii] = auc(fpr, tpr)

        if conf_total is None:
            conf_total = conf
            conf_norm_total = confusion_matrix(y_true_all, y_pred_all, normalize="true")
        else:
            conf_total += conf
            conf_norm_total += confusion_matrix(y_true_all, y_pred_all, normalize="true")

    # Construct the strictly typed output
    result = ExperimentResult(
        config=config,
        accuracy_mean=float(acc.mean()),
        accuracy_std=float(acc.std()),
        precision_mean=float(pre.mean()),
        precision_std=float(pre.std()),
        recall_mean=float(rec.mean()),
        recall_std=float(rec.std()),
        f1_mean=float(f1s.mean()),
        f1_std=float(f1s.std()),
        auc_mean=float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else 0.0,
        auc_std=float(np.nanstd(aucs)) if not np.all(np.isnan(aucs)) else 0.0,
        confusion_matrix=conf_total / config.num_iters,
        confusion_matrix_norm=conf_norm_total / config.num_iters,
    )

    if print_report:
        print(f"\nFeatures: {config.feature_set}")
        print(f"Accuracy  : {result.accuracy_mean*100:.4f} ± {result.accuracy_std*100:.4f}")

    return result