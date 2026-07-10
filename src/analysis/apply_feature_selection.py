# src/analysis/run_experiment_grid.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Support function
# ------------------------------------------------------------------
def apply_feature_selection(
    X: np.ndarray, 
    train_idx: np.ndarray, 
    test_idx: np.ndarray, 
    feature_set: str,
    acoustic_cols: list,
    spectral_cols: list
) -> tuple[np.ndarray, np.ndarray]:
    """Applies PCA per feature block and filters columns based on the feature set.

    Parameters
    ----------
    X : np.ndarray
        The complete feature matrix of shape (n_samples, n_features).
    train_idx : np.ndarray
        Array of indices corresponding to the training split.
    test_idx : np.ndarray
        Array of indices corresponding to the testing/validation split.
    feature_set : str
        The target feature subset to extract. Must be one of 'all',
        'spectral', or 'acoustic'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - X_tr : Formatted and transformed training feature matrix.
        - X_te : Formatted and transformed testing feature matrix.

    Raises
    ------
    ValueError
        If `feature_set` is not one of 'all', 'spectral', or 'acoustic'.
    """
        
    # 1. Acoustic-only features (no PCA required)
    if feature_set.startswith("acoustic"):
        X_tr = X[train_idx, :][:, acoustic_cols]  # Jitter, Shimmer, HNR
        X_te = X[test_idx, :][:, acoustic_cols]
        return X_tr, X_te
        
    # 2. Paper configs
        
    elif feature_set.endswith("paper"):

        # Define block slicing boundaries and target PCA components
        blocks = {
            "mfbm_mean1-6": (0, 6, 2),
            "mfbm_mean7-12": (6, 12, 1),
            "mfbm_std1-6": (12, 18, 2),
            "mfbm_std7-12": (18, 24, 1),
        }

        Xtr_out = []
        Xte_out = []

        # Apply PCA processing per spectral block
        for start, end, n_comp in blocks.values():
            pca = PCA(n_components=n_comp)
            Xtr_block = pca.fit_transform(X[train_idx, start:end])
            Xte_block = pca.transform(X[test_idx, start:end])

            Xtr_out.append(Xtr_block)
            Xte_out.append(Xte_block)

        # Append acoustic features if the 'all' set is selected
        if feature_set == "combined_paper":
            Xtr_out.append(X[train_idx, :][:, acoustic_cols])  # Jitter, Shimmer, HNR
            Xte_out.append(X[test_idx, :][:, acoustic_cols])

        return np.hstack(Xtr_out), np.hstack(Xte_out)
        
        
    # 3. Spectral-only and separated configs       
    elif feature_set.startswith("spectral") or feature_set.startswith("sep"):   
    
        n_comp = int(feature_set.split("_")[1])

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx, :][:, spectral_cols])
        Xte = scaler.transform(X[test_idx, :][:, spectral_cols])

        pca = PCA(n_components=n_comp)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)
        
        if feature_set.startswith("spectral"): 
            return Xtr, Xte     
            
        Xtr_acoustic = X[train_idx, :][:, acoustic_cols]
        Xte_acoustic = X[test_idx, :][:, acoustic_cols]
        
        if feature_set.endswith("raw"): 
            return np.hstack([Xtr, Xtr_acoustic]), np.hstack([Xte, Xte_acoustic])     
        
        scaler = StandardScaler()
        Xtr_acoustic = scaler.fit_transform(Xtr_acoustic)
        Xte_acoustic = scaler.transform(Xte_acoustic)        

        if feature_set.endswith("a01"): 
            pca = PCA(n_components=1)
            
        elif feature_set.endswith("a02"):    
            pca = PCA(n_components=2)     

        else:
            raise ValueError(
                "Something is wrong inside the S config loop"
            )           
            
        Xtr_acoustic = pca.fit_transform(Xtr_acoustic)
        Xte_acoustic = pca.transform(Xte_acoustic)
        
        return np.hstack([Xtr, Xtr_acoustic]), np.hstack([Xte, Xte_acoustic])    
        
    
        
    # 4. All combined
    elif feature_set.startswith("combined"):

        n_comp = int(feature_set.split("_")[1])

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx, :])
        Xte = scaler.transform(X[test_idx, :])


        pca = PCA(n_components=n_comp)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)

        return Xtr, Xte   
    
    
    else:
        raise ValueError(
            "feature_set not recognised"
        )

