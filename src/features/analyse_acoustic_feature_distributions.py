# -*- coding: utf-8 -*-
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def analyse_acoustic_feature_distributions(
    datasets: Dict[str, pd.DataFrame],
) -> None:
    """
    Analyse the statistical distributions of the acoustic features
    (localjitter, localshimmer and hnr) for every dataframe contained
    in the datasets dictionary.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary whose keys are dataset names and values are
        pandas DataFrames.
    """

    candidate_distributions = {
        "gaussian": stats.norm,
    }

    acoustic_features = [
        "localjitter",
        "localshimmer",
        "hnr",
    ]

    for dataset_name, df in datasets.items():

        print("\n==============================")
        print(f"{dataset_name} acoustic feats analysis")
        print("==============================")

        results = []

        for feature in acoustic_features:

            data = df[feature].to_numpy()
            data = data[np.isfinite(data)]

            row = {
                "feature_name": feature,
                "mean": np.mean(data),
                "std": np.std(data),
                "skewness": stats.skew(data),
                "kurtosis": stats.kurtosis(data),
            }

            # Candidate distributions
            for dist_name, dist in candidate_distributions.items():
                try:
                    params = dist.fit(data)
                    _, p = stats.kstest(data, dist.name, args=params)
                    row[f"{dist_name}_p"] = p
                except Exception:
                    row[f"{dist_name}_p"] = np.nan

            results.append(row)

        results_df = (
            pd.DataFrame(results)
            .set_index("feature_name")
        )

        print(results_df)