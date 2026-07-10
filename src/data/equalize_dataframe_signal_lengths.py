import numpy as np
import logging

from typing import Literal, Optional

import pandas as pd


logger = logging.getLogger(__name__)


PreserveMode = Literal[
    "beginning",
    "middle",
    "end"
]


def equalize_dataframe_signal_lengths(
    datasets: dict[str, pd.DataFrame],
    preserve: PreserveMode = "middle",
    normalize: Optional[str] = None,
    dc_remove: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Equalize signal duration across multiple datasets
    using time duration instead of raw sample count.

    The target duration is defined as the shortest signal duration
    among all provided datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Dictionary containing datasets.
        Each dataframe must contain:
            - signal
            - fs

    preserve : {"beginning", "middle", "end"}
        Region of the signal to preserve when truncating.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Processed datasets with equal signal duration.
    """


    valid_modes = [
        "beginning",
        "middle",
        "end"
    ]


    if preserve not in valid_modes:

        raise ValueError(
            f"preserve must be one of {valid_modes}"
        )


    # -----------------------------------------
    # Compute minimum duration per dataset
    # -----------------------------------------

    min_durations = {}


    for dataset_name, df in datasets.items():

        durations = [
            len(sig) / fs
            for sig, fs in zip(
                df["signal"],
                df["fs"]
            )
        ]

        min_durations[dataset_name] = min(durations)


    global_min_duration = min(
        min_durations.values()
    )


    logger.info(
        "Minimum duration per dataset (s): %s",
        min_durations
    )

    logger.info(
        "Global minimum duration (s): %.3f",
        global_min_duration
    )


    # -----------------------------------------
    # Truncate signals
    # -----------------------------------------

    datasets_out = {}


    for dataset_name, df in datasets.items():

        df_new = df.copy()

        truncated_signals = []


        for signal, fs in zip(
            df_new["signal"],
            df_new["fs"]
        ):


            target_len = int(
                global_min_duration * fs
            )

            length = len(signal)


            if length <= target_len:

                truncated_signals.append(signal)

                continue


            if preserve == "beginning":

                signal_cut = signal[:target_len]


            elif preserve == "end":

                signal_cut = signal[-target_len:]


            else:  # middle

                start = (length - target_len) // 2

                signal_cut = signal[
                    start:start + target_len
                ]


            truncated_signals.append(
                signal_cut
            )


        df_new["signal"] = truncated_signals
        
        # -----------------------------
        # 1. DC removal
        # -----------------------------
        if dc_remove:

            df_new["signal"] = df_new["signal"].apply(
                lambda x: x - np.mean(x)
            )
            
            
        # -----------------------------
        # 5. Normalization
        # -----------------------------
        if normalize == "peak":

            df_new["signal"] = df_new["signal"].apply(
                lambda x: x / np.max(np.abs(x))
                if np.max(np.abs(x)) > 0 else x
            )


        elif normalize == "rms":

            df_new["signal"] = df_new["signal"].apply(
                lambda x: x / np.sqrt(np.mean(x**2))
                if np.sqrt(np.mean(x**2)) > 0 else x
            )


        datasets_out[dataset_name] = df_new


    return datasets_out