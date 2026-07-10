# -*- coding: utf-8 -*-
import pandas as pd
import logging

from typing import Optional

logger = logging.getLogger(__name__)

def remove_group_classes(
    datasets: dict[str, pd.DataFrame],
    remove_groups: Optional[list[str]] = None
) -> dict[str, pd.DataFrame]:
    """
    Remove selected classes/groups from multiple datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Dictionary of datasets.

    remove_groups : list[str] or None
        Groups/classes to remove.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Processed datasets.
    """

    if remove_groups is None:
        remove_groups = []

    datasets_processed = {}

    for corpus_name, df in datasets.items():

        df_new = df.copy()

        if remove_groups:

            df_new = df_new[
                ~df_new["group"].isin(remove_groups)
            ].reset_index(drop=True)

        datasets_processed[corpus_name] = df_new
        
        logger.info(
            "%s: %d samples removed",
            corpus_name,
            len(df) - len(df_new)
)

    return datasets_processed