import numpy as np
import pandas as pd

from typing import Tuple, Union


def split_train_valid_test_on_id_clients(
    data: pd.DataFrame,
    test_size: float = 0.20,
    valid_size: float = 0.0,
    id_clients: str = "id_pdl",
    seed: int = 0,
    shuffle: bool = True,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
]:
    """
    Split a DataFrame into **train / validation / test** subsets without leaking
    rows that belong to the same logical entity (e.g., client, PDL, user).

    Parameters
    ----------
    data : pandas.DataFrame
        The full dataset.  
        It **must** contain an identifier column ``id_col`` — all rows sharing the
        same identifier are kept in the same split.
    test_size : float, default ``0.20``
        Fraction of *unique* identifiers to allocate to the **test** split.
        Must lie in the half-open interval ``[0, 1)``.
    valid_size : float, default ``0.0``
        Fraction of *unique* identifiers to allocate to the **validation** split,
        *after* removing the test set.  
        Set to ``0`` to skip the validation split.
    id_clients : str, default ``"id_pdl"``
        Name of the identifier column.
    seed : int, default ``0``
        Random seed ensuring reproducible splits.
    shuffle : bool, default ``True``
        Shuffle rows within each split.  
        Set to ``False`` if row order must be preserved.

    Returns
    -------
    tuple
        If ``valid_size == 0``  
        &nbsp;&nbsp;→ ``(train_df, test_df)``  
        else  
        &nbsp;&nbsp;→ ``(train_df, valid_df, test_df)``

    Notes
    -----
    * The function **guarantees** that no identifier appears in more than one split.
    * Returned DataFrames are *copies* of the input; altering them never changes
      ``data``.
    * The validation split is optional to keep the signature ergonomic when you
      only need a train/test cut.

    Examples
    --------
    ```python
    train_df, valid_df, test_df = split_train_valid_test(
        df, test_size=0.2, valid_size=0.1, id_col="client_id", seed=42
    )
    ```
    """
    # ---- sanity checks ----------------------------------------------------- #
    if not 0 <= test_size < 1:
        raise ValueError("`test_size` must be in the interval [0, 1].")
    if not 0 <= valid_size < 1:
        raise ValueError("`valid_size` must be in the interval [0, 1].")
    if test_size + valid_size >= 1:
        raise ValueError("`test_size + valid_size` must be < 1.")

    # ---- shuffle identifiers ---------------------------------------------- #
    rng = np.random.default_rng(seed)
    unique_ids = data[id_clients].unique()
    rng.shuffle(unique_ids)

    # ---- slice ids --------------------------------------------------------- #
    n_test = int(len(unique_ids) * test_size)
    test_ids = unique_ids[:n_test]
    remaining_ids = unique_ids[n_test:]

    n_valid = int(len(remaining_ids) * valid_size)
    valid_ids = remaining_ids[:n_valid]
    train_ids = remaining_ids[n_valid:]

    # ---- helper to materialise each split ---------------------------------- #
    def _subset(ids):
        df = data[data[id_clients].isin(ids)].copy()
        return (
            df.sample(frac=1, random_state=seed).reset_index(drop=True)
            if shuffle
            else df.reset_index(drop=True)
        )

    train_df = _subset(train_ids)
    test_df = _subset(test_ids)

    if valid_size == 0:
        return train_df, test_df

    valid_df = _subset(valid_ids)
    return train_df, valid_df, test_df


def balance_data(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Balance the input DataFrame by undersampling the majority class based on the specified label column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with a binary column for class labels.
    label_col : str
        The column name used as the binary class label.

    Returns
    -------
    pd.DataFrame
        A balanced DataFrame with equal number of positive and negative samples.
    """
    # Separate by label
    df_pos = df[df[label_col] == 1]
    df_neg = df[df[label_col] == 0]

    # Determine minority count
    n_min = min(len(df_pos), len(df_neg))

    # Sample each class to the size of the minority class
    df_pos_bal = df_pos.sample(n=n_min, random_state=42)
    df_neg_bal = df_neg.sample(n=n_min, random_state=42)

    # Concatenate and shuffle
    df_balanced = pd.concat([df_pos_bal, df_neg_bal]).sample(frac=1, random_state=42)

    return df_balanced