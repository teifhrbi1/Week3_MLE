import pandas as pd
from sklearn.model_selection import train_test_split


def random_split(
    df: pd.DataFrame,
    *,
    target: str,
    test_size: float,
    seed: int,
    stratify: bool,
):
    """
    Random split with optional stratification for binary classification.

    Requirements:
    - uses train_test_split
    - uses stratify=y when stratify=True
    - returns train/test DataFrames with reset_index(drop=True)
    """
    y = df[target]
    strat = y if stratify else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
