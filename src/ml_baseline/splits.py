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


def group_split(df, group_col: str, test_size: float = 0.2, seed: int = 42):
    """Split rows into train/test بحيث نفس الـ group ما يتكرر في الطرفين."""
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    if group_col not in df.columns:
        raise ValueError(
            f"group_col '{group_col}' not found in df.columns={list(df.columns)}"
        )

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(df))
    groups = df[group_col].astype(str).values

    train_idx, test_idx = next(gss.split(idx, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def time_split(df, time_col: str, test_size: float = 0.2):
    """
    Time-based split:
    sort by time_col ascending, last test_size fraction is test.
    Returns: (train_df, test_df)
    """
    import pandas as pd

    if time_col not in df.columns:
        raise ValueError(
            f"time_col '{time_col}' not found in df.columns={list(df.columns)}"
        )

    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")

    if d[time_col].isna().all():
        raise ValueError(
            f"time_col '{time_col}' could not be parsed as datetime (all NaT)."
        )

    d = d.sort_values(time_col).reset_index(drop=True)

    n = len(d)
    if n < 2:
        return d.copy(), d.iloc[0:0].copy()  # train all, empty test

    cut = int(round(n * (1 - test_size)))
    cut = max(1, min(cut, n - 1))  # ensure non-empty train & test

    train_df = d.iloc[:cut].copy()
    test_df = d.iloc[cut:].copy()
    return train_df, test_df
