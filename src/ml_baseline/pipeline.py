from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge


def build_pipeline(*, X: pd.DataFrame, task: str):
    """
    Week 3 baseline pipeline:
    - numeric: median imputation
    - categorical: most_frequent imputation + one-hot (ignore unknown)
    - classification: LogisticRegression
    - regression: Ridge
    """
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    if task == "classification":
        model = LogisticRegression(max_iter=500, solver="liblinear", random_state=42)
    else:
        model = Ridge()

    return Pipeline(steps=[("pre", pre), ("model", model)])
