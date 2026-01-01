from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class InputSchema:
    required_feature_columns: list[str]
    optional_id_columns: list[str]
    forbidden_columns: list[str]
    feature_dtypes: dict[str, str]


def _is_numeric_dtype_str(dt: str) -> bool:
    s = (dt or "").lower()
    return any(
        x in s
        for x in ["int", "int64", "int32", "float", "float64", "float32", "number"]
    )


def validate_and_align(
    df_in: pd.DataFrame, schema: InputSchema
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate inference/training input against a schema and return:
      - X: DataFrame with REQUIRED features only, aligned in schema order
      - ids: DataFrame with OPTIONAL id columns if present (passthrough)

    Rules:
      - Fail fast if any forbidden columns are present (e.g., target/leakage)
      - Fail fast if any required features are missing (error names the columns)
      - Normalize dtypes: numeric -> to_numeric(errors="coerce"), else -> string dtype
    """
    # 1) Forbidden columns check (fail fast)
    forbidden_present = [c for c in schema.forbidden_columns if c in df_in.columns]
    if forbidden_present:
        raise ValueError(
            f"Forbidden columns present in inference input: {forbidden_present}"
        )

    # 2) Missing required features check (fail fast)
    missing = [c for c in schema.required_feature_columns if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    df = df_in.copy()

    # 3) Extract optional IDs if present
    id_cols = [c for c in schema.optional_id_columns if c in df.columns]
    ids = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)

    # 4) Normalize dtypes (tolerant: coerce)
    for c, dt in (schema.feature_dtypes or {}).items():
        if c not in df.columns:
            continue
        if _is_numeric_dtype_str(dt):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            # Use pandas "string" dtype to avoid mixed object issues
            df[c] = df[c].astype("string")

    # 5) Align X to required features only in the exact schema order
    X = df[schema.required_feature_columns].copy()
    return X, ids


def schema_from_dict(d: dict[str, Any]) -> InputSchema:
    # Helper (optional): supports slightly different keys
    return InputSchema(
        required_feature_columns=d.get("required_feature_columns")
        or d.get("required_columns")
        or [],
        optional_id_columns=d.get("optional_id_columns") or d.get("id_columns") or [],
        forbidden_columns=d.get("forbidden_columns") or d.get("forbidden") or [],
        feature_dtypes=d.get("feature_dtypes") or d.get("dtypes") or {},
    )
