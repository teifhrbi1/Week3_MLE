import pandas as pd
import pytest

from ml_baseline.schema import InputSchema, validate_and_align


def test_validate_and_align_forbidden_column_fails() -> None:
    schema = InputSchema(
        required_feature_columns=["a"],
        optional_id_columns=[],
        forbidden_columns=["y"],
        feature_dtypes={"a": "number"},
    )
    df = pd.DataFrame({"a": [1, 2], "y": [0, 1]})

    with pytest.raises(ValueError, match=r"Forbidden columns present"):
        validate_and_align(df, schema)


def test_validate_and_align_missing_required_column_fails() -> None:
    schema = InputSchema(
        required_feature_columns=["a", "b"],
        optional_id_columns=[],
        forbidden_columns=[],
        feature_dtypes={"a": "number", "b": "number"},
    )
    df = pd.DataFrame({"a": [1, 2]})

    with pytest.raises(ValueError, match=r"Missing required feature columns.*b"):
        validate_and_align(df, schema)
