from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class InputSchema:
    required_feature_columns: list[str]
    feature_dtypes: dict[str, str]
    optional_id_columns: list[str] = field(default_factory=list)
    forbidden_columns: list[str] = field(default_factory=list)

    @staticmethod
    def from_training_df(
        df: pd.DataFrame, *, target: str, id_cols: list[str]
    ) -> "InputSchema":
        optional_ids = [c for c in id_cols if c in df.columns]
        feature_cols = [c for c in df.columns if c not in set([target] + optional_ids)]
        feature_dtypes = {c: str(df[c].dtype) for c in feature_cols}
        return InputSchema(
            required_feature_columns=feature_cols,
            feature_dtypes=feature_dtypes,
            optional_id_columns=optional_ids,
            forbidden_columns=[target],
        )

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "required_feature_columns": self.required_feature_columns,
                    "feature_dtypes": self.feature_dtypes,
                    "optional_id_columns": self.optional_id_columns,
                    "forbidden_columns": self.forbidden_columns,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def load(path: Path) -> "InputSchema":
        d = json.loads(path.read_text(encoding="utf-8"))
        return InputSchema(
            required_feature_columns=list(d["required_feature_columns"]),
            feature_dtypes=dict(d["feature_dtypes"]),
            optional_id_columns=list(d.get("optional_id_columns", [])),
            forbidden_columns=list(d.get("forbidden_columns", [])),
        )


def validate_and_align(df_in: pd.DataFrame, schema: InputSchema) -> tuple[pd.DataFrame, pd.DataFrame]:
    forbidden = [c for c in schema.forbidden_columns if c in df_in.columns]
    assert not forbidden, f"Forbidden columns present in inference input: {forbidden}"

    missing = [c for c in schema.required_feature_columns if c not in df_in.columns]
    assert not missing, f"Missing required feature columns: {missing}"

    df = df_in.copy()

    # optional passthrough IDs
    id_cols = [c for c in schema.optional_id_columns if c in df.columns]
    passthrough = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)

    # dtype normalization (simple)
    for c, dt in schema.feature_dtypes.items():
        if c not in df.columns:
            continue
        if any(x in dt.lower() for x in ["int", "float"]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].astype("string")

    X = df[schema.required_feature_columns].copy()
    return X, passthrough
