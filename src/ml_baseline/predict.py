from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import joblib

from ml_baseline.schema import InputSchema, schema_from_dict, validate_and_align


@dataclass(frozen=True)
class PredictConfig:
    run_dir: Path
    input_path: Path
    output_path: Path
    threshold: Optional[float] = None


def read_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Input must be .csv or .parquet")


def write_tabular(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError("Output must be .csv or .parquet")


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def run_predict(cfg: PredictConfig) -> None:
    meta = _load_json(cfg.run_dir / "run_meta.json")

    schema_path = cfg.run_dir / "schema" / "input_schema.json"
    schema_raw = _load_json(schema_path)
    schema = schema_from_dict(schema_raw)

    if not schema.required_feature_columns:
        fcols = meta.get("feature_cols") or []
        id_cols = meta.get("id_cols") or []
        target_col = meta.get("target")
        schema = InputSchema(
            required_feature_columns=list(fcols),
            optional_id_columns=list(id_cols),
            forbidden_columns=[target_col] if target_col else [],
            feature_dtypes={c: "number" for c in fcols},
        )

    model = joblib.load(cfg.run_dir / "model" / "model.joblib")

    df_in = read_tabular(cfg.input_path)
    X, ids = validate_and_align(df_in, schema)

    saved_t = meta.get("threshold") or meta.get("decision_threshold") or meta.get("chosen_threshold")
    t = cfg.threshold if cfg.threshold is not None else (saved_t if isinstance(saved_t, (int, float)) else 0.5)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba[:, 0]
        out = pd.DataFrame({"score": score, "prediction": (score >= float(t)).astype(int)})
    else:
        out = pd.DataFrame({"prediction": model.predict(X)})

    if ids.shape[1] > 0:
        out = pd.concat([ids.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

    write_tabular(out, cfg.output_path)
