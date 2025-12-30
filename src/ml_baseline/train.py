from __future__ import annotations

from pathlib import Path
import json
import datetime as dt

import numpy as np
import pandas as pd
import joblib

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

from ml_baseline.splits import random_split
from ml_baseline.pipeline import build_pipeline


def _pick_data_path(root: Path, target: str) -> Path:
    cands = [
        root / "data" / "processed" / "features.csv",
        root / "data" / "processed" / "sample_features.csv",
    ]
    # Prefer file that contains the target column
    for p in cands:
        if p.exists():
            try:
                cols = pd.read_csv(p, nrows=0).columns.tolist()
                if target in cols:
                    return p
            except Exception:
                pass
    # fallback
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("No features file found under data/processed/")


def _infer_task(y: pd.Series) -> str:
    return "classification" if y.nunique(dropna=True) == 2 else "regression"


def _id_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if c.lower() in {"id", "user_id"} or c.lower().endswith("_id")]


def _ensure_stratify_ok(n: int, test_size: float) -> tuple[bool, float]:
    """
    Stratified split for binary classification requires test size >= 2 rows.
    """
    if n < 4:
        return False, test_size  # too tiny; disable stratify
    min_ts = 2 / n
    return True, max(test_size, min_ts)


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = None
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_train(*, target: str = "target", seed: int = 42, test_size: float = 0.2) -> str:
    root = Path(__file__).resolve().parents[2]
    (root / "models" / "runs").mkdir(parents=True, exist_ok=True)
    (root / "models" / "registry").mkdir(parents=True, exist_ok=True)

    data_path = _pick_data_path(root, target)
    df = pd.read_csv(data_path)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Columns={df.columns.tolist()}")

    y = df[target]
    task = _infer_task(y)

    # Stratify only for binary classification
    stratify_flag = (task == "classification") and (y.nunique(dropna=True) == 2)
    if stratify_flag:
        stratify_flag, test_size = _ensure_stratify_ok(len(df), test_size)

    # Split
    train_df, test_df = random_split(
        df,
        target=target,
        test_size=test_size,
        seed=seed,
        stratify=stratify_flag,
    )

    # Columns contract
    id_cols = _id_cols(df.columns.tolist())
    feature_cols = [c for c in df.columns if c not in set(id_cols + [target])]

    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]

    # --- Build + fit pipeline (Task 5) ---
    pipe = build_pipeline(X=X_train, task=task)

    if task == "classification":
        y_train = train_df[target].astype(int).to_numpy()
        y_true  = test_df[target].astype(int).to_numpy()
        pipe.fit(X_train, y_train)

        # Probabilities (avoid "string to float" by preprocessor)
        proba = pipe.predict_proba(X_test)
        y_score = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        y_pred = (y_score >= 0.5).astype(int)

        model_metrics = classification_metrics(y_true, y_score, threshold=0.5)

        preds = pd.DataFrame({"y_true": y_true, "y_score": y_score, "y_pred": y_pred})

        # --- Dummy baseline (Task 4) ---
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        dproba = dummy.predict_proba(X_test)
        dscore = dproba[:, 1] if dproba.shape[1] > 1 else dproba[:, 0]
        baseline = classification_metrics(y_true, dscore, threshold=0.5)

    else:
        y_train = train_df[target].astype(float).to_numpy()
        y_true  = test_df[target].astype(float).to_numpy()
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        model_metrics = regression_metrics(y_true, y_pred)

        preds = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

        # --- Dummy baseline (Task 4) ---
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        d_pred = dummy.predict(X_test)
        baseline = regression_metrics(y_true, d_pred)

    # Run dir
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{ts}__{task}__seed{seed}"
    run_dir = root / "models" / "runs" / run_id
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "schema").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    # Save schema
    (run_dir / "schema" / "input_schema.json").write_text(
        json.dumps(
            {
                "id_cols": id_cols,
                "target": target,
                "feature_cols": feature_cols,
                "forbidden_cols": [target],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save artifacts
    joblib.dump(pipe, run_dir / "model" / "model.joblib")

    # Tables
    keep_cols = (id_cols + feature_cols + [target]) if id_cols else (feature_cols + [target])
    test_df[keep_cols].to_csv(run_dir / "tables" / "holdout_input.csv", index=False)

    if id_cols:
        preds.insert(0, id_cols[0], test_df[id_cols[0]].values)
    preds.to_csv(run_dir / "tables" / "holdout_predictions.csv", index=False)

    # Metrics
    (run_dir / "metrics" / "baseline_holdout.json").write_text(
        json.dumps(
            {
                "baseline_type": dummy.__class__.__name__,
                "task": task,
                "seed": seed,
                "test_size": float(test_size),
                "target": target,
                "metrics": baseline,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (run_dir / "metrics" / "holdout_metrics.json").write_text(
        json.dumps(
            {
                "model_type": pipe.named_steps["model"].__class__.__name__,
                "task": task,
                "seed": seed,
                "test_size": float(test_size),
                "target": target,
                "metrics": model_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Meta + registry
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "task": task,
                "seed": seed,
                "data_path": str(data_path.relative_to(root)),
                "target": target,
                "id_cols": id_cols,
                "feature_cols": feature_cols,
                "test_size": float(test_size),
                "stratify": bool(stratify_flag),
                "created_at_utc": ts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "models" / "registry" / "latest.txt").write_text(run_id, encoding="utf-8")

    print(f"Run dir: {run_dir.resolve()}")
    print(f"Done. Saved run: {run_id}")
    return run_id
