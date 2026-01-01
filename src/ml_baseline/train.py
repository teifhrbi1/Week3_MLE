from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score

from .config import Paths, TrainConfig
from .io import best_effort_ext, read_tabular, write_tabular
from .metrics import (
    bootstrap_ci,
    choose_threshold_max_f1,
    classification_metrics,
    regression_metrics,
)
from .pipeline import build_pipeline
from .schema import InputSchema
from .splits import group_split, random_split, time_split

log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pip_freeze() -> str:
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as e:
        return f"# pip freeze failed: {e!r}\n"


def make_run_id(*, task: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{ts}__{task}__seed{seed}"


def _jsonable_cfg(cfg: TrainConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["features_path"] = str(cfg.features_path)
    d["id_cols"] = list(cfg.id_cols)
    return d


def run_train(cfg: TrainConfig, *, root: Path | None = None) -> Path:
    """Train a baseline ML model and save a versioned run folder."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = Paths.from_repo_root() if root is None else Paths(root=root)

    run_id = make_run_id(task=cfg.task, seed=cfg.seed)
    run_dir = paths.runs_dir / run_id

    # Create run folders
    for d in ["model", "metrics", "tables", "schema", "env"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    log.info("Run dir: %s", run_dir)

    # Load data
    df = read_tabular(cfg.features_path)
    assert cfg.target in df.columns, f"Missing target column: {cfg.target}"
    df = df.dropna(subset=[cfg.target]).reset_index(drop=True)

    # IDs should not be used as features
    id_cols_present = [c for c in cfg.id_cols if c in df.columns]

    # Optional: enforce one-row-per-id if id_cols provided
    if id_cols_present:
        dup = df.duplicated(subset=id_cols_present, keep=False)
        assert not dup.any(), f"Duplicate rows for id_cols={id_cols_present} (n={int(dup.sum())})"

    # Split
    stratify = cfg.task == "classification" and df[cfg.target].nunique() == 2
    if cfg.split_strategy == "random":
        train_df, test_df = random_split(
            df, target=cfg.target, test_size=cfg.test_size, seed=cfg.seed, stratify=stratify
        )
    elif cfg.split_strategy == "time":
        assert cfg.time_col, "time split requires --time-col"
        train_df, test_df = time_split(df, time_col=cfg.time_col, test_size=cfg.test_size)
    else:
        assert cfg.group_col, "group split requires --group-col"
        train_df, test_df = group_split(df, group_col=cfg.group_col, test_size=cfg.test_size, seed=cfg.seed)

    # X/y (drop target and IDs)
    drop_cols = [cfg.target, *id_cols_present]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[cfg.target]
    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df[cfg.target]

    # Schema contract
    schema = InputSchema.from_training_df(train_df, target=cfg.target, id_cols=list(cfg.id_cols))
    schema.dump(run_dir / "schema" / "input_schema.json")

    # ---- Baseline dummy (holdout) ----
    baseline: dict[str, Any]
    if cfg.task == "classification":
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        proba = dummy.predict_proba(X_test)
        y_score = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        y_true = np.asarray(y_test).astype(int)
        baseline = classification_metrics(y_true, y_score, threshold=0.5)
    else:
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        y_pred = dummy.predict(X_test)
        y_true = np.asarray(y_test).astype(float)
        baseline = regression_metrics(y_true, y_pred)

    (run_dir / "metrics" / "baseline_holdout.json").write_text(
        json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
    )

    # ---- Model pipeline ----
    pipe = build_pipeline(X=X_train, task=cfg.task)
    pipe.fit(X_train, y_train)

    ext = best_effort_ext()

    # Save holdout input for inference skew checks (IDs + required features)
    holdout_input = X_test.copy()
    if id_cols_present:
        holdout_input = pd.concat(
            [test_df[id_cols_present].reset_index(drop=True), holdout_input.reset_index(drop=True)], axis=1
        )
    holdout_input_path = run_dir / "tables" / f"holdout_input{ext}"
    write_tabular(holdout_input, holdout_input_path)

    # Holdout predictions + metrics
    if cfg.task == "classification":
        proba = pipe.predict_proba(X_test)
        y_score = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        y_true = np.asarray(y_test).astype(int)

        threshold = cfg.threshold_value
        if cfg.threshold_strategy == "max_f1":
            threshold = choose_threshold_max_f1(y_true, y_score)

        metrics = classification_metrics(y_true, y_score, threshold)
        metrics["positive_rate_holdout"] = float(y_true.mean())
        metrics["roc_auc_ci"] = bootstrap_ci(y_true, y_score, lambda a, b: float(roc_auc_score(a, b)))

        preds = pd.DataFrame({"score": y_score, "prediction": (y_score >= threshold).astype(int)})
        if id_cols_present:
            preds = pd.concat([test_df[id_cols_present].reset_index(drop=True), preds], axis=1)
        preds[cfg.target] = y_true

    else:
        y_pred = pipe.predict(X_test)
        y_true = np.asarray(y_test).astype(float)

        metrics = regression_metrics(y_true, y_pred)
        metrics["mae_ci"] = bootstrap_ci(y_true, y_pred, lambda a, b: float(mean_absolute_error(a, b)))

        preds = pd.DataFrame({"prediction": y_pred})
        if id_cols_present:
            preds = pd.concat([test_df[id_cols_present].reset_index(drop=True), preds], axis=1)
        preds[cfg.target] = y_true

    (run_dir / "metrics" / "holdout_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    holdout_preds_path = run_dir / "tables" / f"holdout_predictions{ext}"
    write_tabular(preds, holdout_preds_path)

    # Save model
    joblib.dump(pipe, run_dir / "model" / "model.joblib")

    # Environment snapshot
    (run_dir / "env" / "pip_freeze.txt").write_text(_pip_freeze(), encoding="utf-8")
    (run_dir / "env" / "env_meta.json").write_text(
        json.dumps({"python": sys.version, "platform": platform.platform()}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Run metadata (single file: what/why/how)
    meta = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": cfg.task,
        "features_path": str(cfg.features_path),
        "features_sha256": _sha256(cfg.features_path),
        "cfg": _jsonable_cfg(cfg),
        "baseline_holdout": baseline,
        "holdout_metrics": metrics,
        "artifacts": {
            "model": str((run_dir / "model" / "model.joblib").relative_to(run_dir)),
            "schema": str((run_dir / "schema" / "input_schema.json").relative_to(run_dir)),
            "holdout_predictions": str(holdout_preds_path.relative_to(run_dir)),
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Update registry pointer
    paths.registry_dir.mkdir(parents=True, exist_ok=True)
    (paths.registry_dir / "latest.txt").write_text(run_id + "\n", encoding="utf-8")

    log.info("Done. Saved run: %s", run_id)
    return run_dir
