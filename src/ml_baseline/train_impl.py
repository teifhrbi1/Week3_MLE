from __future__ import annotations

from pathlib import Path
import json
import datetime as dt
import joblib
import pandas as pd

from ml_baseline.splits import random_split


def _pick_data_path(root: Path, target: str) -> Path:
    """
    Prefer features.csv if it exists and contains the requested target;
    otherwise fallback to sample_features.csv.
    """
    cands = [
        root / "data" / "processed" / "features.csv",
        root / "data" / "processed" / "sample_features.csv",
    ]
    for p in cands:
        if p.exists():
            try:
                cols = pd.read_csv(p, nrows=0).columns.tolist()
                if target in cols:
                    return p
            except Exception:
                pass
    # fallback: first existing
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("No features file found under data/processed/")


def run_train(target: str = "target", seed: int = 42, test_size: float = 0.2) -> str:
    root = Path(__file__).resolve().parents[2]
    root.joinpath("models", "runs").mkdir(parents=True, exist_ok=True)
    root.joinpath("models", "registry").mkdir(parents=True, exist_ok=True)

    data_path = _pick_data_path(root, target)
    df = pd.read_csv(data_path)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Columns={df.columns.tolist()}")

    # Determine task + stratify flag
    y = df[target]
    task = "classification" if y.nunique(dropna=True) == 2 else "regression"
    stratify_flag = (task == "classification") and (y.nunique(dropna=True) == 2)

    # Random stratified split (Task 3)
    train_df, test_df = random_split(
        df,
        target=target,
        test_size=test_size,
        seed=seed,
        stratify=stratify_flag,
    )

    # Baseline model: predict train mean (probability for classification)
    y_train = train_df[target].astype(float)
    p_mean = float(y_train.mean()) if len(y_train) else 0.5

    # Build run dir
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{ts}__{task}__seed{seed}"
    run_dir = root / "models" / "runs" / run_id
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "schema").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    # Schema
    id_cols = [c for c in df.columns if c.lower() in {"id", "user_id"} or c.lower().endswith("_id")]
    feature_cols = [c for c in df.columns if c not in set(id_cols + [target])]

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

    # Save model artifact (joblib)
    model_obj = {
        "type": "mean_baseline",
        "task": task,
        "seed": seed,
        "target": target,
        "p_mean": p_mean,
        "id_cols": id_cols,
        "feature_cols": feature_cols,
    }
    joblib.dump(model_obj, run_dir / "model" / "model.joblib")

    # Tables
    (test_df[id_cols + feature_cols + [target]] if id_cols else test_df[feature_cols + [target]]).to_csv(
        run_dir / "tables" / "holdout_input.csv",
        index=False,
    )

    # Predictions
    if task == "classification":
        y_true = test_df[target].astype(int)
        y_proba = pd.Series([p_mean] * len(test_df))
        y_pred = (y_proba >= 0.5).astype(int)

        preds = pd.DataFrame({"y_true": y_true, "y_proba": y_proba, "y_pred": y_pred})
    else:
        y_true = test_df[target].astype(float)
        y_pred = pd.Series([p_mean] * len(test_df))
        preds = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

    if id_cols:
        preds.insert(0, id_cols[0], test_df[id_cols[0]].values)

    preds.to_csv(run_dir / "tables" / "holdout_predictions.csv", index=False)

    # Metrics (minimal)
    if task == "classification":
        tp = int(((preds["y_true"] == 1) & (preds["y_pred"] == 1)).sum())
        tn = int(((preds["y_true"] == 0) & (preds["y_pred"] == 0)).sum())
        fp = int(((preds["y_true"] == 0) & (preds["y_pred"] == 1)).sum())
        fn = int(((preds["y_true"] == 1) & (preds["y_pred"] == 0)).sum())
        acc = float((tp + tn) / max(1, len(preds)))
        prec = float(tp / max(1, (tp + fp)))
        rec = float(tp / max(1, (tp + fn)))
        f1 = float((2 * prec * rec / max(1e-9, (prec + rec))) if (prec + rec) else 0.0)

        (run_dir / "metrics" / "baseline_holdout.json").write_text(
            json.dumps(
                {"baseline": "mean_target_probability", "train_positive_rate": p_mean, "holdout_size": int(len(preds))},
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "metrics" / "holdout_metrics.json").write_text(
            json.dumps(
                {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}},
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        (run_dir / "metrics" / "holdout_metrics.json").write_text(
            json.dumps({"holdout_size": int(len(preds))}, indent=2),
            encoding="utf-8",
        )

    # run_meta + registry/latest.txt
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
                "test_size": test_size,
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
