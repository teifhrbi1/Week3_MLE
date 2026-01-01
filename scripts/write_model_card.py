from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_run_dir(runs_dir: Path) -> Path:
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise SystemExit("❌ No runs found under models/runs. Run training first.")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def best_git_commit() -> str:
    # Prefer git commit if available; otherwise "N/A"
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "N/A"


def pick_dataset_hash(meta: Dict[str, Any], feat_path: Path) -> str:
    # Try common keys first, else compute from the feature table file
    for k in meta.keys():
        lk = k.lower()
        if "sha256" in lk or ("hash" in lk and "data" in lk) or ("dataset" in lk and "hash" in lk):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    if feat_path.exists():
        return sha256_file(feat_path)
    return "N/A"


def find_schema(run_dir: Path) -> Dict[str, Any]:
    # Try multiple possible schema locations
    candidates = [
        run_dir / "schema" / "input_schema.json",
        run_dir / "schema" / "schema.json",
        run_dir / "schema.json",
        run_dir / "input_schema.json",
    ]
    for p in candidates:
        if p.exists():
            return load_json(p)
    return {}


def normalize_schema(schema: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[list[str], list[str], list[str]]:
    # Return: id_cols, feature_cols, forbidden_cols (best-effort)
    id_cols = []
    feature_cols = []
    forbidden_cols = []

    # Newer style might store like:
    # required_feature_columns, optional_id_columns, forbidden_columns
    if "optional_id_columns" in schema and isinstance(schema["optional_id_columns"], list):
        id_cols = [str(x) for x in schema["optional_id_columns"]]
    if "required_feature_columns" in schema and isinstance(schema["required_feature_columns"], list):
        feature_cols = [str(x) for x in schema["required_feature_columns"]]
    if "forbidden_columns" in schema and isinstance(schema["forbidden_columns"], list):
        forbidden_cols = [str(x) for x in schema["forbidden_columns"]]

    # Older style might store like:
    # id_cols, feature_cols, forbidden_cols
    if not id_cols and isinstance(schema.get("id_cols"), list):
        id_cols = [str(x) for x in schema["id_cols"]]
    if not feature_cols and isinstance(schema.get("feature_cols"), list):
        feature_cols = [str(x) for x in schema["feature_cols"]]
    if not forbidden_cols and isinstance(schema.get("forbidden_cols"), list):
        forbidden_cols = [str(x) for x in schema["forbidden_cols"]]

    # Fall back to run_meta.json if schema is empty
    if not id_cols and isinstance(meta.get("id_cols"), list):
        id_cols = [str(x) for x in meta["id_cols"]]
    if not feature_cols and isinstance(meta.get("feature_cols"), list):
        feature_cols = [str(x) for x in meta["feature_cols"]]

    # Ensure target is forbidden at inference (common best practice)
    target = meta.get("target")
    if isinstance(target, str) and target and target not in forbidden_cols:
        forbidden_cols = forbidden_cols + [target]

    return id_cols, feature_cols, forbidden_cols


def load_holdout_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "metrics" / "holdout_metrics.json"
    if p.exists():
        return load_json(p)
    return {}


def split_baseline_model(metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Try multiple common shapes
    if isinstance(metrics.get("baseline"), dict) and isinstance(metrics.get("model"), dict):
        return metrics["baseline"], metrics["model"]
    if isinstance(metrics.get("baseline_metrics"), dict) and isinstance(metrics.get("model_metrics"), dict):
        return metrics["baseline_metrics"], metrics["model_metrics"]
    # If it's a flat dict (no baseline), treat as model metrics only
    return {}, metrics if isinstance(metrics, dict) else {}


def fmt_metrics(m: Dict[str, Any]) -> str:
    if not m:
        return "N/A (not recorded)"
    parts = []
    for k in sorted(m.keys()):
        v = m[k]
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            parts.append(f"{k}={v}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        elif isinstance(v, str) and len(v) <= 40:
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "N/A (non-numeric metrics)"


def decision_text(target: str) -> str:
    t = (target or "").lower()
    if "spam" in t:
        return "Flag incoming emails as spam to route/block messages."
    if "high_value" in t or "highvalue" in t:
        return "Flag high-value users for prioritization / VIP handling."
    return f"Predict `{target}` to support downstream routing/prioritization decisions."


def main() -> None:
    repo_root = Path(".")
    runs_dir = repo_root / "models" / "runs"
    run_dir = latest_run_dir(runs_dir)

    # Load run_meta (required)
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"❌ run_meta.json not found in latest run: {run_dir.name}")
    meta = load_json(meta_path)

    run_id = str(meta.get("run_id") or run_dir.name)
    target = str(meta.get("target") or "N/A")
    seed = meta.get("seed", "N/A")
    test_size = meta.get("test_size", "N/A")
    split_strategy = meta.get("split_strategy", "random")
    stratify = meta.get("stratify", True)

    # Feature table path used for training (prefer run_meta)
    data_path_str = str(meta.get("data_path") or "data/processed/features.csv")
    feat_path = repo_root / data_path_str if not Path(data_path_str).is_absolute() else Path(data_path_str)

    dataset_hash = pick_dataset_hash(meta, feat_path)

    schema = find_schema(run_dir)
    id_cols, feature_cols, forbidden_cols = normalize_schema(schema, meta)

    unit = "record"
    if id_cols:
        unit = f"entity identified by {id_cols}"
    elif "user_id" in (feature_cols or []):
        unit = "user"

    # Metrics
    holdout = load_holdout_metrics(run_dir)
    baseline_m, model_m = split_baseline_model(holdout)

    # Repro info
    git_commit = str(meta.get("git_commit") or meta.get("commit") or best_git_commit())
    pip_freeze_path = run_dir / "env" / "pip_freeze.txt"
    env_line = str(pip_freeze_path) if pip_freeze_path.exists() else "N/A (pip_freeze.txt not found)"

    # Write model card
    out_md = repo_root / "reports" / "model_card.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    md = f"""# Model Card — Week 3 Baseline

## Problem
- Target: `{target}`
- Unit of analysis: one row per {unit}
- Decision enabled: {decision_text(target)}

## Data
- Feature table: `{data_path_str}`
- Dataset hash (sha256): `{dataset_hash}`
- Required features: `{feature_cols}`
- ID columns (optional at inference): `{id_cols}`
- Forbidden columns at inference: `{forbidden_cols}`

## Splits
- Holdout: {split_strategy}{' stratified' if stratify else ''}, test_size={test_size}, seed={seed}

## Metrics (holdout)
- Baseline: {fmt_metrics(baseline_m)}
- Model: {fmt_metrics(model_m)}

## Limitations
- This is a baseline model; performance may be limited vs. tuned models.
- Metrics may not generalize if data distribution shifts (feature drift / label drift).
- Threshold selection and calibration are not fully optimized (monitor precision/recall trade-offs).

## Monitoring sketch
- Data quality checks: missingness, schema changes, out-of-range values, categorical drift.
- Performance checks (if labels become available): track accuracy/F1 (or PR-AUC), plus precision/recall by segment.
- Alerting: trigger alerts on drift thresholds or sustained metric drops; log predictions + input stats.

## Reproducibility
- Run id: `{run_id}`
- Git commit: `{git_commit}`
- Env: `{env_line}`
- Artifacts: `{run_dir}`
"""

    out_md.write_text(md)
    print(f"✅ Wrote: {out_md}")
    print(f"✅ Using latest run: {run_dir.name}")


if __name__ == "__main__":
    main()
