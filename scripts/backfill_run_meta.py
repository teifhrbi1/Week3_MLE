from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_features_file(schema_dir: Path) -> Path | None:
    if not schema_dir.exists():
        return None
    candidates = []
    # Prefer explicit "features" / "contract" / "schema" files
    patterns = [
        "features.*",
        "feature_table.*",
        "dataset_contract.*",
        "schema.*",
        "*.json",
        "*.yml",
        "*.yaml",
        "*.csv",
        "*.parquet",
    ]
    for pat in patterns:
        candidates.extend(sorted(schema_dir.glob(pat)))
    # Return first real file (not dir)
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_first_metrics(metrics_dir: Path) -> dict:
    if not metrics_dir.exists():
        return {}
    # pick first json in metrics/
    for p in sorted(metrics_dir.glob("*.json")):
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    # if none, return empty
    return {}


def list_artifacts(run_dir: Path) -> dict:
    # Provide a small, useful pointer map (relative paths)
    def rel(p: Path) -> str:
        return str(p.relative_to(run_dir))

    out = {}
    # common folders you already have
    for key, pat in [
        ("run_meta", "run_meta.json"),
        ("model_dir", "model"),
        ("metrics_dir", "metrics"),
        ("tables_dir", "tables"),
        ("schema_dir", "schema"),
    ]:
        p = run_dir / pat
        if p.exists():
            out[key] = rel(p)
    # common files if they exist
    for p in sorted((run_dir / "metrics").glob("*.json")):
        out[f"metrics_file:{p.name}"] = rel(p)
    for p in sorted((run_dir / "tables").glob("*")):
        if p.is_file():
            out[f"table:{p.name}"] = rel(p)
    for p in sorted((run_dir / "schema").glob("*")):
        if p.is_file():
            out[f"schema:{p.name}"] = rel(p)
    return out


def main():
    rid_path = Path("models/registry/latest.txt")
    if not rid_path.exists():
        raise SystemExit("❌ missing models/registry/latest.txt")

    rid = rid_path.read_text().strip()
    if not rid:
        raise SystemExit("❌ latest.txt empty")

    run_dir = Path("models/runs") / rid
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"❌ missing {meta_path}")

    meta = json.loads(meta_path.read_text())

    # Required top-level keys
    meta["run_id"] = meta.get("run_id") or rid
    meta["timestamp_utc"] = meta.get("timestamp_utc") or utc_now_iso_z()

    # cfg: try to load a cfg file if exists; else default to empty dict
    cfg = meta.get("cfg")
    if not isinstance(cfg, dict):
        # attempt load from schema/cfg.json if present
        cfg_path = run_dir / "schema" / "cfg.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception:
                cfg = {}
        else:
            cfg = {}
    meta["cfg"] = cfg

    # features_sha256: hash a "features/contract/schema" file inside schema/
    schema_dir = run_dir / "schema"
    f = pick_features_file(schema_dir)
    if f is None:
        # fallback: hash list of schema filenames so it's never empty
        names = "\n".join(sorted([p.name for p in schema_dir.glob("*") if p.is_file()]))
        meta["features_sha256"] = hashlib.sha256(names.encode("utf-8")).hexdigest()
        meta.setdefault("notes", []).append(
            "features_sha256 fallback: hashed schema filenames (no schema file found)."
        )
    else:
        meta["features_sha256"] = sha256_file(f)
        meta.setdefault("notes", []).append(f"features_sha256 source: {f.name}")

    # metrics + artifacts pointers
    metrics = meta.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        meta["metrics"] = load_first_metrics(run_dir / "metrics")
    meta["artifacts"] = list_artifacts(run_dir)

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print("✅ updated:", meta_path)


if __name__ == "__main__":
    main()
