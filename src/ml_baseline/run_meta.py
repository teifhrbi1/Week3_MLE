from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable_cfg(cfg: Any) -> Any:
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "model_dump"):  # pydantic v2
        return cfg.model_dump()
    if hasattr(cfg, "dict"):  # pydantic v1
        return cfg.dict()
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return str(cfg)


def write_run_meta(
    run_dir: Path,
    run_id: str,
    cfg: Any,
    features_path: Path,
    baseline_holdout: Optional[Dict[str, Any]],
    holdout_metrics: Dict[str, Any],
    artifacts: Dict[str, Any],
    registry_dir: Path = Path("models/registry"),
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    features_path = Path(features_path)

    meta: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "features_path": str(features_path),
        "features_sha256": _sha256(features_path) if features_path.exists() else None,
        "cfg": _jsonable_cfg(cfg),
        "baseline_holdout": baseline_holdout,
        "holdout_metrics": holdout_metrics,
        "artifacts": artifacts,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "latest.txt").write_text(run_id + "\n", encoding="utf-8")

    return meta
