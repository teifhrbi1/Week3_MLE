from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _resolve_run_id(run_ref: str, registry_dir: Path = Path("models/registry")) -> str:
    if run_ref == "latest":
        p = registry_dir / "latest.txt"
        if not p.exists():
            raise FileNotFoundError("models/registry/latest.txt not found (no latest pointer yet).")
        return p.read_text(encoding="utf-8").strip()
    return run_ref


def load_run_meta(run_ref: str, runs_dir: Path = Path("models/runs")) -> Dict[str, Any]:
    run_id = _resolve_run_id(run_ref)
    meta_path = runs_dir / run_id / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_meta.json not found at: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def show_run(run_ref: str) -> None:
    meta = load_run_meta(run_ref)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
