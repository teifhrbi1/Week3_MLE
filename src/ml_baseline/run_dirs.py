from __future__ import annotations
from pathlib import Path


def resolve_run_dir(run: str, *, models_dir: Path) -> Path:
    if run == "latest":
        p = models_dir / "registry" / "latest.txt"
        if not p.exists():
            raise FileNotFoundError("No latest.txt found. Train a model first.")
        run_id = p.read_text(encoding="utf-8").strip()
        if not run_id:
            raise FileNotFoundError("latest.txt is empty. Train a model first.")
        return models_dir / "runs" / run_id
    return Path(run).expanduser().resolve()
