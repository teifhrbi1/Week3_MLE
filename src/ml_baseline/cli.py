"""
Week 3 ML baseline system CLI
- Fixes entrypoint: ml-baseline expects `main` symbol.
- Adds: `ml-baseline show-run latest` to print run_meta.json as JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="Week 3 ML baseline system")


def _resolve_run_id(run_id: str) -> str:
    if run_id != "latest":
        return run_id
    p = Path("models/registry/latest.txt")
    if not p.exists():
        raise typer.Exit("❌ models/registry/latest.txt missing. Create registry pointer first.")
    rid = p.read_text().strip()
    if not rid:
        raise typer.Exit("❌ models/registry/latest.txt is empty.")
    return rid


@app.command("show-run")
def show_run(run_id: str = typer.Argument("latest", help="Run id OR 'latest'")) -> None:
    """Print run_meta.json for a run."""
    rid = _resolve_run_id(run_id)
    meta_path = Path("models/runs") / rid / "run_meta.json"
    if not meta_path.exists():
        raise typer.Exit(f"❌ Missing: {meta_path}")
    meta = json.loads(meta_path.read_text())
    print(json.dumps(meta, indent=2, ensure_ascii=False))


@app.command("train")
def train() -> None:
    """Train baseline (placeholder)."""
    typer.echo("train: placeholder")


@app.command("evaluate")
def evaluate() -> None:
    """Evaluate model (placeholder)."""
    typer.echo("evaluate: placeholder")


@app.command("report")
def report() -> None:
    """Generate reports (placeholder)."""
    typer.echo("report: placeholder")


def main() -> None:
    # Entry point expected by the installed console script
    app()


if __name__ == "__main__":
    main()
