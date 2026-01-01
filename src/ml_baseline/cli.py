from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional
import typer

from .config import Paths, PredictConfig, TrainConfig
from .io import best_effort_ext
from .predict import resolve_run_dir, run_predict
from .sample_data import make_sample_feature_table
from .train import run_train

app = typer.Typer(add_completion=False, help="Week 3 — baseline ML system (train/eval/predict).")
log = logging.getLogger(__name__)


@app.callback()
def _main() -> None:
    pass


@app.command("make-sample-data")
def make_sample_data(n_users: int = 50) -> None:
    """Write a small demo feature table to data/processed/."""
    path = make_sample_feature_table(n_users=n_users)
    typer.echo(f"Wrote sample feature table: {path}")


@app.command()
def train(
    target: str = typer.Option(..., "--target", help="Target column name."),
    task: str = typer.Option("classification", "--task", help="classification|regression"),
    split_strategy: str = typer.Option("random", "--split-strategy", help="random|time|group"),
    features: Optional[Path] = typer.Option(None, "--features", help="Path to features table."),
    test_size: float = typer.Option(0.2, "--test-size", min=0.05, max=0.5),
    seed: int = typer.Option(42, "--seed"),
    threshold_strategy: str = typer.Option("fixed", "--threshold-strategy", help="fixed|max_f1"),
    threshold_value: float = typer.Option(0.5, "--threshold", min=0.0, max=1.0),
) -> None:
    """Train a baseline model and save artifacts."""
    paths = Paths.from_repo_root()
    feat_path = features
    if feat_path is None:
        ext = best_effort_ext()
        feat_path = paths.data_processed_dir / f"features{ext}"

    cfg = TrainConfig(
        features_path=feat_path,
        target=target,
        task=task,  # type: ignore[arg-type]
        split_strategy=split_strategy,  # type: ignore[arg-type]
        test_size=test_size,
        seed=seed,
        threshold_strategy=threshold_strategy,  # type: ignore[arg-type]
        threshold_value=threshold_value,
    )
    run_dir = run_train(cfg)
    typer.echo(f"Saved run: {run_dir}")


@app.command()
def predict(
    run: str = typer.Option("latest", "--run", help="'latest' or a path to a run dir."),
    input: Path = typer.Option(..., "--input", exists=True, help="Input CSV/Parquet."),
    output: Path = typer.Option(Path("outputs/preds.csv"), "--output", help="Output path."),
    threshold: float | None = typer.Option(None, "--threshold", help="Override decision threshold (classification)."),
) -> None:
    """Batch predict using a saved run."""
    paths = Paths.from_repo_root()
    run_dir = resolve_run_dir(run, models_dir=paths.models_dir)
    cfg = PredictConfig(run_dir=run_dir, input_path=input, output_path=output, threshold=threshold)
    run_predict(cfg)
    typer.echo(f"Wrote: {output}")


@app.command("show-run")
def show_run(run: str = "latest") -> None:
    """Print run_meta.json for a saved run."""
    paths = Paths.from_repo_root()
    run_dir = resolve_run_dir(run, models_dir=paths.models_dir)
    meta_path = run_dir / "run_meta.json"
    typer.echo(meta_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
