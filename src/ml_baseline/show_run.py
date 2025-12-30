from __future__ import annotations
from pathlib import Path
import json


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_run_id(which: str) -> str:
    root = _root()
    if which == "latest":
        p = root / "models" / "registry" / "latest.txt"
        if not p.exists():
            raise FileNotFoundError(
                "models/registry/latest.txt not found (run train first)."
            )
        return p.read_text(encoding="utf-8").strip()
    return which


def show_run(which: str = "latest") -> None:
    root = _root()
    run_id = resolve_run_id(which)
    run_dir = root / "models" / "runs" / run_id
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_meta.json not found for run_id={run_id}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    print("run_id:", run_id)
    print("run_dir:", run_dir.resolve())
    print("\nrun_meta.json:")
    print(json.dumps(meta, indent=2))

    # (Optional but helpful) verify expected artifacts exist
    expected = [
        run_dir / "model" / "model.joblib",
        run_dir / "schema" / "input_schema.json",
        run_dir / "metrics" / "baseline_holdout.json",
        run_dir / "metrics" / "holdout_metrics.json",
        run_dir / "tables" / "holdout_input.csv",
        run_dir / "tables" / "holdout_predictions.csv",
        run_dir / "run_meta.json",
        root / "models" / "registry" / "latest.txt",
    ]
    missing = [str(p.relative_to(root)) for p in expected if not p.exists()]
    if missing:
        print("\n❌ Missing artifacts:")
        for m in missing:
            print(" -", m)
    else:
        print("\n✅ All expected artifacts exist for this run.")
