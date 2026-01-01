
from __future__ import annotations
from pathlib import Path
import json
import argparse

def latest_run_dir(models_dir: Path) -> Path:
    runs_dir = models_dir / "runs"
    runs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not runs:
        raise SystemExit(f"❌ No runs found in: {runs_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)

def read_json(p: Path) -> dict:
    if not p.exists():
        raise SystemExit(f"❌ Missing file: {p}")
    return json.loads(p.read_text())

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def format_metric(v):
    if v is None:
        return "N/A"
    return f"{v:.4f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="latest")
    ap.add_argument("--out", default="reports/eval_summary.md")
    args = ap.parse_args()

    root = Path(".").resolve()
    models_dir = root / "models"
    run_dir = latest_run_dir(models_dir) if args.run == "latest" else (models_dir / "runs" / args.run)
    if not run_dir.exists():
        raise SystemExit(f"❌ Run dir not found: {run_dir}")

    meta = read_json(run_dir / "run_meta.json")
    metrics_path = run_dir / "metrics" / "holdout_metrics.json"
    metrics = read_json(metrics_path)

    # normalize
    baseline = metrics.get("baseline") or metrics.get("baseline_metrics")
    model = metrics.get("model") or metrics.get("model_metrics") or metrics

    preferred = ["roc_auc", "auc", "f1", "f1_score", "accuracy", "precision", "recall"]

    def gather(m):
        if not isinstance(m, dict):
            return {}
        out = {}
        for k in preferred:
            if k in m:
                out[k] = safe_float(m.get(k))
        for k, v in m.items():
            if k not in out and isinstance(v, (int, float)):
                out[k] = float(v)
        return out

    base_m = gather(baseline or {})
    model_m = gather(model or {})

    def metrics_lines(title, mdict):
        if not mdict:
            return [f"- {title}: N/A"]
        lines = [f"- {title}:"]
        for k in preferred:
            if k in mdict:
                lines.append(f"  - {k}: {format_metric(mdict[k])}")
        return lines

    task = meta.get("task", "classification")
    target = meta.get("target", "Unknown")
    data_path = meta.get("data_path", "Unknown")
    run_id = meta.get("run_id") or run_dir.name
    threshold = meta.get("threshold") or meta.get("decision_threshold") or meta.get("chosen_threshold") or 0.5

    md = []
    md.append("# Evaluation Summary — Week 3 Baseline\n")
    md.append("## What you trained")
    md.append(f"- Task: {task}")
    md.append(f"- Target: `{target}`")
    md.append(f"- Feature table used in training: `{data_path}`")
    md.append(f"- Run id: `{run_id}`\n")

    md.append("## Results (holdout)")
    md.extend(metrics_lines("Baseline", base_m))
    md.extend(metrics_lines("Model", model_m))
    md.append("")

    md.append("## Error analysis")
    md.append("- Review false positives/negatives by saving holdout predictions (y_true, y_pred, proba, ids).")
    md.append("- Leakage check: confirm target column removed from inference input and no post-outcome features.")
    md.append("- Slice errors by segments (e.g., country, n_orders bins) to identify weak areas.\n")

    md.append("## Recommendation")
    md.append("- Recommendation: **DON’T SHIP YET** unless baseline is present and model clearly improves holdout metric.")
    md.append(f"- Threshold used: {threshold}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"✅ Wrote: {out_path}")
    print(f"✅ Using run folder: {run_dir.name}")

if __name__ == "__main__":
    main()
