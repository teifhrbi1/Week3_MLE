import json
from pathlib import Path
import pandas as pd

def main() -> None:
    runs_dir = Path("models/runs")
    run_dirs = [p for p in runs_dir.glob("*") if p.is_dir()]
    if not run_dirs:
        raise SystemExit("❌ No runs found under models/runs. Run: uv run ml-baseline train")

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)

    schema_path = latest_run / "schema.json"
    if not schema_path.exists():
        raise SystemExit(f"❌ schema.json not found in latest run: {latest_run.name}")

    schema = json.loads(schema_path.read_text())
    id_cols = schema.get("id_cols") or []
    feature_cols = schema.get("feature_cols") or []
    forbidden_cols = schema.get("forbidden_cols") or []
    target_col = schema.get("target")

    inp = Path("data/processed/features.csv")
    if not inp.exists():
        raise SystemExit(f"❌ Missing input file: {inp}")

    df = pd.read_csv(inp)

    # Drop forbidden + target (safe for inference)
    drop_cols = [c for c in forbidden_cols if c in df.columns]
    if target_col and target_col in df.columns:
        drop_cols.append(target_col)
    if drop_cols:
        df = df.drop(columns=sorted(set(drop_cols)))

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print("❌ Latest run:", latest_run.name)
        print("❌ Input columns:", list(df.columns))
        print("❌ Expected feature_cols:", feature_cols)
        print("❌ Missing:", missing)
        print("\n➡️ معنى هذا: latest run متدرّب على داتا مختلفة عن features.csv حقّتك.")
        print("➡️ الحل: درّبي run جديد على نفس features.csv (ثم اعيدي predict).")
        raise SystemExit(1)

    keep = [c for c in id_cols if c in df.columns] + feature_cols

    out = Path("data/processed/features_infer.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(out, index=False)

    print("✅ Latest run:", latest_run.name)
    print("✅ Wrote:", out)
    print("✅ Columns:", keep)

if __name__ == "__main__":
    main()
