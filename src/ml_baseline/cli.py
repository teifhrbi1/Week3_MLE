import argparse
from pathlib import Path
import csv

DEFAULT_OUT = Path("data/processed/sample_features.csv")

def make_sample_data(out_path: Path, n_rows: int = 50) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "feature_1", "feature_2", "target"])
        for i in range(n_rows):
            f1 = i % 10
            f2 = (i * 3) % 7
            y = 1 if (f1 + f2) % 2 == 0 else 0
            w.writerow([f"row_{i:03d}", f1, f2, y])
    return out_path

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ml-baseline", description="Week 3 ML baseline system")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Train baseline (placeholder)")
    sub.add_parser("evaluate", help="Evaluate model (placeholder)")
    sub.add_parser("report", help="Generate reports (placeholder)")

    ps = sub.add_parser("make-sample-data", help="Create a tiny sample feature table under data/processed")
    ps.add_argument("--rows", type=int, default=50)
    ps.add_argument("--out", type=str, default=str(DEFAULT_OUT))

    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "make-sample-data":
        out = make_sample_data(Path(args.out), n_rows=int(args.rows))
        print(f"✅ wrote: {out}")
        return 0

    print(f"✅ Command received: {args.cmd}")
    return 0
