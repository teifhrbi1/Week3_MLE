from __future__ import annotations
import argparse
import csv
from pathlib import Path

from ml_baseline.train import run_train
from ml_baseline.show_run import show_run

def make_sample_data() -> None:
    out = Path("data/processed/sample_features.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "feature_1", "feature_2", "target"])
        w.writerow(["u001", 3, 36, 0])
        w.writerow(["u002", 7, 105, 1])
        w.writerow(["u003", 1, 9, 0])
        w.writerow(["u004", 10, 80, 1])
    print(f"✅ wrote: {out}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ml-baseline", description="Week 3 ML baseline system")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train baseline")
    p_train.add_argument("--target", default="target")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--test-size", type=float, default=0.2)

    p_show = sub.add_parser("show-run", help="Show run metadata")
    p_show.add_argument("which", nargs="?", default="latest", help="latest or a run_id")

    sub.add_parser("evaluate", help="Evaluate (placeholder)")
    sub.add_parser("report", help="Report (placeholder)")
    sub.add_parser("make-sample-data", help="Create sample data")
    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "make-sample-data":
        make_sample_data()
        return 0

    if args.cmd == "train":
        run_train(target=args.target, seed=args.seed, test_size=args.test_size)
        return 0

    if args.cmd == "show-run":
        show_run(args.which)
        return 0

    print(f"✅ Command received: {args.cmd} (placeholder)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
