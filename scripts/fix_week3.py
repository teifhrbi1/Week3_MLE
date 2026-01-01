"""
Week 3 — Fix script (safe + commented)

Why you were getting errors:
1) sample_data wrote a file without extension -> io.write_tabular رفضه.
2) cli.train called run_train(cfg) لكن run_train كان معرف بدون args -> TypeError.
3) بعد ما درّبتي run جديد، كان يتدرب على sample_features.csv بدل features.csv -> schema mismatch في predict.

What this script does:
- Ensures best_effort_ext() always returns ".csv" (supported everywhere).
- Patches run_train to accept cfg=None (backward compatible).
  If cfg is None -> cfg = TrainConfig()
- Removes "fallback" in cli.py that hides the bug (run_train() without cfg).
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ml_baseline"


def die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def py_files() -> list[Path]:
    if not SRC.exists():
        die(f"❌ Missing folder: {SRC}")
    return sorted(SRC.rglob("*.py"))


# ----------------------------
# 1) Patch sample_data.py
# ----------------------------
def patch_sample_data_ext() -> None:
    """
    Ensure sample data writer always uses a supported extension.
    Previously ext == '' produced .../features (no extension) -> ValueError in write_tabular().
    """
    candidates = []
    for p in py_files():
        t = read_text(p)
        if "best_effort_ext" in t and "make_sample_feature_table" in t:
            candidates.append(p)

    if not candidates:
        print("ℹ️ sample_data.py not found (skipping)")
        return

    p = next((x for x in candidates if x.name == "sample_data.py"), candidates[0])
    t = read_text(p)
    original = t

    # Replace best_effort_ext() implementation with a stable one
    if re.search(r"def\s+best_effort_ext\s*\(", t):
        t = re.sub(
            r"(?ms)def\s+best_effort_ext\s*\([^)]*\)\s*:\s*.*?\n(?=\s*def|\Z)",
            (
                "def best_effort_ext() -> str:\n"
                '    """Return a supported extension for writing tabular data.\n'
                "    We default to CSV because it's universally supported.\n"
                '    """\n'
                '    return ".csv"\n\n'
            ),
            t,
            count=1,
        )

    if t != original:
        write_text(p, t)
        print(f"✅ Patched best_effort_ext -> '.csv' in {p.relative_to(ROOT)}")
    else:
        print(f"ℹ️ best_effort_ext already ok in {p.relative_to(ROOT)}")


# ----------------------------
# 2) Patch run_train signature
# ----------------------------
def patch_run_train_cfg() -> None:
    """
    Make run_train accept cfg=None.
    If cfg is None: cfg = TrainConfig()
    Also remove any line inside the function that overwrites cfg unconditionally.
    """
    run_train_files = []
    for p in py_files():
        if re.search(r"(?m)^\s*def\s+run_train\s*\(", read_text(p)):
            run_train_files.append(p)

    if not run_train_files:
        die("❌ Can't find def run_train(...) inside src/ml_baseline")

    # Prefer train.py if present
    p = next(
        (x for x in run_train_files if x.name in {"train.py", "training.py"}),
        run_train_files[0],
    )
    lines = read_text(p).splitlines(True)
    original = "".join(lines)

    # Find def line
    def_i = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*def\s+run_train\s*\(", ln):
            def_i = i
            break
    if def_i is None:
        die(f"❌ Unexpected: couldn't locate run_train line in {p}")

    # Determine indent
    def_indent = re.match(r"^(\s*)", lines[def_i]).group(1)
    body_indent = def_indent + "    "

    # 2.1) Signature -> def run_train(cfg=None):
    lines[def_i] = re.sub(
        r"^(\s*)def\s+run_train\s*\([^)]*\)\s*:",
        r"\1def run_train(cfg=None):",
        lines[def_i],
    )

    # 2.2) Remove lines that overwrite cfg directly
    for j in range(def_i + 1, len(lines)):
        ln = lines[j]
        # stop if we leave the function block (indent back)
        if (
            ln.strip()
            and (len(re.match(r"^(\s*)", ln).group(1)) <= len(def_indent))
            and not ln.startswith(body_indent)
        ):
            break

        stripped = ln.strip().replace(" ", "")
        if stripped.startswith("cfg=TrainConfig(") or stripped == "cfg=TrainConfig()":
            lines[j] = body_indent + "# cfg comes from CLI (do not overwrite)\n"

    # 2.3) Ensure we have:
    # if cfg is None:
    #     cfg = TrainConfig()
    # Insert it early in the function (after possible docstring).
    # Find insertion point: after docstring if present, otherwise right after def line.
    insert_at = def_i + 1

    # Skip blank lines
    while insert_at < len(lines) and lines[insert_at].strip() == "":
        insert_at += 1

    # Skip docstring block if immediately present
    if insert_at < len(lines) and re.match(r'^\s*("""|\'\'\')', lines[insert_at]):
        quote = '"""' if '"""' in lines[insert_at] else "'''"
        insert_at += 1
        while insert_at < len(lines) and quote not in lines[insert_at]:
            insert_at += 1
        if insert_at < len(lines):
            insert_at += 1  # include closing quote line

    # Only insert if not already present
    joined = "".join(lines[def_i : def_i + 80])
    if "if cfg is None" not in joined:
        block = (
            body_indent
            + "# Backward compatible: allow calling run_train() with no cfg\n"
            + body_indent
            + "if cfg is None:\n"
            + body_indent
            + "    cfg = TrainConfig()\n\n"
        )
        lines.insert(insert_at, block)

    new = "".join(lines)
    if new != original:
        write_text(p, new)
        print(f"✅ Patched run_train(cfg=None) in {p.relative_to(ROOT)}")
    else:
        print(f"ℹ️ run_train already ok in {p.relative_to(ROOT)}")


# ----------------------------
# 3) Patch cli.py fallback
# ----------------------------
def patch_cli_fallback() -> None:
    """
    Remove try/except fallback that calls run_train() without cfg.
    It hides the real bug and can cause training to use defaults unexpectedly.
    """
    p = ROOT / "src" / "ml_baseline" / "cli.py"
    if not p.exists():
        die(f"❌ Missing: {p}")

    t = read_text(p)
    original = t

    # Convert:
    # try:
    #   run_dir = run_train(cfg)
    # except TypeError:
    #   run_dir = run_train()
    # to:
    # run_dir = run_train(cfg)
    t = re.sub(
        r"(?ms)^(\s*)try:\s*\n"
        r"\1\s+run_dir\s*=\s*run_train\(cfg\)\s*\n"
        r"\1except\s+TypeError:\s*\n"
        r"(?:.*\n)*?"
        r"\1\s+run_dir\s*=\s*run_train\(\)\s*\n",
        r"\1run_dir = run_train(cfg)\n",
        t,
        count=1,
    )

    if t != original:
        write_text(p, t)
        print("✅ Removed run_train() fallback in cli.py")
    else:
        print("ℹ️ No fallback block found in cli.py (ok)")


def main() -> None:
    print("=== Fixing Week 3 repo ===")
    patch_sample_data_ext()
    patch_run_train_cfg()
    patch_cli_fallback()
    print("✅ Fix script finished.")


if __name__ == "__main__":
    main()
