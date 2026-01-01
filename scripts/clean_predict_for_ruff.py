from __future__ import annotations
from pathlib import Path

p = Path("src/ml_baseline/predict.py")
lines = p.read_text(encoding="utf-8").splitlines(True)

# 1) لو فيه بلوك "Compatibility wrapper" نقصّه (غالبًا هو سبب التكرار)
out = []
cut = False
for line in lines:
    if line.strip().startswith("# --- Compatibility wrapper"):
        cut = True
        break
    out.append(line)

lines = out if cut else lines

# 2) احذف import مكرر في وسط الملف (نخلي أول import فقط)
seen_import = set()
clean = []
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped in {
        "from pathlib import Path",
        "from typing import Optional, Union",
        "PathLike = Union[str, Path]",
    }:
        if stripped in seen_import:
            continue
        seen_import.add(stripped)
    clean.append(line)
lines = clean

# 3) لو resolve_run_dir مكرر، خلّي آخر واحد فقط
def_idxs = [i for i, line in enumerate(lines) if line.startswith("def resolve_run_dir")]
if len(def_idxs) > 1:
    keep_start = def_idxs[-1]
    # احذف كل شيء من أول تعريف إلى قبل آخر تعريف
    first_start = def_idxs[0]
    del lines[first_start:keep_start]

p.write_text("".join(lines), encoding="utf-8")
print("✅ Cleaned:", p)
