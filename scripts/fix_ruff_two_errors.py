from __future__ import annotations
from pathlib import Path
import re


def drop_unused_import(path: str, pattern: str) -> None:
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    new = re.sub(pattern, "", txt, flags=re.MULTILINE)
    if new != txt:
        p.write_text(new, encoding="utf-8")
        print(f"✅ patched {path}")
    else:
        print(f"ℹ️ no change {path}")


# (A) src/ml_baseline/cli.py : remove unused `import json`
drop_unused_import("src/ml_baseline/cli.py", r"^import\s+json\s*\n")

# (B) src/ml_baseline/sample_data.py : remove unused best_effort_ext from import line
p = Path("src/ml_baseline/sample_data.py")
if p.exists():
    txt = p.read_text(encoding="utf-8")
    txt2 = txt.replace(
        "from .io import best_effort_ext, write_tabular\n",
        "from .io import write_tabular\n",
    )
    if txt2 != txt:
        p.write_text(txt2, encoding="utf-8")
        print("✅ patched src/ml_baseline/sample_data.py")
    else:
        print("ℹ️ no change src/ml_baseline/sample_data.py")
else:
    print("ℹ️ src/ml_baseline/sample_data.py not found")
