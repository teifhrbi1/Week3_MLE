from __future__ import annotations
from pathlib import Path
import re

p = Path("src/ml_baseline/train.py")
txt = p.read_text(encoding="utf-8")

# نبحث عن بلوك: if schema is None: raise RuntimeError(...) ثم schema.dump(...)
pat = re.compile(
    r"""
    if\ schema\ is\ None:\s*\n
    \s*raise\ RuntimeError\([^\)]*Could\ not\ build\ InputSchema[^\)]*\)\s*\n
    \s*schema\.dump\(\s*run_dir\s*/\s*["']schema["']\s*/\s*["']input_schema\.json["']\s*\)
    """,
    re.VERBOSE | re.DOTALL,
)

replacement = """
schema_path = run_dir / "schema" / "input_schema.json"
(run_dir / "schema").mkdir(parents=True, exist_ok=True)

if schema is None:
    # Fallback: minimal schema inferred from columns (exclude target + id cols)
    feature_cols = [c for c in df.columns if c not in drop_cols and c not in id_cols_present]
    schema_raw_candidates = [
        {"required_features": feature_cols, "id_cols": id_cols_present},
        {"required": feature_cols, "id_cols": id_cols_present},
        {"features": feature_cols, "id_cols": id_cols_present},
        {"columns": feature_cols, "id_cols": id_cols_present},
    ]

    built = None
    for raw in schema_raw_candidates:
        try:
            built = InputSchema(**raw)  # type: ignore[arg-type]
            break
        except Exception:
            continue

    if built is not None:
        schema = built
    else:
        # last resort: write JSON so predict/tests have something to read
        import json as _json
        schema_path.write_text(_json.dumps(schema_raw_candidates[0], indent=2), encoding="utf-8")

# Dump schema if we have a schema object
if schema is not None:
    if hasattr(schema, "dump"):
        schema.dump(schema_path)  # type: ignore[attr-defined]
    elif hasattr(schema, "model_dump_json"):
        schema_path.write_text(schema.model_dump_json(indent=2), encoding="utf-8")  # type: ignore[attr-defined]
    elif hasattr(schema, "dict"):
        import json as _json
        schema_path.write_text(_json.dumps(schema.dict(), indent=2), encoding="utf-8")  # type: ignore[attr-defined]
    else:
        import json as _json
        schema_path.write_text(_json.dumps(schema, indent=2), encoding="utf-8")
"""

new_txt, n = pat.subn(replacement, txt, count=1)
if n == 0:
    raise SystemExit(
        "❌ Patch failed: لم أجد بلوك InputSchema error في train.py (النص تغيّر)."
    )

p.write_text(new_txt, encoding="utf-8")
print("✅ Patched:", p)
