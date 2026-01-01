from pathlib import Path
import re

p = Path("src/ml_baseline/cli.py")
txt = p.read_text(encoding="utf-8")

if re.search(r"^\s*def\s+main\s*\(", txt, flags=re.M):
    print("ℹ️ main() already exists")
    raise SystemExit(0)

m = re.search(r"^\s*(\w+)\s*=\s*typer\.Typer\s*\(", txt, flags=re.M)
app_var = m.group(1) if m else "app"

txt = (
    txt.rstrip()
    + f"""

def main():
    \"\"\"Console entrypoint for `ml-baseline`.\"\"\"
    {app_var}()

if __name__ == "__main__":
    main()
"""
)
p.write_text(txt, encoding="utf-8")
print(f"✅ Added main() calling {app_var}()")
