
## Day-by-day summary (what we did + what “done” looks like)

### Day 1 — Data contract + CLI skeleton
**Summary:** Defined the ML problem (**unit of analysis**, **target**, **IDs**, **leakage**) and built a runnable CLI skeleton.

**Done when:**
- `uv run ml-baseline --help` works
- tests pass: `uv run pytest -q`
- `reports/model_card.md` exists with a draft spec (UoA + target + primary metric)
- *(optional)* `make-sample-data` generates `data/processed/features.csv`

---

### Day 2 — Train + baseline
**Summary:** Added training that creates a **versioned run folder** and writes baseline artifacts (dummy model) + model artifact.

**Done when:**
- `uv run ml-baseline train --target <target>` runs
- `models/registry/latest.txt` updates
- run folder contains: `run_meta.json`, schema, env snapshot, model, baseline metrics

---

### Day 3 — Evaluate + artifacts
**Summary:** Produced credible holdout evaluation and saved the artifacts needed for shipping/debugging.

**Done when:**
- `metrics/holdout_metrics.json` exists (plus baseline metrics)
- `tables/holdout_predictions.*` exists (includes score/prediction + true target for analysis)
- `tables/holdout_input.*` exists (features-only, **no target**)

**Note:** ROC-AUC can be undefined if holdout contains only one class (expected warning in that case).

---

### Day 4 — Predict CLI + skew check
**Summary:** Implemented reliable batch inference using schema guardrails + dtype normalization, and verified training-serving alignment (“skew check”).

**Done when:**
- `uv run ml-baseline predict --run latest --input ... --output ...` works
- failures are helpful for:
  - forbidden columns (target present)
  - missing required columns
- skew check passes by predicting on `tables/holdout_input.*` with matching row counts

---

### Day 5 — README + final polish (submission-ready)
**Summary:** Polished documentation + ensured repo is teammate-runnable and quality gates pass.

**Done when:**
- README commands are copy/paste runnable
- `reports/model_card.md` and `reports/eval_summary.md` are filled
- `pytest` + `ruff` checks pass

---

## How to run the whole project (end-to-end)

> Run these from the repo root (where `pyproject.toml` is).

### 1) Setup + sanity
```bash
uv sync
uv run ml-baseline --help
uv run pytest -q
