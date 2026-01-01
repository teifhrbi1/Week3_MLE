# Model Card — Week 3 Baseline

## Problem
- Target: `target`
- Unit of analysis: one row per entity identified by ['id']
- Decision enabled: Predict `target` to support downstream routing/prioritization decisions.

## Data
- Feature table: `data/processed/sample_features.csv`
- Dataset hash (sha256): `e26003482c9d25510fa51048a9efb862791117daf6ed8c1f8a9fbc6c3f4b7545`
- Required features: `['feature_1', 'feature_2', 'is_high_value']`
- ID columns (optional at inference): `['id']`
- Forbidden columns at inference: `['target']`

## Splits
- Holdout: random stratified, test_size=0.2, seed=42

## Metrics (holdout)
- Baseline: N/A (not recorded)
- Model: model_type=LogisticRegression, seed=42, target=target, task=classification, test_size=0.2000

## Limitations
- This is a baseline model; performance may be limited vs. tuned models.
- Metrics may not generalize if data distribution shifts (feature drift / label drift).
- Threshold selection and calibration are not fully optimized (monitor precision/recall trade-offs).

## Monitoring sketch
- Data quality checks: missingness, schema changes, out-of-range values, categorical drift.
- Performance checks (if labels become available): track accuracy/F1 (or PR-AUC), plus precision/recall by segment.
- Alerting: trigger alerts on drift thresholds or sustained metric drops; log predictions + input stats.

## Reproducibility
- Run id: `2026-01-01T15-24-24Z__classification__seed42`
- Git commit: `cbf151f1368fd1239a60c7d56f33a00a21dbdc37`
- Env: `N/A (pip_freeze.txt not found)`
- Artifacts: `models/runs/2026-01-01T15-24-24Z__classification__seed42`
