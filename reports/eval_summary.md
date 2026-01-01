# Evaluation Summary — Week 3 Baseline

## What you trained
- Task: classification
- Target: `target`
- Feature table used in training: `data/processed/sample_features.csv`
- Run id: `2026-01-01T15-24-24Z__classification__seed42`

## Results (holdout)
- Baseline: N/A
- Model:

## Error analysis
- Review false positives/negatives by saving holdout predictions (y_true, y_pred, proba, ids).
- Leakage check: confirm target column removed from inference input and no post-outcome features.
- Slice errors by segments (e.g., country, n_orders bins) to identify weak areas.

## Recommendation
- Recommendation: **DON’T SHIP YET** unless baseline is present and model clearly improves holdout metric.
- Threshold used: 0.5
