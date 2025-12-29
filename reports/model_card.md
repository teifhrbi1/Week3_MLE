# Model Card — Week 3 (Draft)

## 1) What is the prediction?
- **Target (y):** `is_high_value`
- **Unit of analysis:** one row = one user (aggregated features per user)
- **Decision supported:** prioritize high-value users for retention/marketing actions

## 2) Data contract (inference)
- **ID passthrough columns:** `user_id`
- **Required feature columns (X):** `country`, `n_orders`, `avg_amount`, `total_amount`
- **Forbidden columns:** `is_high_value` (target) + any leakage fields that encode the label or future outcomes

## 3) Evaluation plan (fill on Day 2–3)
- **Split strategy:** stratified train/validation split on `is_high_value` (or time-based if timestamps exist)
- **Primary metric:** ROC-AUC (and/or F1 for positive-class focus)
