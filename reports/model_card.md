# Model Card — Week 3 (Draft)

## 1) What is the prediction?
- **Target (y):** `is_high_value`
- **Target meaning:** binary label indicating whether a user is classified as high-value (`1`) or not (`0`).
- **Unit of analysis:** one row = one **user** (aggregated user-level features)

## 2) Data contract (inference)
- **ID passthrough columns:** `user_id`
- **Required feature columns (X):** `country`, `n_orders`, `avg_amount`, `total_amount`
- **Forbidden columns:** `is_high_value` (target) + any label/leakage columns (e.g., `target`, `label`, post-outcome or future-derived fields)

## 3) Notes (draft)
- **Assumptions:** features are pre-aggregated per user and available at prediction time.
- **Data quality checks:** no missing `user_id`; valid numeric ranges for `n_orders`, `avg_amount`, `total_amount`.
