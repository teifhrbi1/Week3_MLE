from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float | None]:
    """Core binary-classification metrics.

    Note: ROC AUC is undefined if only one class is present in y_true.
    """
    y_pred = (y_score >= threshold).astype(int)

    try:
        roc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc = None

    try:
        pr = float(average_precision_score(y_true, y_score))
    except Exception:
        pr = None

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def choose_threshold_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall arrays are len(thresholds)+1
    f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(np.argmax(f1s))
    return float(thresholds[best])


def bootstrap_ci(
    y_true: np.ndarray,
    y_score_or_pred: np.ndarray,
    metric_fn,
    *,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, float | None]:
    """Simple bootstrap CI.

    Some metrics (like ROC AUC) are undefined on resamples with only one class.
    We skip failed resamples and compute quantiles on the remaining ones.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = np.arange(n)

    # Estimate on the full sample
    try:
        estimate = float(metric_fn(y_true, y_score_or_pred))
    except Exception:
        estimate = None

    stats: list[float] = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        try:
            v = float(metric_fn(y_true[b], y_score_or_pred[b]))
        except Exception:
            continue
        if np.isfinite(v):
            stats.append(v)

    if len(stats) < 10:
        return {"estimate": estimate, "ci_low": None, "ci_high": None}

    arr = np.asarray(stats, dtype=float)
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }
