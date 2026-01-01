from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Paths
from .io import best_effort_ext, write_tabular


def make_sample_feature_table(
    *, root: Path | None = None, n_users: int = 50, seed: int = 42
) -> Path:
    paths = Paths.from_repo_root() if root is None else Paths(root=root)
    paths.data_processed_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    user_id = [f"u{i:03d}" for i in range(1, n_users + 1)]
    country = rng.choice(["US", "CA", "GB"], size=n_users, replace=True)
    n_orders = rng.integers(1, 10, size=n_users)
    avg_amount = rng.normal(loc=10, scale=3, size=n_users).clip(min=1)
    total_amount = n_orders * avg_amount
    # simple binary target (demo only)
    is_high_value = (total_amount >= 80).astype(int)
    df = pd.DataFrame(
        {
            "user_id": user_id,
            "country": country,
            "n_orders": n_orders,
            "avg_amount": avg_amount.round(2),
            "total_amount": total_amount.round(2),
            "is_high_value": is_high_value,
        }
    )

    out_path = paths.data_processed_dir / "features.csv"
    write_tabular(df, out_path)
    return out_path
