from pathlib import Path
from typing import Union, Optional

import pandas as pd

PathLike = Union[str, Path]


def best_effort_ext(p: Optional[PathLike] = None) -> str:
    """
    Return lowercase file extension (without the dot).
    If p is None -> "".
    """
    if p is None:
        return ""
    return Path(str(p)).suffix.lower().lstrip(".")


def read_tabular(path: PathLike) -> pd.DataFrame:
    p = Path(path)
    ext = best_effort_ext(p)
    if ext == "csv":
        return pd.read_csv(p)
    if ext in ("parquet", "pq"):
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported tabular format: {p} (ext='{ext}')")


def write_tabular(df: pd.DataFrame, path: PathLike, index: bool = False) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = best_effort_ext(p)

    if ext == "csv":
        df.to_csv(p, index=index)
        return p
    if ext in ("parquet", "pq"):
        df.to_parquet(p, index=index)
        return p

    raise ValueError(f"Unsupported tabular format: {p} (ext='{ext}')")
