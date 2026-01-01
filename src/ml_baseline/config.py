from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Any

PathLike = Union[str, Path]


def _find_repo_root(start: Path) -> Path:
    """
    Find repo root by walking up until we see pyproject.toml or .git.
    Fallback to the highest parent if not found.
    """
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start.parents[-1] if start.parents else start


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def from_repo_root(cls, start: Optional[PathLike] = None) -> "Paths":
        """
        Create Paths by auto-detecting the repo root.
        """
        start_path = Path(start) if start is not None else Path(__file__)
        root = _find_repo_root(start_path)
        return cls(root=root)

    # Common dirs (project expects these names)
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def data_processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def data_raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def runs_dir(self) -> Path:
        return self.models_dir / "runs"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"


# ✅ نخليها "مرنة" لأن ملفات train/predict عندك ممكن تتوقع حقول مختلفة
class TrainConfig:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)


class PredictConfig:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)
