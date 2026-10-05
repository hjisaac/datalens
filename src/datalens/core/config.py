"""Job configuration for a map-reduce statistics run.

Supports:
- Direct Python instantiation via :class:`AnalysisConfig`
- Construction from dicts via :func:`build_config`
- Loading from YAML via :func:`load_config` or :meth:`AnalysisConfig.from_yaml`
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .registry import known_kinds

_DEFAULTS: dict[str, Any] = {
    "root": None,
    "file_pattern": "*.parquet",
    "batch_size": 65_536,
    "workers": None,
    "partition_depth": None,
    "include_row_count": True,
    "top_categories": None,
    "readers": {},
    "quantiles": None,
    "quantile_compression": 100.0,
    "plots": False,
    "plot_dir": "plots",
    "plot_format": "png",
    "plot_captions": True,
    "plot_style": "datalens",
}


@dataclass
class AnalysisConfig:
    """Strongly-typed job configuration for dataset statistics.

    Usable directly as a Python class or loaded from a YAML configuration.
    Supports both attribute access (``config.batch_size``) and item access (``config["batch_size"]``).
    """

    columns: dict[str, str] = field(default_factory=dict)
    root: str | Path | None = None
    file_pattern: str = "*.parquet"
    batch_size: int = 65_536
    workers: int | None = None
    partition_depth: int | None = None
    include_row_count: bool = True
    top_categories: int | None = None
    readers: dict[str, str] = field(default_factory=dict)
    quantiles: list[float] | None = None
    quantile_compression: float = 100.0
    plots: bool = False
    plot_dir: str | Path = "plots"
    plot_format: str = "png"
    plot_captions: bool = True
    plot_style: str = "datalens"

    def __post_init__(self) -> None:
        if self.root is not None:
            self.root = Path(self.root)
        self.columns = dict(self.columns)
        self.readers = dict(self.readers)
        if self.quantiles is not None:
            self.quantiles = [float(q) for q in self.quantiles]
            for q in self.quantiles:
                if not (0.0 <= q <= 1.0):
                    raise ValueError(f"Quantile values must be between 0.0 and 1.0 inclusive, got {q}")
        self._validate()

    def _validate(self) -> None:
        if not self.columns:
            raise ValueError("At least one column → accumulator mapping is required.")
        unknown = set(self.columns.values()) - known_kinds()
        if unknown:
            known = ", ".join(sorted(known_kinds()))
            raise ValueError(f"Unknown accumulator kind(s): {sorted(unknown)}. Known: {known}")

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AnalysisConfig:
        """Load configuration from a YAML file."""
        return load_config(path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnalysisConfig:
        """Create configuration from a dictionary."""
        return build_config(data)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary of the configuration."""
        return config_summary(self)



def build_config(data: Mapping[str, Any] | AnalysisConfig | DictConfig) -> AnalysisConfig:
    """Layer ``data`` over defaults, validate, and return an :class:`AnalysisConfig`."""
    if isinstance(data, AnalysisConfig):
        return data

    if isinstance(data, DictConfig):
        raw = OmegaConf.to_container(data, resolve=True)
    else:
        raw = dict(data)

    merged = dict(_DEFAULTS)
    merged.update(raw)

    return AnalysisConfig(
        columns=merged.get("columns", {}),
        root=merged.get("root"),
        file_pattern=merged.get("file_pattern", "*.parquet"),
        batch_size=merged.get("batch_size", 65_536),
        workers=merged.get("workers"),
        partition_depth=merged.get("partition_depth"),
        include_row_count=merged.get("include_row_count", True),
        top_categories=merged.get("top_categories"),
        readers=merged.get("readers", {}),
    )


def load_config(path: str | Path) -> AnalysisConfig:
    """Load, validate, and return an :class:`AnalysisConfig` from a YAML file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    loaded = OmegaConf.load(path)
    return build_config(loaded)


def resolved_workers(config: AnalysisConfig | DictConfig) -> int:
    """Return the concrete worker count for ``config``."""
    workers = config.workers if hasattr(config, "workers") else config.get("workers")
    if workers is not None:
        return max(1, workers)
    return max(1, os.cpu_count() or 1)


def config_summary(config: AnalysisConfig | DictConfig) -> dict[str, Any]:
    """JSON-friendly echo of the configuration."""
    root = getattr(config, "root", None)
    return {
        "root": str(root) if root is not None else None,
        "file_pattern": getattr(config, "file_pattern", "*.parquet"),
        "columns": dict(getattr(config, "columns", {})),
        "batch_size": getattr(config, "batch_size", 65_536),
        "workers": resolved_workers(config),
        "partition_depth": getattr(config, "partition_depth", None),
        "include_row_count": getattr(config, "include_row_count", True),
        "top_categories": getattr(config, "top_categories", None),
        "readers": dict(getattr(config, "readers", {})),
    }


CONFIG: AnalysisConfig | None = None


def boot_config(config: AnalysisConfig) -> None:
    """Boot this process's config for the job about to run."""
    global CONFIG
    CONFIG = config
