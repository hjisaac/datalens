"""Dataset statistics — parallel map-reduce statistics for tabular datasets."""

from __future__ import annotations

from .core import (
    Accumulator,
    AnalysisConfig,
    AnalysisResult,
    CardinalityAccumulator,
    CategoricalAccumulator,
    CountAccumulator,
    DataTask,
    FileTask,
    NumericAccumulator,
    PartialStats,
    StatsConfig,
    Task,
    build_config,
    discover_tasks,
    load_config,
    resolve_tasks,
    run_analysis,
)

__all__ = [
    "Accumulator",
    "AnalysisConfig",
    "AnalysisResult",
    "CardinalityAccumulator",
    "CategoricalAccumulator",
    "CountAccumulator",
    "DataTask",
    "FileTask",
    "NumericAccumulator",
    "PartialStats",
    "StatsConfig",
    "Task",
    "build_config",
    "discover_tasks",
    "load_config",
    "resolve_tasks",
    "run_analysis",
]


def __getattr__(name: str):
    """Lazy legacy exports for the pre-refactor InstaNovo-style API."""
    legacy_names = {
        "CatalogEntry",
        "FieldMap",
        "IdentityConfig",
        "MAPPABLE_SCALAR_FIELDS",
        "MetricToggles",
        "StatisticsResult",
        "field_map_from_pairs",
        "load_catalog",
        "run_dataset_statistics",
    }
    if name not in legacy_names:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _load_legacy(name)


def _load_legacy(name: str):
    from catalog import CatalogEntry, load_catalog
    from config import (
        MAPPABLE_SCALAR_FIELDS,
        FieldMap,
        MetricToggles,
        field_map_from_pairs,
    )
    from identity import IdentityConfig
    from runner import StatisticsResult, discover_tasks as legacy_discover_tasks, run_dataset_statistics

    mapping = {
        "CatalogEntry": CatalogEntry,
        "FieldMap": FieldMap,
        "IdentityConfig": IdentityConfig,
        "MAPPABLE_SCALAR_FIELDS": MAPPABLE_SCALAR_FIELDS,
        "MetricToggles": MetricToggles,
        "StatisticsResult": StatisticsResult,
        "field_map_from_pairs": field_map_from_pairs,
        "load_catalog": load_catalog,
        "run_dataset_statistics": run_dataset_statistics,
        "discover_tasks": legacy_discover_tasks,
    }
    return mapping[name]
