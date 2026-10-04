"""DataLens — Fast, streaming map-reduce statistics for tabular datasets."""

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
__version__ = "0.1.0"

