"""Map-reduce statistics core."""

from .accumulators import (
    CardinalityAccumulator,
    CategoricalAccumulator,
    CountAccumulator,
    NumericAccumulator,
    QuantileAccumulator,
)
from .config import (
    AnalysisConfig,
    boot_config,
    build_config,
    config_summary,
    load_config,
    resolved_workers,
)
from .discovery import discover_tasks, resolve_tasks
from .mapper import fold_batch, map_task
from .orchestrator import run_analysis
from .reducer import reduce_group, render_partial
from .registry import Accumulator, build_partial_stats, create_accumulator, known_kinds
from .shuffler import shuffle
from .types import AnalysisResult, DataTask, FileTask, PartialStats, Task

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
    "QuantileAccumulator",
    "Task",
    "boot_config",
    "build_config",
    "build_partial_stats",
    "config_summary",
    "create_accumulator",
    "discover_tasks",
    "fold_batch",
    "known_kinds",
    "load_config",
    "map_task",
    "reduce_group",
    "render_partial",
    "resolve_tasks",
    "resolved_workers",
    "run_analysis",
    "shuffle",
]
