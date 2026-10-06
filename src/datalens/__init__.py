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
    QuantileAccumulator,
    Task,
    build_config,
    discover_tasks,
    load_config,
    resolve_tasks,
    run_analysis,
)
from .viz import (
    apply_style,
    generate_plots,
    get_palette,
    get_style_rc,
    plot_beeswarm_box,
    plot_categorical_frequency,
    plot_numeric_summary,
    plot_partition_comparison,
    plot_quantile_distribution,
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
    "QuantileAccumulator",
    "Task",
    "apply_style",
    "build_config",
    "discover_tasks",
    "generate_plots",
    "get_palette",
    "get_style_rc",
    "load_config",
    "plot_beeswarm_box",
    "plot_categorical_frequency",
    "plot_numeric_summary",
    "plot_partition_comparison",
    "plot_quantile_distribution",
    "resolve_tasks",
    "run_analysis",
]
__version__ = "0.1.0"

