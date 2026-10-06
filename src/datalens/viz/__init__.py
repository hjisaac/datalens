"""DataLens visualization package — publication-grade statistical charts and dashboards."""

from __future__ import annotations

from .plots import (
    generate_plots,
    plot_beeswarm_box,
    plot_categorical_frequency,
    plot_numeric_summary,
    plot_partition_comparison,
    plot_quantile_distribution,
)
from .styles import apply_style, get_palette, get_style_rc

__all__ = [
    "apply_style",
    "generate_plots",
    "get_palette",
    "get_style_rc",
    "plot_beeswarm_box",
    "plot_categorical_frequency",
    "plot_numeric_summary",
    "plot_partition_comparison",
    "plot_quantile_distribution",
]
