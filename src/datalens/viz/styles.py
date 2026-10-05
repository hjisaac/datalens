"""Publication-grade visual styling and themes for DataLens plots."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

DATALENS_PALETTE = [
    "#2B5C8F",  # Deep navy / DataLens primary
    "#2A9D8F",  # Teal / emerald
    "#E76F51",  # Warm coral
    "#E9C46A",  # Gold / amber
    "#457B9D",  # Steel blue
    "#F4A261",  # Sand / peach
    "#9B5DE5",  # Purple
    "#00BBF9",  # Cyan
]

PAPER_PALETTE = [
    "#1A365D",  # Dark academic navy
    "#742A2A",  # Burgundy
    "#234E52",  # Deep pine
    "#7B341E",  # Rust
    "#4A5568",  # Slate
    "#2D3748",  # Charcoal
]

DARK_PALETTE = [
    "#38BDF8",  # Sky blue
    "#34D399",  # Mint green
    "#F472B6",  # Pink
    "#FBBF24",  # Amber
    "#A78BFA",  # Violet
    "#60A5FA",  # Blue
]

PRESETS: dict[str, dict[str, Any]] = {
    "datalens": {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Inter", "Helvetica Neue", "Arial", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "normal",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#E2E8F0",
    },
    "paper": {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#000000",
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman", "serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.labelsize": 10,
        "axes.labelweight": "normal",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#000000",
        "mathtext.fontset": "cm",
    },
    "dark": {
        "figure.facecolor": "#0F172A",
        "axes.facecolor": "#1E293B",
        "axes.edgecolor": "#475569",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#334155",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "text.color": "#F8FAFC",
        "axes.labelcolor": "#F8FAFC",
        "xtick.color": "#CBD5E1",
        "ytick.color": "#CBD5E1",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "legend.fontsize": 9,
        "legend.facecolor": "#1E293B",
        "legend.edgecolor": "#475569",
    },
}


def get_palette(style_name: str = "datalens") -> list[str]:
    """Return the color palette for a given style preset."""
    match style_name.lower():
        case "paper" | "academic":
            return PAPER_PALETTE
        case "dark":
            return DARK_PALETTE
        case _:
            return DATALENS_PALETTE


def get_style_rc(style: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    """Resolve a style parameter into an rcParams dictionary."""
    if style is None:
        return dict(PRESETS["datalens"])
    if isinstance(style, dict):
        return dict(style)
    if isinstance(style, (str, Path)) and Path(style).is_file():
        import matplotlib.pyplot as plt
        return dict(plt.rc_params_from_file(str(style), use_default_template=False))
    key = str(style).lower()
    return dict(PRESETS.get(key, PRESETS["datalens"]))


@contextlib.contextmanager
def apply_style(style: str | Path | dict[str, Any] | None = None) -> Iterator[None]:
    """Context manager to cleanly apply style parameters without permanent global mutation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rc_dict = get_style_rc(style)
    with plt.rc_context(rc_dict):
        yield
