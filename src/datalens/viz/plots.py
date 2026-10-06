"""High-fidelity, publication-ready statistical plotting for DataLens."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .styles import apply_style, get_palette


def _ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def plot_quantile_distribution(
    column: str,
    stats: dict[str, Any],
    out_path: Path | str,
    sla: float | None = None,
    captions: bool = True,
    style: str = "datalens",
    dpi: int = 300,
) -> Path:
    """Generate a dual-panel box & percentile fan + CDF curve from QuantileAccumulator stats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_file = _ensure_dir(out_path)
    palette = get_palette(style)
    primary = palette[0]
    secondary = palette[1] if len(palette) > 1 else "#2A9D8F"
    accent = palette[2] if len(palette) > 2 else "#E76F51"

    count = stats.get("count", 0)
    p01 = stats.get("p01")
    p05 = stats.get("p05")
    p25 = stats.get("p25")
    p50 = stats.get("p50")
    p75 = stats.get("p75")
    p90 = stats.get("p90")
    p95 = stats.get("p95")
    p99 = stats.get("p99")
    min_v = stats.get("min")
    max_v = stats.get("max")
    iqr = stats.get("iqr")
    effective_sla = sla if sla is not None else stats.get("sla")

    with apply_style(style):
        fig, (ax_box, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4.5))

        # Panel 1: Percentile Box & Fan Plot
        if p50 is not None and p25 is not None and p75 is not None:
            # Box for IQR (P25 to P75)
            box_left = p25
            box_width = p75 - p25
            rect = plt.Rectangle((box_left, 0.35), box_width, 0.3, facecolor=primary, alpha=0.35, edgecolor=primary, linewidth=1.5)
            ax_box.add_patch(rect)

            # Outer fan range (P01 to P99 or P05 to P95)
            low_whisker = p01 if p01 is not None else (min_v if min_v is not None else p25)
            high_whisker = p99 if p99 is not None else (max_v if max_v is not None else p75)
            ax_box.plot([low_whisker, p25], [0.5, 0.5], color=primary, linestyle="-", linewidth=1.4)
            ax_box.plot([p75, high_whisker], [0.5, 0.5], color=primary, linestyle="-", linewidth=1.4)
            ax_box.plot([low_whisker, low_whisker], [0.42, 0.58], color=primary, linewidth=1.5, label="P01 / P99")
            ax_box.plot([high_whisker, high_whisker], [0.42, 0.58], color=primary, linewidth=1.5)

            # Median line (P50)
            ax_box.plot([p50, p50], [0.35, 0.65], color=accent, linewidth=2.5, label=f"Median ({p50:.3g})")

            # Min and Max markers
            if min_v is not None:
                ax_box.scatter([min_v], [0.5], color=primary, marker="d", s=35, zorder=5, label=f"Min ({min_v:.3g})")
            if max_v is not None:
                ax_box.scatter([max_v], [0.5], color=primary, marker="d", s=35, zorder=5, label=f"Max ({max_v:.3g})")

            # SLA threshold line
            if effective_sla is not None:
                ax_box.axvline(effective_sla, color="#D9534F", linestyle="--", linewidth=1.8, label=f"SLA ({effective_sla:.3g})")

            ax_box.set_ylim(0.1, 0.9)
            ax_box.set_yticks([])
            ax_box.set_xlabel(f"{column} Values")
            if captions:
                ax_box.set_title(f"{column} — Percentile Box & Spread", fontsize=11, fontweight="bold")
                ax_box.legend(loc="upper right", framealpha=0.9, fontsize=8)
            ax_box.spines[["top", "right", "left"]].set_visible(False)

        # Panel 2: Quantile CDF S-Curve
        q_keys = [(0.01, p01), (0.05, p05), (0.25, p25), (0.50, p50), (0.75, p75), (0.90, p90), (0.95, p95), (0.99, p99)]
        valid_points = [(val, q) for q, val in q_keys if val is not None]

        if valid_points:
            x_vals = [p[0] for p in valid_points]
            y_vals = [p[1] for p in valid_points]
            ax_cdf.plot(x_vals, y_vals, marker="o", color=secondary, linewidth=2, markersize=5, label="Empirical CDF")
            if p50 is not None:
                ax_cdf.axhline(0.5, color=accent, linestyle=":", linewidth=1.2, label=f"P50 ({p50:.3g})")
            if effective_sla is not None:
                ax_cdf.axvline(effective_sla, color="#D9534F", linestyle="--", linewidth=1.8, label=f"SLA ({effective_sla:.3g})")
            ax_cdf.set_ylim(-0.02, 1.02)
            ax_cdf.set_xlabel(f"{column} Values")
            ax_cdf.set_ylabel("Cumulative Probability (Quantile)")
            if captions:
                ax_cdf.set_title("Quantile Function (CDF)", fontsize=11, fontweight="bold")
                ax_cdf.legend(loc="lower right", framealpha=0.9, fontsize=8)
            ax_cdf.spines[["top", "right"]].set_visible(False)

        # Statistical callout box (enabled unless captions=False for papers)
        if captions and p50 is not None:
            stat_text = (
                f"Samples: {count:,}\n"
                f"Median: {p50:.3g}\n"
                f"IQR: {iqr:.3g}" if iqr is not None else f"Median: {p50:.3g}\n"
            )
            if p25 is not None and p75 is not None:
                stat_text += f"\nP25: {p25:.3g} | P75: {p75:.3g}"
            if p01 is not None and p99 is not None:
                stat_text += f"\nP01: {p01:.3g} | P99: {p99:.3g}"
            if min_v is not None and max_v is not None:
                stat_text += f"\nMin: {min_v:.3g} | Max: {max_v:.3g}"

            if effective_sla is not None:
                sla_pct = stats.get("sla_exceeded_pct")
                sla_cnt = stats.get("sla_exceeded_count")
                if sla_pct is not None and sla_cnt is not None:
                    stat_text += f"\nSLA: {effective_sla:.3g} | Exceeded: {sla_cnt:,} ({sla_pct:.1f}%)"
                elif sla_pct is not None:
                    stat_text += f"\nSLA: {effective_sla:.3g} | Exceeded: {sla_pct:.1f}%"
                else:
                    stat_text += f"\nSLA Limit: {effective_sla:.3g}"

            ax_box.text(
                0.03, 0.95, stat_text,
                transform=ax_box.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9),
                fontsize=8.5,
            )

        fig.tight_layout()
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_file


def plot_beeswarm_box(
    column: str,
    data: Any,
    out_path: Path | str,
    partition_col: str | None = None,
    sla: float | None = None,
    captions: bool = True,
    style: str = "datalens",
    dpi: int = 300,
) -> Path:
    """Generate a combined Box Plot + Jittered Beeswarm density strip chart.

    Displays distribution quartiles, median, and mean alongside jittered raw sample points
    to visualize local clustering, multimodality, and tail density. Supports an optional
    SLA / threshold limit line.

    Args:
        column: Name of the numeric column being plotted.
        data: Data source. Can be an AnalysisResult, a dictionary of partition samples,
            a statistics dictionary with a 'sample' field, a sequence of floats, or a columnar dict.
        out_path: Output file path (.png, .svg, .pdf).
        partition_col: Optional column name to partition by when data is a tabular mapping.
        sla: Optional SLA or limit threshold. Rendered as a red dashed horizontal reference line.
        captions: If True, adds in-figure title, legend, and summary statistics callout box.
        style: Visual theme ('datalens', 'paper', 'dark').
        dpi: Output resolution.

    Returns:
        Path to the saved figure file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_file = _ensure_dir(out_path)
    palette = get_palette(style)

    groups: dict[str, list[float]] = {}
    effective_sla = sla

    # Unpack data
    if hasattr(data, "to_dict"):
        data = data.to_dict()

    if isinstance(data, dict) and "groups" in data and isinstance(data["groups"], dict):
        for g_name, g_stats in data["groups"].items():
            if isinstance(g_stats, dict) and column in g_stats:
                col_info = g_stats[column]
                if isinstance(col_info, dict):
                    if effective_sla is None and "sla" in col_info:
                        effective_sla = col_info["sla"]
                    if "sample" in col_info and col_info["sample"]:
                        groups[str(g_name)] = [float(x) for x in col_info["sample"]]
    elif isinstance(data, dict) and "sample" in data:
        if effective_sla is None and "sla" in data:
            effective_sla = data["sla"]
        groups["_all"] = [float(x) for x in data["sample"]]
    elif isinstance(data, dict) and any(isinstance(v, (list, tuple, np.ndarray)) for v in data.values()):
        if partition_col is not None and partition_col in data and column in data:
            col_vals = data[column]
            part_vals = data[partition_col]
            for p, v in zip(part_vals, col_vals):
                if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    groups.setdefault(str(p), []).append(float(v))
        else:
            for k, v in data.items():
                if isinstance(v, (list, tuple, np.ndarray)):
                    clean = [float(x) for x in v if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
                    if clean:
                        groups[str(k)] = clean
    elif isinstance(data, (list, tuple, np.ndarray)):
        clean = [float(x) for x in data if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
        if clean:
            groups[column] = clean

    if not groups:
        raise ValueError(f"No sample data found to generate beeswarm plot for column '{column}'.")

    if len(groups) > 1 and "_all" in groups:
        groups = {k: v for k, v in groups.items() if k != "_all"}

    part_names = list(groups.keys())

    with apply_style(style):
        fig_width = max(8.0, len(part_names) * 2.2)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        all_points: list[float] = []

        for idx, (p_name, vals) in enumerate(groups.items()):
            vals_arr = np.asarray(vals, dtype=float)
            vals_arr = vals_arr[np.isfinite(vals_arr)]
            if len(vals_arr) == 0:
                continue
            all_points.extend(vals_arr.tolist())

            c = palette[idx % len(palette)]

            # Jittered strip overlay (beeswarm-style density)
            rng = np.random.default_rng(42 + idx)
            jitter = rng.uniform(-0.18, 0.18, size=len(vals_arr))
            ax.scatter(
                idx + jitter,
                vals_arr,
                color=c,
                alpha=0.40,
                s=24,
                edgecolors="none",
                zorder=3,
            )

            # Box plot metrics
            q25, q50, q75 = np.percentile(vals_arr, [25, 50, 75])
            mean_val = float(np.mean(vals_arr))
            iqr = q75 - q25
            low_w = max(float(np.min(vals_arr)), q25 - 1.5 * iqr)
            high_w = min(float(np.max(vals_arr)), q75 + 1.5 * iqr)

            # Box rectangle
            rect = plt.Rectangle(
                (idx - 0.28, q25),
                0.56,
                max(1e-6, q75 - q25),
                facecolor=c,
                alpha=0.35,
                edgecolor=c,
                linewidth=1.5,
                zorder=2,
            )
            ax.add_patch(rect)

            # Median line
            ax.plot([idx - 0.28, idx + 0.28], [q50, q50], color="#D9534F", linewidth=2.2, zorder=4)

            # Mean marker
            ax.scatter(
                [idx],
                [mean_val],
                marker="o",
                facecolor="white",
                edgecolor="#1E293B",
                s=40,
                linewidth=1.5,
                zorder=5,
                label="Mean" if idx == 0 else None,
            )

            # Whiskers
            ax.plot([idx, idx], [low_w, q25], color=c, linewidth=1.3, zorder=2)
            ax.plot([idx, idx], [q75, high_w], color=c, linewidth=1.3, zorder=2)
            ax.plot([idx - 0.1, idx + 0.1], [low_w, low_w], color=c, linewidth=1.3, zorder=2)
            ax.plot([idx - 0.1, idx + 0.1], [high_w, high_w], color=c, linewidth=1.3, zorder=2)

        # SLA limit line
        if effective_sla is not None:
            ax.axhline(
                effective_sla,
                color="#D9534F",
                linestyle="--",
                linewidth=1.8,
                label=f"SLA Limit ({effective_sla:g})",
                zorder=6,
            )

        ax.set_xticks(range(len(part_names)))
        clean_labels = [p if p != "_all" else column for p in part_names]
        ax.set_xticklabels(
            clean_labels,
            rotation=30 if len(part_names) > 4 else 0,
            ha="right" if len(part_names) > 4 else "center",
        )
        ax.set_ylabel(f"{column} Values")
        ax.spines[["top", "right"]].set_visible(False)

        if captions:
            title_suffix = "by Partition" if len(part_names) > 1 else ""
            ax.set_title(f"{column} — Box & Beeswarm Density {title_suffix}".strip(), fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

            if all_points:
                stat_lines = [f"Sampled Points: {len(all_points):,}"]
                if effective_sla is not None:
                    exc_cnt = int(np.sum(np.asarray(all_points) > effective_sla))
                    exc_pct = (exc_cnt / len(all_points) * 100) if len(all_points) else 0.0
                    stat_lines.append(f"SLA Limit: {effective_sla:g}")
                    stat_lines.append(f"Exceeded (> {effective_sla:g}): {exc_cnt:,} ({exc_pct:.1f}%)")

                ax.text(
                    0.03,
                    0.95,
                    "\n".join(stat_lines),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9),
                    fontsize=8.5,
                )

        fig.tight_layout()
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_file


def plot_categorical_frequency(
    column: str,
    stats: dict[str, Any],
    out_path: Path | str,
    captions: bool = True,
    top: int = 15,
    style: str = "datalens",
    dpi: int = 300,
) -> Path:
    """Generate a horizontal frequency bar chart for CategoricalAccumulator stats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_file = _ensure_dir(out_path)
    palette = get_palette(style)
    bar_color = palette[0]

    unique_count = stats.get("unique", 0)
    raw_counts = stats.get("counts", {})
    # Take top N
    items = list(raw_counts.items())[:top]
    # Reverse for top-down display in horizontal bar
    items.reverse()

    categories = [str(k) for k, _ in items]
    counts = [int(v) for _, v in items]
    total_samples = sum(raw_counts.values()) if raw_counts else 0

    with apply_style(style):
        height = max(3.5, len(categories) * 0.38)
        fig, ax = plt.subplots(figsize=(8, height))

        bars = ax.barh(categories, counts, color=bar_color, height=0.65, edgecolor="none", alpha=0.85)

        # Add data labels at the end of each bar
        max_c = max(counts) if counts else 1
        for bar, count_val in zip(bars, counts):
            pct = (count_val / total_samples * 100) if total_samples > 0 else 0.0
            label = f" {count_val:,} ({pct:.1f}%)" if captions else f" {count_val:,}"
            ax.text(
                bar.get_width() + (max_c * 0.01),
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=8.5,
                color="#334155",
            )

        ax.set_xlim(0, max_c * 1.25)
        ax.set_xlabel("Frequency Count")
        if captions:
            ax.set_title(f"{column} — Category Frequencies", fontsize=11, fontweight="bold")
            # Summary badge
            badge = f"Distinct Classes: {unique_count:,}\nTotal Ingested: {total_samples:,}"
            ax.text(
                0.97, 0.05, badge,
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9),
                fontsize=8.5,
            )

        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_file


def plot_numeric_summary(
    column: str,
    stats: dict[str, Any],
    out_path: Path | str,
    captions: bool = True,
    style: str = "datalens",
    dpi: int = 300,
) -> Path:
    """Generate a summary interval & spread chart for NumericAccumulator stats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_file = _ensure_dir(out_path)
    palette = get_palette(style)
    color = palette[0]

    count = stats.get("count", 0)
    mean = stats.get("mean")
    std = stats.get("std")
    min_v = stats.get("min")
    max_v = stats.get("max")

    with apply_style(style):
        fig, ax = plt.subplots(figsize=(7, 2.8))

        if mean is not None:
            # Min-Max span line
            if min_v is not None and max_v is not None:
                ax.plot([min_v, max_v], [0.5, 0.5], color="#94A3B8", linewidth=1.5, linestyle="--", label="[Min, Max] Range")
                ax.scatter([min_v, max_v], [0.5, 0.5], color="#64748B", marker="|", s=80, zorder=4)

            # Mean ± Std error bar
            if std is not None:
                ax.errorbar(mean, 0.5, xerr=std, fmt="o", color=color, ecolor=color, elinewidth=2.5, capsize=5, markersize=8, label=f"Mean ± Std ({mean:.3g} ± {std:.3g})")
            else:
                ax.scatter([mean], [0.5], color=color, s=60, label=f"Mean ({mean:.3g})")

            ax.set_ylim(0.2, 0.8)
            ax.set_yticks([])
            ax.set_xlabel(f"{column} Values")
            if captions:
                ax.set_title(f"{column} — Mean & Standard Deviation", fontsize=11, fontweight="bold")
                ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
                std_str = f"{std:.3g}" if std is not None else "N/A"
                min_str = f"{min_v:.3g}" if min_v is not None else "N/A"
                max_str = f"{max_v:.3g}" if max_v is not None else "N/A"
                stat_text = f"Count: {count:,}\nMean: {mean:.3g}\nStd: {std_str}\nMin: {min_str}\nMax: {max_str}"
                ax.text(
                    0.03, 0.95, stat_text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9),
                    fontsize=8.5,
                )
            ax.spines[["top", "right", "left"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_file


def plot_partition_comparison(
    column: str,
    groups: dict[str, dict[str, Any]],
    out_path: Path | str,
    kind: str = "quantile",
    sla: float | None = None,
    captions: bool = True,
    style: str = "datalens",
    dpi: int = 300,
) -> Path:
    """Generate partition comparison figures comparing metrics across dataset partitions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_file = _ensure_dir(out_path)
    palette = get_palette(style)

    # Filter out global partition if present
    parts = {k: v for k, v in groups.items() if k != "_all" and column in v}
    if not parts:
        parts = {k: v for k, v in groups.items() if column in v}

    with apply_style(style):
        part_names = list(parts.keys())
        fig, ax = plt.subplots(figsize=(max(8, len(part_names) * 1.5), 5))

        if kind == "quantile":
            # Multi-partition box plot
            positions = list(range(len(part_names)))
            for i, p_name in enumerate(part_names):
                p_stats = parts[p_name][column]
                p25 = p_stats.get("p25")
                p50 = p_stats.get("p50")
                p75 = p_stats.get("p75")
                p01 = p_stats.get("p01")
                p99 = p_stats.get("p99")
                min_v = p_stats.get("min")
                max_v = p_stats.get("max")
                c = palette[i % len(palette)]

                if p50 is not None and p25 is not None and p75 is not None:
                    rect = plt.Rectangle((i - 0.25, p25), 0.5, p75 - p25, facecolor=c, alpha=0.4, edgecolor=c, linewidth=1.5)
                    ax.add_patch(rect)
                    ax.plot([i - 0.25, i + 0.25], [p50, p50], color="#D9534F", linewidth=2.2)
                    low_w = p01 if p01 is not None else min_v
                    high_w = p99 if p99 is not None else max_v
                    if low_w is not None:
                        ax.plot([i, i], [low_w, p25], color=c, linewidth=1.3)
                        ax.plot([i - 0.1, i + 0.1], [low_w, low_w], color=c, linewidth=1.3)
                    if high_w is not None:
                        ax.plot([i, i], [p75, high_w], color=c, linewidth=1.3)
                        ax.plot([i - 0.1, i + 0.1], [high_w, high_w], color=c, linewidth=1.3)

            ax.set_xticks(positions)
            ax.set_xticklabels(part_names, rotation=30 if len(part_names) > 4 else 0, ha="right")
            ax.set_ylabel(f"{column} Values")
            if captions:
                ax.set_title(f"{column} — Partition Comparison (Box & Median)", fontsize=11, fontweight="bold")
        else:
            # Numeric means with std
            means = [parts[p][column].get("mean", 0.0) for p in part_names]
            stds = [parts[p][column].get("std", 0.0) for p in part_names]
            ax.bar(part_names, means, yerr=stds, capsize=4, color=palette[0], alpha=0.8, edgecolor="none")
            ax.set_ylabel(f"{column} Mean")
            if captions:
                ax.set_title(f"{column} — Partition Means Comparison", fontsize=11, fontweight="bold")

        # SLA horizontal reference line
        if sla is not None:
            ax.axhline(sla, color="#D9534F", linestyle="--", linewidth=1.8, label=f"SLA Limit ({sla:g})", zorder=6)
            if captions:
                ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return out_file


def generate_plots(
    analysis_result: Any,
    out_dir: Path | str = "plots",
    format: str = "png",
    sla: float | dict[str, float] | None = None,
    captions: bool = True,
    style: str = "datalens",
    dpi: int = 300,
) -> dict[str, Path]:
    """Batch-generate all statistical plots for an AnalysisResult and return a dictionary of paths."""
    out_directory = Path(out_dir)
    out_directory.mkdir(parents=True, exist_ok=True)

    result_dict = analysis_result.to_dict() if hasattr(analysis_result, "to_dict") else analysis_result
    groups = result_dict.get("groups", {})
    if not groups:
        return {}

    # Primary group for single-group stats
    primary_group_key = "_all" if "_all" in groups else next(iter(groups.keys()))
    primary_stats = groups[primary_group_key]

    created: dict[str, Path] = {}

    for col_name, col_data in primary_stats.items():
        if not isinstance(col_data, dict):
            continue

        col_sla: float | None = None
        if sla is not None:
            if isinstance(sla, dict):
                col_sla = sla.get(col_name)
            elif isinstance(sla, (int, float)):
                col_sla = float(sla)
        elif "sla" in col_data:
            col_sla = col_data["sla"]

        # Check accumulator kind
        if "p50" in col_data:
            # Quantile distribution plot
            fname = f"{col_name}_quantile.{format}"
            path = plot_quantile_distribution(col_name, col_data, out_directory / fname, sla=col_sla, captions=captions, style=style, dpi=dpi)
            created[f"{col_name}_quantile"] = path

            # If sample points are available, generate beeswarm + box plot
            if "sample" in col_data and col_data["sample"]:
                b_fname = f"{col_name}_beeswarm.{format}"
                b_path = plot_beeswarm_box(col_name, analysis_result, out_directory / b_fname, sla=col_sla, captions=captions, style=style, dpi=dpi)
                created[f"{col_name}_beeswarm"] = b_path

        elif "counts" in col_data:
            # Categorical frequency plot
            fname = f"{col_name}_categorical.{format}"
            path = plot_categorical_frequency(col_name, col_data, out_directory / fname, captions=captions, style=style, dpi=dpi)
            created[f"{col_name}_categorical"] = path

        elif "mean" in col_data:
            # Numeric summary plot
            fname = f"{col_name}_numeric.{format}"
            path = plot_numeric_summary(col_name, col_data, out_directory / fname, captions=captions, style=style, dpi=dpi)
            created[f"{col_name}_numeric"] = path

    # If multiple partitions exist, generate partition comparison plots
    non_all_keys = [k for k in groups.keys() if k != "_all"]
    if len(non_all_keys) > 1:
        first_group = groups[non_all_keys[0]]
        for col_name, col_data in first_group.items():
            if isinstance(col_data, dict) and ("p50" in col_data or "mean" in col_data):
                kind = "quantile" if "p50" in col_data else "numeric"
                col_sla = None
                if sla is not None:
                    if isinstance(sla, dict):
                        col_sla = sla.get(col_name)
                    elif isinstance(sla, (int, float)):
                        col_sla = float(sla)
                elif "sla" in col_data:
                    col_sla = col_data["sla"]

                fname = f"{col_name}_partition_comparison.{format}"
                path = plot_partition_comparison(col_name, groups, out_directory / fname, kind=kind, sla=col_sla, captions=captions, style=style, dpi=dpi)
                created[f"{col_name}_partition_comparison"] = path

                # Also generate partition beeswarm if samples exist across partitions
                has_partition_samples = any(
                    isinstance(groups.get(p, {}).get(col_name), dict) and groups[p][col_name].get("sample")
                    for p in non_all_keys
                )
                if has_partition_samples:
                    b_fname = f"{col_name}_partition_beeswarm.{format}"
                    b_path = plot_beeswarm_box(col_name, analysis_result, out_directory / b_fname, sla=col_sla, captions=captions, style=style, dpi=dpi)
                    created[f"{col_name}_partition_beeswarm"] = b_path

    return created
