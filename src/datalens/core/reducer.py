"""Reduce phase: merge partial statistics and render final summaries."""

from __future__ import annotations

from . import config as config_module
from .registry import build_partial_stats
from .types import PartialStats


def reduce_group(key: tuple[str, ...], partials: list[PartialStats]) -> dict:
    """Merge all partials for one partition key and render the result dict."""
    merged = build_partial_stats(config_module.CONFIG.columns)
    for partial in partials:
        merged.merge(partial)
    return render_partial(key, merged)


def render_partial(key: tuple[str, ...], partial: PartialStats) -> dict:
    """Render a merged partial to a JSON-friendly dictionary, using the booted config."""
    config = config_module.CONFIG
    output: dict = {"partition": list(key)}
    if config.include_row_count:
        output["rows"] = partial.row_count
    for column, accumulator in partial.metrics.items():
        kind = config.columns[column]
        if kind == "categorical":
            output[column] = accumulator.result(top=config.top_categories)
        elif kind == "quantile":
            col_sla = config_module.resolve_column_sla(getattr(config, "sla", None), column)
            output[column] = accumulator.result(
                quantiles=getattr(config, "quantiles", None),
                sla=col_sla,
            )
        else:
            output[column] = accumulator.result()
    return output
