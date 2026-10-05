"""Map phase: read one input split and emit a partition key with partial statistics."""

from __future__ import annotations

import logging
from typing import Any

from . import config as config_module
from .readers import ColumnBatch, numeric_values
from .registry import build_partial_stats
from .types import PartialStats, Task

logger = logging.getLogger(__name__)


def fold_batch(batch: ColumnBatch, partial: PartialStats, config: Any) -> None:
    """Fold a single column batch into running partial statistics."""
    if config.include_row_count:
        partial.row_count += batch.num_rows
    for column, kind in config.columns.items():
        if not batch.has_column(column):
            continue
        values = batch.column(column)
        acc = partial.metrics[column]
        match kind:
            case "numeric" | "quantile":
                acc.update(numeric_values(values))
            case "categorical":
                acc.update(v for v in values if v is not None)
            case "cardinality":
                acc.update(values)
            case "count":
                acc.update(sum(1 for v in values if v is not None))
            case _:
                raise ValueError(f"Unhandled accumulator kind: {kind!r}")



def map_task(task: Task) -> tuple[tuple[str, ...], PartialStats]:
    """Map one task split to ``(partition_key, partial)``, using the booted config."""
    config = config_module.CONFIG
    partial = build_partial_stats(config.columns)
    columns = tuple(config.columns)
    readers = getattr(config, "readers", None)

    for batch in task.iter_batches(columns=columns, batch_size=config.batch_size, readers=readers):
        fold_batch(batch, partial, config)

    return task.partition, partial


def map_task_safe(task: Task) -> tuple[tuple[str, ...], PartialStats] | None:
    """Map one split, returning ``None`` on failure (for the orchestrator)."""
    try:
        return map_task(task)
    except Exception:
        logger.exception("Failed to map task %s", task)
        return None
