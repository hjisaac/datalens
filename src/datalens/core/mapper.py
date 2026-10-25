"""Map phase: read one input split and emit a partition key with partial statistics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import config as config_module
from .readers import ColumnBatch, numeric_values
from .registry import build_partial_stats
from .types import FileTask, PartialStats, Task

logger = logging.getLogger(__name__)


def _build_batch_exemplars(
    batch: ColumnBatch,
    target_column: str,
    valid_indices: np.ndarray,
    finite_values: np.ndarray,
    config: Any,
    *,
    row_offset: int = 0,
    source_name: str | None = None,
) -> list[dict[str, Any]]:
    """Build per-row exemplar attribution dictionaries aligned with ``finite_values``."""
    id_col = getattr(config, "id_column", None)
    preview_col = getattr(config, "preview_column", None)
    exemplar_cols = getattr(config, "exemplar_columns", None)

    has_id = bool(id_col and batch.has_column(id_col))
    has_preview = bool(preview_col and batch.has_column(preview_col))
    id_vals = batch.column(id_col) if has_id else None
    preview_vals = batch.column(preview_col) if has_preview else None

    if exemplar_cols is not None:
        context_cols = [
            c for c in exemplar_cols if batch.has_column(c) and c not in (target_column, id_col, preview_col)
        ]
    elif not has_id:
        context_cols = [c for c in batch.columns if c not in (target_column, id_col, preview_col)]
    else:
        context_cols = []

    context_data = {c: batch.column(c) for c in context_cols}
    exemplars: list[dict[str, Any]] = []

    for k, raw_idx in enumerate(valid_indices):
        i = int(raw_idx)
        entry: dict[str, Any] = {"value": float(finite_values[k])}
        if has_id and id_vals is not None:
            entry["id"] = id_vals[i]
        else:
            entry["row_index"] = row_offset + i
            if source_name is not None:
                entry["source"] = source_name
        if has_preview and preview_vals is not None:
            p_val = preview_vals[i]
            entry["preview"] = p_val[:200] if isinstance(p_val, str) and len(p_val) > 200 else p_val
        for c, col_vals in context_data.items():
            c_val = col_vals[i]
            entry[c] = c_val[:200] if isinstance(c_val, str) and len(c_val) > 200 else c_val
        exemplars.append(entry)

    return exemplars


def fold_batch(
    batch: ColumnBatch,
    partial: PartialStats,
    config: Any,
    *,
    row_offset: int = 0,
    source_name: str | None = None,
) -> None:
    """Fold a single column batch into running partial statistics."""
    if config.include_row_count:
        partial.row_count += batch.num_rows
    with_exemplars = bool(getattr(config, "with_exemplars", False))
    for column, kind in config.columns.items():
        if not batch.has_column(column):
            continue
        values = batch.column(column)
        acc = partial.metrics[column]
        match kind:
            case "numeric":
                acc.update(numeric_values(values))
            case "quantile":
                arr = numeric_values(values)
                if with_exemplars:
                    valid_indices = np.flatnonzero(~np.isnan(arr))
                    if valid_indices.size > 0:
                        finite = arr[valid_indices]
                        exemplars = _build_batch_exemplars(
                            batch,
                            column,
                            valid_indices,
                            finite,
                            config,
                            row_offset=row_offset,
                            source_name=source_name,
                        )
                        acc.update(finite, exemplars=exemplars)
                else:
                    acc.update(arr)
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
    read_cols = list(config.columns)
    if getattr(config, "with_exemplars", False):
        extras = [
            getattr(config, "id_column", None),
            getattr(config, "preview_column", None),
            *(getattr(config, "exemplar_columns", None) or []),
        ]
        for col in extras:
            if col and col not in read_cols:
                read_cols.append(col)
    columns = tuple(read_cols)
    readers = getattr(config, "readers", None)
    source_name: str | None = None
    if isinstance(task, FileTask):
        task_path = Path(task.path)
        root_path = getattr(config, "root", None)
        if root_path is not None:
            try:
                rel = task_path.resolve().relative_to(Path(root_path).resolve())
                source_name = rel.as_posix() if rel.as_posix() != "." else task_path.name
            except ValueError:
                source_name = str(task.path)
        else:
            source_name = str(task.path)
    row_offset = 0

    for batch in task.iter_batches(columns=columns, batch_size=config.batch_size, readers=readers):
        fold_batch(batch, partial, config, row_offset=row_offset, source_name=source_name)
        row_offset += batch.num_rows

    return task.partition, partial


def map_task_safe(task: Task) -> tuple[tuple[str, ...], PartialStats] | None:
    """Map one split, returning ``None`` on failure (for the orchestrator)."""
    try:
        return map_task(task)
    except Exception:
        logger.exception("Failed to map task %s", task)
        return None
