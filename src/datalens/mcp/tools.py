"""Core dataset inspection and profiling tools for the DataLens MCP server."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.config import AnalysisConfig
from ..core.orchestrator import run_analysis
from ..core.readers import read_batches, resolve_reader

SUPPORTED_EXTENSIONS = (".parquet", ".csv", ".tsv", ".jsonl", ".ndjson")


def _find_dataset_files(root_path: Path, pattern: str | None = None) -> list[Path]:
    """Find all supported data files under a file or directory path."""
    if root_path.is_file():
        return [root_path]

    if not root_path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {root_path}")

    files: list[Path] = []
    if pattern:
        files.extend(root_path.rglob(pattern))
    else:
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(root_path.rglob(f"*{ext}"))

    # Deduplicate and sort deterministically
    return sorted({f.resolve() for f in files if f.is_file()})


def inspect_dataset(path: str, sample_size: int = 100) -> dict[str, Any]:
    """Inspect dataset schema, row samples, column types, and suggest metric accumulators.

    Args:
        path: Path to a dataset file (Parquet, CSV, TSV, JSONL) or directory of files.
        sample_size: Maximum rows to sample for schema and data type inference.
    """
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    files = _find_dataset_files(target)
    if not files:
        raise ValueError(f"No supported tabular dataset files found in {path}")

    total_bytes = sum(f.stat().st_size for f in files)
    sample_file = files[0]
    format_name = resolve_reader(sample_file)

    # Read initial batch for schema inspection
    sample_batch = None
    try:
        for batch in read_batches(sample_file, columns=[], batch_size=sample_size):
            sample_batch = batch
            break
    except Exception as exc:
        raise RuntimeError(f"Failed to read sample from {sample_file}: {exc}") from exc

    if sample_batch is None or sample_batch.num_rows == 0:
        return {
            "path": str(target),
            "file_count": len(files),
            "total_size_bytes": total_bytes,
            "format": format_name,
            "sample_rows": 0,
            "columns": {},
            "recommended_config": {"columns": {}},
        }

    # Inspect each column
    col_analysis: dict[str, Any] = {}
    recommended_columns: dict[str, str] = {}

    for col in sample_batch.columns:
        values = sample_batch.column(col)
        non_nulls = [v for v in values if v is not None and str(v).strip() != ""]
        null_count = len(values) - len(non_nulls)

        if not non_nulls:
            col_type = "null"
            rec_kind = "count"
            distinct_sample = []
        else:
            # Check boolean
            if all(isinstance(v, bool) or str(v).lower() in ("true", "false", "0", "1") for v in non_nulls):
                col_type = "boolean"
                rec_kind = "categorical"
                distinct_sample = sorted(list({str(v) for v in non_nulls}))[:5]
            else:
                # Check numeric (int/float)
                is_numeric = True
                is_integer = True
                for v in non_nulls:
                    try:
                        f_val = float(v)
                        if isinstance(v, float) or "." in str(v) or not f_val.is_integer():
                            is_integer = False
                    except (ValueError, TypeError):
                        is_numeric = False
                        is_integer = False
                        break

                if is_numeric:
                    col_type = "integer" if is_integer else "float"
                    # If integer with very few unique values, suggest categorical
                    unique_vals = {float(v) for v in non_nulls}
                    if is_integer and len(unique_vals) <= 10:
                        rec_kind = "categorical"
                    else:
                        rec_kind = "quantile"
                    distinct_sample = sorted(list(unique_vals))[:5]
                else:
                    # String / Categorical / High-cardinality
                    col_type = "string"
                    unique_str = {str(v) for v in non_nulls}
                    ratio = len(unique_str) / len(non_nulls)
                    if len(unique_str) <= 25 or ratio < 0.2:
                        rec_kind = "categorical"
                    else:
                        rec_kind = "cardinality"
                    distinct_sample = sorted(list(unique_str))[:5]

        col_analysis[col] = {
            "type": col_type,
            "nulls_in_sample": null_count,
            "distinct_in_sample": len(set(str(v) for v in non_nulls)),
            "sample_values": distinct_sample,
            "recommended_kind": rec_kind,
        }
        recommended_columns[col] = rec_kind

    # Detect partition directories if input is a directory
    partitions_found: list[str] = []
    if target.is_dir():
        parent_dirs = {f.parent.relative_to(target).as_posix() for f in files}
        partitions_found = sorted([p for p in parent_dirs if p != "."])

    return {
        "path": str(target),
        "file_count": len(files),
        "total_size_bytes": total_bytes,
        "format": format_name,
        "sample_rows": sample_batch.num_rows,
        "partitions_detected": partitions_found[:20],
        "columns": col_analysis,
        "recommended_config": {
            "root": str(target),
            "columns": recommended_columns,
            "include_row_count": True,
        },
    }


def compute_statistics(
    path: str,
    columns: dict[str, str],
    file_pattern: str | None = None,
    partition_depth: int | None = None,
    quantiles: list[float] | None = None,
    top_categories: int | None = 10,
    workers: int | None = None,
    batch_size: int = 65_536,
) -> dict[str, Any]:
    """Execute streaming Map-Reduce statistics on a tabular dataset.

    Args:
        path: Path to dataset root directory or single file.
        columns: Mapping of column names to accumulator kind ('numeric', 'quantile', 'categorical', 'cardinality', 'count').
        file_pattern: Glob pattern to filter files (e.g. '*.parquet', '*.csv').
        partition_depth: Folder depth for partition grouping.
        quantiles: Custom percentiles to evaluate for quantile accumulators (e.g. [0.01, 0.5, 0.99]).
        top_categories: Maximum number of category frequencies to retain per categorical column.
        workers: Process count (None = all CPUs, 1 = serial in-process).
        batch_size: Rows per streaming batch.
    """
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if target.is_file():
        root = target.parent
        resolved_pattern = file_pattern or target.name
    else:
        root = target
        if not file_pattern:
            files = _find_dataset_files(target)
            resolved_pattern = f"*{files[0].suffix}" if files else "*.parquet"
        else:
            resolved_pattern = file_pattern

    config = AnalysisConfig(
        root=root,
        columns=columns,
        file_pattern=resolved_pattern,
        partition_depth=partition_depth,
        quantiles=quantiles,
        top_categories=top_categories,
        workers=workers,
        batch_size=batch_size,
    )

    result = run_analysis(config)
    return result.to_dict()


def profile_dataset(
    path: str,
    include_quantiles: bool = True,
    max_categories: int = 10,
    workers: int | None = None,
) -> dict[str, Any]:
    """Automated one-click profiling of any tabular dataset (Parquet, CSV, TSV, JSONL).

    Automatically discovers files, inspects the schema, maps columns to statistical
    accumulators, and computes complete streaming distribution metrics.

    Args:
        path: Path to dataset root directory or file.
        include_quantiles: If True, uses T-Digest quantile accumulators for numeric columns.
        max_categories: Maximum number of category frequencies to return.
        workers: Process count (None = all CPUs, 1 = serial).
    """
    inspection = inspect_dataset(path)
    rec_columns = dict(inspection["recommended_config"]["columns"])

    if not rec_columns:
        return {
            "dataset": inspection,
            "statistics": {},
            "status": "empty_dataset",
        }

    # Adjust numeric/quantile based on include_quantiles flag
    if not include_quantiles:
        for col, kind in rec_columns.items():
            if kind == "quantile":
                rec_columns[col] = "numeric"

    target = Path(path).expanduser().resolve()
    stats = compute_statistics(
        path=str(target),
        columns=rec_columns,
        top_categories=max_categories,
        workers=workers,
    )

    return {
        "dataset": {
            "path": inspection["path"],
            "file_count": inspection["file_count"],
            "total_size_bytes": inspection["total_size_bytes"],
            "format": inspection["format"],
            "sample_rows": inspection["sample_rows"],
            "partitions": inspection["partitions_detected"],
        },
        "schema": inspection["columns"],
        "statistics": stats,
    }
