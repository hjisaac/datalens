"""Map-side readers that turn files into column batches."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_READERS: dict[str, str] = {
    ".parquet": "parquet",
    ".csv": "csv",
    ".tsv": "tsv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}


@dataclass(frozen=True)
class ColumnBatch:
    """A column-oriented record batch produced by a reader."""

    num_rows: int
    columns: dict[str, list]

    def has_column(self, name: str) -> bool:
        return name in self.columns

    def column(self, name: str) -> list:
        return self.columns[name]


def resolve_reader(path: str | Path, readers: dict[str, str] | None = None) -> str:
    """Return the reader kind for ``path`` using extension and optional overrides."""
    path = Path(path)
    if readers:
        for pattern, kind in readers.items():
            if fnmatch_match(path.name, pattern):
                return kind
    suffix = path.suffix.lower()
    try:
        return DEFAULT_READERS[suffix]
    except KeyError as exc:
        known = ", ".join(sorted(DEFAULT_READERS))
        raise ValueError(f"No reader for {path.name!r}. Supported extensions: {known}") from exc


def fnmatch_match(name: str, pattern: str) -> bool:
    """Match ``name`` against a glob ``pattern`` (importable for tests)."""
    import fnmatch

    return fnmatch.fnmatch(name, pattern)


def read_batches(
    path: str | Path,
    *,
    columns: Sequence[str],
    batch_size: int,
    reader: str | None = None,
    readers: dict[str, str] | None = None,
) -> Iterator[ColumnBatch]:
    """Yield column batches from ``path`` using the selected reader."""
    path = Path(path)
    kind = reader or resolve_reader(path, readers)
    if kind == "parquet":
        yield from _read_parquet_batches(path, columns, batch_size)
    elif kind == "csv":
        yield from _read_delimited_batches(path, columns, batch_size, delimiter=",")
    elif kind == "tsv":
        yield from _read_delimited_batches(path, columns, batch_size, delimiter="\t")
    elif kind == "jsonl":
        yield from _read_jsonl_batches(path, columns, batch_size)
    else:
        raise ValueError(f"Unknown reader kind: {kind!r}")


def _read_parquet_batches(path: Path, columns: Sequence[str], batch_size: int) -> Iterator[ColumnBatch]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read Parquet files. "
            "Install it with: pip install 'dataset-stats[parquet]' or pip install pyarrow"
        ) from exc

    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    read_columns = sorted(set(columns) & available)
    if not read_columns and parquet_file.metadata is not None and parquet_file.metadata.num_rows > 0:
        read_columns = sorted(available)
    for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns or None):
        yield _arrow_batch_to_column_batch(record_batch)


def _arrow_batch_to_column_batch(batch: Any) -> ColumnBatch:
    columns = {name: batch.column(name).to_pylist() for name in batch.schema.names}
    return ColumnBatch(num_rows=batch.num_rows, columns=columns)


def _read_delimited_batches(
    path: Path,
    columns: Sequence[str],
    batch_size: int,
    *,
    delimiter: str,
) -> Iterator[ColumnBatch]:
    # Fast path: use pyarrow.csv if installed
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv

        parse_options = pacsv.ParseOptions(delimiter=delimiter)
        convert_options = pacsv.ConvertOptions(include_columns=list(columns) if columns else None)
        reader = pacsv.open_csv(path, parse_options=parse_options, convert_options=convert_options)
        for record_batch in reader:
            yield _arrow_batch_to_column_batch(record_batch)
        return
    except (ImportError, Exception):
        pass

    # Standard fallback: Python built-in csv (no dependencies)
    with path.open(newline="", encoding="utf-8") as handle:
        dict_reader = csv.DictReader(handle, delimiter=delimiter)
        buffer: list[dict[str, str]] = []
        fieldnames = dict_reader.fieldnames or []
        selected = [name for name in columns if name in fieldnames] if columns else fieldnames
        for row in dict_reader:
            buffer.append(row)
            if len(buffer) >= batch_size:
                yield _dict_rows_to_batch(buffer, selected)
                buffer = []
        if buffer:
            yield _dict_rows_to_batch(buffer, selected)


def _dict_rows_to_batch(rows: list[dict[str, str]], columns: Sequence[str]) -> ColumnBatch:
    column_data = {name: [row.get(name) for row in rows] for name in columns}
    return ColumnBatch(num_rows=len(rows), columns=column_data)


def _read_jsonl_batches(path: Path, columns: Sequence[str], batch_size: int) -> Iterator[ColumnBatch]:
    buffer: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            buffer.append(json.loads(line))
            if len(buffer) >= batch_size:
                yield _json_rows_to_batch(buffer, columns)
                buffer = []
    if buffer:
        yield _json_rows_to_batch(buffer, columns)


def _json_rows_to_batch(rows: list[dict], columns: Sequence[str]) -> ColumnBatch:
    if columns:
        keys = [name for name in columns if any(name in row for row in rows)]
    else:
        keys = sorted({key for row in rows for key in row})
    column_data = {name: [row.get(name) for row in rows] for name in keys}
    return ColumnBatch(num_rows=len(rows), columns=column_data)


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def numeric_values(values: Any) -> np.ndarray:
    """Cast values to a float64 numpy array, treating None / nulls as NaN."""
    if isinstance(values, np.ndarray):
        if values.dtype == np.float64:
            return values
        return values.astype(np.float64)
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return np.array(
            [np.nan if v is None or v == "" else _safe_float(v) for v in values],
            dtype=np.float64,
        )


def in_memory_to_batches(
    data: Any,
    columns: Sequence[str],
    batch_size: int,
) -> Iterator[ColumnBatch]:
    """Yield ColumnBatch instances from in-memory structures."""
    # Check for PyArrow Table / RecordBatchReader without hard-importing pyarrow
    if hasattr(data, "to_batches") and hasattr(data, "schema"):
        batches = data.to_batches(max_chunksize=batch_size)
        for rb in batches:
            yield _arrow_batch_to_column_batch(rb)
        return

    if isinstance(data, list):
        for i in range(0, len(data), batch_size):
            chunk = data[i : i + batch_size]
            yield _dict_rows_to_batch(chunk, columns)
        return

    if isinstance(data, dict):
        first_val = next(iter(data.values())) if data else []
        if isinstance(first_val, (list, tuple, np.ndarray)):
            num_rows = len(first_val)
            for i in range(0, num_rows, batch_size):
                cols = {col: list(data[col][i : i + batch_size]) for col in columns if col in data}
                chunk_len = len(next(iter(cols.values()))) if cols else min(batch_size, num_rows - i)
                yield ColumnBatch(num_rows=chunk_len, columns=cols)
            return

    if hasattr(data, "to_arrow"):
        table = data.to_arrow()
        yield from in_memory_to_batches(table, columns=columns, batch_size=batch_size)
        return

    if hasattr(data, "to_dict"):
        cols = {col: data[col].tolist() for col in columns if col in getattr(data, "columns", ())}
        yield from in_memory_to_batches(cols, columns=columns, batch_size=batch_size)
        return

    raise TypeError(f"Unsupported in-memory data type: {type(data)!r}")
