"""Shared types and Task abstractions for the map-reduce statistics core."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

AccKind = Literal["numeric", "categorical", "cardinality", "count", "quantile", "percentiles", "tdigest"]


class Task(ABC):
    """Abstract input split / unit of map work."""

    partition: tuple[str, ...]

    @property
    def is_in_memory(self) -> bool:
        """True if the data resides in memory (runs in-process, bypassing worker pool)."""
        return False

    @abstractmethod
    def iter_batches(
        self,
        columns: Sequence[str],
        batch_size: int,
        readers: dict[str, str] | None = None,
    ) -> Iterator[Any]:
        """Yield ColumnBatch instances for the mapper to fold."""


@dataclass(frozen=True)
class FileTask(Task):
    """One mappable file split on disk (typically a Parquet, CSV, TSV, or JSONL file)."""

    path: str
    partition: tuple[str, ...] = ()

    @property
    def is_in_memory(self) -> bool:
        return False

    def iter_batches(
        self,
        columns: Sequence[str],
        batch_size: int,
        readers: dict[str, str] | None = None,
    ) -> Iterator[Any]:
        from .readers import read_batches

        yield from read_batches(
            self.path,
            columns=columns,
            batch_size=batch_size,
            readers=readers,
        )


@dataclass
class DataTask(Task):
    """One mappable in-memory data split (list of dicts, columnar dict, or PyArrow Table)."""

    data: Any
    partition: tuple[str, ...] = ()

    @property
    def is_in_memory(self) -> bool:
        return True

    def iter_batches(
        self,
        columns: Sequence[str],
        batch_size: int,
        readers: dict[str, str] | None = None,
    ) -> Iterator[Any]:
        from .readers import in_memory_to_batches

        yield from in_memory_to_batches(
            self.data,
            columns=columns,
            batch_size=batch_size,
        )


@dataclass
class PartialStats:
    """Mergeable partial statistics produced by the map phase."""

    row_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: PartialStats) -> PartialStats:
        """Merge another partial into this one."""
        self.row_count += other.row_count
        for name, accumulator in other.metrics.items():
            if name in self.metrics:
                self.metrics[name].merge(accumulator)
            else:
                self.metrics[name] = accumulator
        return self


@dataclass
class AnalysisResult:
    """Outcome of a full analysis run."""

    groups: dict[tuple[str, ...], dict[str, Any]]
    files_processed: int
    files_failed: int
    config_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "config": self.config_summary,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "groups": {
                self._format_partition_key(key): values for key, values in self.groups.items()
            },
        }

    @staticmethod
    def _format_partition_key(key: tuple[str, ...]) -> str:
        if not key:
            return "_all"
        return "/".join(key)

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """Write the results as a formatted JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)
