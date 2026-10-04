"""Mergeable streaming accumulators used to build statistics in a map-reduce style.

Every accumulator supports the same lifecycle:

* ``update(...)`` folds a chunk of values in (called once per parquet record batch),
* ``merge(other)`` combines two partial accumulators of the same type (called when
  reducing per-file results into per-project and per-tier totals),
* ``result()`` renders a plain, JSON-serialisable summary.

All accumulators are picklable so they can cross process boundaries unchanged.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class NumericAccumulator:
    """Streaming summary (count/mean/std/min/max) of a numeric field.

    Uses Welford's online algorithm in Chan's parallel-merge form, tracking the mean
    and ``M2`` (sum of squared deviations from the mean) rather than raw running sums.
    This is numerically stable (no catastrophic cancellation) and merges cleanly
    across workers. Memory is constant and ``NaN`` values are ignored.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        """Fold a numpy array of values into the summary, skipping NaNs."""
        if values.size == 0:
            return
        finite = values[~np.isnan(values)]
        batch_count = int(finite.size)
        if batch_count == 0:
            return
        batch_mean = float(finite.mean())
        batch_m2 = float(np.square(finite - batch_mean, dtype=np.float64).sum())
        self._combine(batch_count, batch_mean, batch_m2, float(finite.min()), float(finite.max()))

    def merge(self, other: "NumericAccumulator") -> "NumericAccumulator":
        """Merge another numeric accumulator into this one and return self."""
        if other.count:
            self._combine(other.count, other.mean, other.m2, other.minimum, other.maximum)
        return self

    def _combine(self, count_b: int, mean_b: float, m2_b: float, min_b: float, max_b: float) -> None:
        """Fold a second group's ``(count, mean, M2, min, max)`` in via Chan's formula."""
        count_a = self.count
        if count_a == 0:
            self.count, self.mean, self.m2, self.minimum, self.maximum = count_b, mean_b, m2_b, min_b, max_b
            return
        total = count_a + count_b
        delta = mean_b - self.mean
        self.mean += delta * count_b / total
        self.m2 += m2_b + delta * delta * count_a * count_b / total
        self.count = total
        self.minimum = min(self.minimum, min_b)
        self.maximum = max(self.maximum, max_b)

    def result(self) -> dict[str, Any]:
        """Render count, mean, population std-dev, min and max."""
        if self.count == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        return {
            "count": self.count,
            "mean": self.mean,
            "std": math.sqrt(self.m2 / self.count),
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class CategoricalAccumulator:
    """Exact frequency table for a low/medium cardinality categorical field."""

    counts: Counter = field(default_factory=Counter)

    def update(self, values: Iterable[Any]) -> None:
        """Fold an iterable of category values into the frequency table."""
        self.counts.update(values)

    def merge(self, other: "CategoricalAccumulator") -> "CategoricalAccumulator":
        """Merge another categorical accumulator into this one and return self."""
        self.counts.update(other.counts)
        return self

    def result(self, top: int | None = None) -> dict[str, Any]:
        """Render the unique count and the (optionally truncated) frequency table.

        Args:
            top: If set, only the ``top`` most common categories are returned under
                ``"counts"`` while ``"unique"`` still reflects the full cardinality.
        """
        items = self.counts.most_common(top)
        return {
            "unique": len(self.counts),
            "counts": {str(key): int(value) for key, value in items},
        }


@dataclass
class CardinalityAccumulator:
    """Exact distinct-value set for a high cardinality field (e.g. peptide sequences).

    Memory grows with the number of distinct values. For corpora large enough that
    this becomes a concern, swap in an approximate counter (e.g. HyperLogLog) behind
    the same ``update``/``merge``/``result`` interface.
    """

    values: set[str] = field(default_factory=set)

    def update(self, values: Iterable[str | None]) -> None:
        """Add a batch of values to the distinct set, ignoring ``None``."""
        self.values.update(value for value in values if value is not None)

    def merge(self, other: "CardinalityAccumulator") -> "CardinalityAccumulator":
        """Union another distinct set into this one and return self."""
        self.values |= other.values
        return self

    @property
    def count(self) -> int:
        """Number of distinct values seen so far."""
        return len(self.values)

    def result(self) -> dict[str, Any]:
        """Render the distinct count."""
        return {"unique": self.count}
