"""Mergeable streaming accumulators for map-reduce statistics.

Each accumulator supports:

* ``update`` — fold a chunk of values in (per batch during map),
* ``merge`` — combine two partials (during reduce),
* ``result`` — render a JSON-serialisable summary.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from .registry import Accumulator


@dataclass
class CountAccumulator(Accumulator):
    """Simple row/event counter."""

    kind: ClassVar[str] = "count"
    count: int = 0

    def update(self, n: int) -> None:
        """Add ``n`` to the running total."""
        self.count += n

    def merge(self, other: CountAccumulator) -> CountAccumulator:
        """Merge another counter into this one."""
        self.count += other.count
        return self

    def result(self) -> dict[str, Any]:
        """Render the count."""
        return {"count": self.count}


@dataclass
class NumericAccumulator(Accumulator):
    """Streaming numeric summary (count, mean, std, min, max) via Welford's algorithm."""

    kind: ClassVar[str] = "numeric"
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        """Fold a numpy array into the summary, skipping NaNs."""
        if values.size == 0:
            return
        finite = values[~np.isnan(values)]
        batch_count = int(finite.size)
        if batch_count == 0:
            return
        batch_mean = float(finite.mean())
        batch_m2 = float(np.square(finite - batch_mean, dtype=np.float64).sum())
        self._combine(batch_count, batch_mean, batch_m2, float(finite.min()), float(finite.max()))

    def merge(self, other: NumericAccumulator) -> NumericAccumulator:
        """Merge another numeric accumulator into this one."""
        if other.count:
            self._combine(other.count, other.mean, other.m2, other.minimum, other.maximum)
        return self

    def _combine(self, count_b: int, mean_b: float, m2_b: float, min_b: float, max_b: float) -> None:
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
class CategoricalAccumulator(Accumulator):
    """Exact frequency table for a categorical column."""

    kind: ClassVar[str] = "categorical"
    counts: Counter = field(default_factory=Counter)

    def update(self, values: Iterable[Any]) -> None:
        """Fold values into the frequency table."""
        self.counts.update(values)

    def merge(self, other: CategoricalAccumulator) -> CategoricalAccumulator:
        """Merge another categorical accumulator into this one."""
        self.counts.update(other.counts)
        return self

    def result(self, top: int | None = None) -> dict[str, Any]:
        """Render unique count and (optionally truncated) frequency table."""
        items = self.counts.most_common(top)
        return {
            "unique": len(self.counts),
            "counts": {str(key): int(value) for key, value in items},
        }


@dataclass
class CardinalityAccumulator(Accumulator):
    """Exact distinct-value set for a high-cardinality column."""

    kind: ClassVar[str] = "cardinality"
    values: set[str] = field(default_factory=set)

    def update(self, values: Iterable[str | None]) -> None:
        """Add values to the distinct set, ignoring nulls."""
        self.values.update(str(value) for value in values if value is not None)

    def merge(self, other: CardinalityAccumulator) -> CardinalityAccumulator:
        """Union another distinct set into this one."""
        self.values |= other.values
        return self

    def result(self) -> dict[str, Any]:
        """Render the distinct count."""
        return {"unique": len(self.values)}
