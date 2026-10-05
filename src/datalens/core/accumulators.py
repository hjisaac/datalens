"""Mergeable streaming accumulators for map-reduce statistics.

Each accumulator supports:

* ``update`` — fold a chunk of values in (per batch during map),
* ``merge`` — combine two partials (during reduce),
* ``result`` — render a JSON-serialisable summary.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
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


@dataclass
class QuantileAccumulator(Accumulator):
    """Streaming quantile/percentile estimator using Ted Dunning's T-Digest (MergingDigest).

    Maintains bounded O(delta) memory while providing high accuracy across both
    central percentiles (median/P50, IQR) and extreme distribution tails (P01, P99, P99.9).
    """

    kind: ClassVar[str] = "quantile"

    delta: float = 100.0
    means: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    weights: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    buffer: list[np.ndarray] = field(default_factory=list)
    min_val: float = math.inf
    max_val: float = -math.inf
    total_weight: float = 0.0

    def update(self, values: np.ndarray) -> None:
        """Fold a batch of numeric values into the digest."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return
        finite = arr[~np.isnan(arr)]
        if finite.size == 0:
            return
        self.min_val = min(self.min_val, float(finite.min()))
        self.max_val = max(self.max_val, float(finite.max()))
        self.buffer.append(finite)
        if sum(len(b) for b in self.buffer) > max(1000, int(5 * self.delta)):
            self.compress()

    def compress(self) -> None:
        """Compress buffered points and existing centroids into merged centroids."""
        if not self.buffer and len(self.means) == 0:
            return
        if self.buffer:
            buf_vals = np.concatenate(self.buffer)
            buf_weights = np.ones(buf_vals.size, dtype=np.float64)
            self.buffer = []
            if len(self.means) > 0:
                all_means = np.concatenate([self.means, buf_vals])
                all_weights = np.concatenate([self.weights, buf_weights])
            else:
                all_means = buf_vals
                all_weights = buf_weights
        else:
            all_means = self.means
            all_weights = self.weights

        order = np.argsort(all_means)
        means = all_means[order]
        weights = all_weights[order]
        total_w = float(weights.sum())
        if total_w == 0:
            return

        delta = self.delta

        def k(q: float) -> float:
            return (delta / (2.0 * math.pi)) * math.asin(max(-1.0, min(1.0, 2.0 * q - 1.0)))

        new_means: list[float] = []
        new_weights: list[float] = []

        w_cum = 0.0
        cur_mu = float(means[0])
        cur_w = float(weights[0])
        q0 = 0.0

        for i in range(1, len(means)):
            cand_w = cur_w + float(weights[i])
            q1 = min(1.0, (w_cum + cand_w) / total_w)
            if k(q1) - k(q0) <= 1.0:
                cur_mu = (cur_mu * cur_w + float(means[i]) * float(weights[i])) / cand_w
                cur_w = cand_w
            else:
                new_means.append(cur_mu)
                new_weights.append(cur_w)
                w_cum += cur_w
                q0 = w_cum / total_w
                cur_mu = float(means[i])
                cur_w = float(weights[i])

        new_means.append(cur_mu)
        new_weights.append(cur_w)
        self.means = np.array(new_means, dtype=np.float64)
        self.weights = np.array(new_weights, dtype=np.float64)
        self.total_weight = total_w

    def merge(self, other: QuantileAccumulator) -> QuantileAccumulator:
        """Merge another quantile accumulator into this one."""
        self.compress()
        other.compress()
        if other.total_weight == 0:
            return self
        if self.total_weight == 0:
            self.means = other.means.copy()
            self.weights = other.weights.copy()
            self.min_val = other.min_val
            self.max_val = other.max_val
            self.total_weight = other.total_weight
            return self

        self.min_val = min(self.min_val, other.min_val)
        self.max_val = max(self.max_val, other.max_val)
        self.means = np.concatenate([self.means, other.means])
        self.weights = np.concatenate([self.weights, other.weights])
        self.buffer = []
        self.compress()
        return self

    def quantile(self, q: float) -> float | None:
        """Compute the estimated value at quantile q in [0, 1]."""
        self.compress()
        if len(self.means) == 0:
            return None
        if len(self.means) == 1 or q <= 0.0:
            return self.min_val
        if q >= 1.0:
            return self.max_val

        total_w = self.total_weight
        target = q * total_w
        cum_weights = np.cumsum(self.weights) - self.weights / 2.0

        if target < cum_weights[0]:
            return self.min_val + (target / cum_weights[0]) * (float(self.means[0]) - self.min_val)
        if target > cum_weights[-1]:
            return float(self.means[-1]) + ((target - cum_weights[-1]) / (total_w - cum_weights[-1])) * (self.max_val - float(self.means[-1]))

        idx = int(np.searchsorted(cum_weights, target))
        if idx < len(cum_weights) and cum_weights[idx] == target:
            return float(self.means[idx])
        left = idx - 1
        right = idx
        span = cum_weights[right] - cum_weights[left]
        if span == 0:
            return float(self.means[left])
        fraction = (target - cum_weights[left]) / span
        return float(self.means[left] + fraction * (self.means[right] - self.means[left]))

    def result(self, quantiles: Sequence[float] | None = None) -> dict[str, Any]:
        """Render count, min, max, iqr, and percentiles."""
        self.compress()
        if self.total_weight == 0:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "p01": None,
                "p05": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "iqr": None,
            }

        target_quantiles = quantiles if quantiles is not None else [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        res: dict[str, Any] = {
            "count": int(self.total_weight),
            "min": self.min_val,
            "max": self.max_val,
        }

        for q in sorted(target_quantiles):
            val = self.quantile(q)
            pct = q * 100
            key = f"p{int(pct):02d}" if pct.is_integer() else f"p{pct:g}"
            res[key] = val

        p25 = self.quantile(0.25)
        p75 = self.quantile(0.75)
        res["iqr"] = (p75 - p25) if (p75 is not None and p25 is not None) else None
        return res

