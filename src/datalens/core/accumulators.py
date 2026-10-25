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


def _normalize_exemplar(ex: Any, val: float) -> dict[str, Any]:
    if isinstance(ex, dict):
        out = dict(ex)
        out.setdefault("value", val)
        return out
    if ex is None:
        return {"value": val}
    return {"id": ex, "value": val}


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
    count: int = 0
    sample_capacity: int = 250
    sample: list[float] = field(default_factory=list)
    centroid_exemplars: list[list[dict[str, Any]]] = field(default_factory=list)
    buffer_exemplars: list[list[dict[str, Any] | None]] = field(default_factory=list)
    min_exemplar: dict[str, Any] | None = None
    max_exemplar: dict[str, Any] | None = None

    def update(self, values: np.ndarray, exemplars: Sequence[Any] | None = None) -> None:
        """Fold a batch of numeric values (and optional row exemplars) into the digest."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return
        valid_mask = ~np.isnan(arr)
        finite = arr[valid_mask]
        if finite.size == 0:
            return

        finite_ex: list[dict[str, Any] | None] | None = None
        if exemplars is not None:
            if len(exemplars) == arr.size:
                raw_ex = [ex for ex, ok in zip(exemplars, valid_mask) if ok]
            elif len(exemplars) == finite.size:
                raw_ex = list(exemplars)
            else:
                raise ValueError(
                    f"Length of exemplars ({len(exemplars)}) must match values ({arr.size})."
                )
            finite_ex = [_normalize_exemplar(ex, float(v)) for ex, v in zip(raw_ex, finite)]

        min_idx = int(np.argmin(finite))
        max_idx = int(np.argmax(finite))
        batch_min = float(finite[min_idx])
        batch_max = float(finite[max_idx])

        if batch_min < self.min_val or (
            batch_min == self.min_val and self.min_exemplar is None and finite_ex is not None
        ):
            self.min_val = batch_min
            if finite_ex is not None:
                self.min_exemplar = finite_ex[min_idx]

        if batch_max > self.max_val or (
            batch_max == self.max_val and self.max_exemplar is None and finite_ex is not None
        ):
            self.max_val = batch_max
            if finite_ex is not None:
                self.max_exemplar = finite_ex[max_idx]

        if finite_ex is not None:
            if not self.buffer_exemplars and self.buffer:
                self.buffer_exemplars = [[None] * b.size for b in self.buffer]
            self.buffer_exemplars.append(finite_ex)
        elif self.buffer_exemplars or self.centroid_exemplars:
            self.buffer_exemplars.append([None] * finite.size)

        self.buffer.append(finite)

        # High-performance vectorized Algorithm R reservoir sampling
        if self.sample_capacity > 0:
            cur_len = len(self.sample)
            needed = self.sample_capacity - cur_len
            if needed > 0:
                take = min(needed, finite.size)
                self.sample.extend(float(x) for x in finite[:take])
                finite_tail = finite[take:]
                start_n = self.count + take
            else:
                finite_tail = finite
                start_n = self.count

            rem = finite_tail.size
            if rem > 0 and start_n >= self.sample_capacity:
                positions = np.arange(start_n + 1, start_n + rem + 1, dtype=np.float64)
                probs = self.sample_capacity / positions
                selected_mask = np.random.random(rem) < probs
                selected_items = finite_tail[selected_mask]
                if selected_items.size > 0:
                    replace_slots = np.random.randint(0, self.sample_capacity, size=selected_items.size)
                    for slot, val in zip(replace_slots, selected_items):
                        self.sample[slot] = float(val)

        self.count += finite.size

        if sum(len(b) for b in self.buffer) > max(1000, int(5 * self.delta)):
            self.compress()

    def compress(self) -> None:
        """Compress buffered points and existing centroids into merged centroids."""
        if not self.buffer and len(self.means) == 0:
            return

        has_exemplars = bool(self.centroid_exemplars or self.buffer_exemplars)
        all_ex: list[list[dict[str, Any]]] | None = None

        if self.buffer:
            buf_vals = np.concatenate(self.buffer)
            buf_weights = np.ones(buf_vals.size, dtype=np.float64)
            self.buffer = []
            if has_exemplars:
                flat_buf_ex: list[list[dict[str, Any]]] = [
                    [ex] if ex is not None else []
                    for chunk in self.buffer_exemplars
                    for ex in chunk
                ]
                self.buffer_exemplars = []
                if len(self.means) > 0:
                    existing_ex = (
                        self.centroid_exemplars
                        if len(self.centroid_exemplars) == len(self.means)
                        else [[] for _ in range(len(self.means))]
                    )
                    all_ex = existing_ex + flat_buf_ex
                else:
                    all_ex = flat_buf_ex
            if len(self.means) > 0:
                all_means = np.concatenate([self.means, buf_vals])
                all_weights = np.concatenate([self.weights, buf_weights])
            else:
                all_means = buf_vals
                all_weights = buf_weights
        else:
            all_means = self.means
            all_weights = self.weights
            if has_exemplars and len(self.centroid_exemplars) == len(self.means):
                all_ex = self.centroid_exemplars

        order = np.argsort(all_means, kind="mergesort")
        means = all_means[order]
        weights = all_weights[order]
        sorted_ex = [all_ex[idx] for idx in order] if all_ex is not None else None
        total_w = float(weights.sum())
        if total_w == 0:
            return

        delta = self.delta

        def k(q: float) -> float:
            return (delta / (2.0 * math.pi)) * math.asin(max(-1.0, min(1.0, 2.0 * q - 1.0)))

        def _pick_exemplars(start: int, end: int, target_mu: float) -> list[dict[str, Any]]:
            assert sorted_ex is not None
            cands = [ex for j in range(start, end) for ex in sorted_ex[j]]
            if len(cands) <= 3:
                return cands
            cands.sort(key=lambda ex: float(ex.get("value", target_mu)))
            mid_idx = min(
                range(len(cands)),
                key=lambda idx: abs(float(cands[idx].get("value", target_mu)) - target_mu),
            )
            picked = [cands[0], cands[mid_idx], cands[-1]]
            unique: list[dict[str, Any]] = []
            for item in picked:
                if item not in unique:
                    unique.append(item)
            return unique

        new_means: list[float] = []
        new_weights: list[float] = []
        new_exemplars: list[list[dict[str, Any]]] = []

        w_cum = 0.0
        cur_mu = float(means[0])
        cur_w = float(weights[0])
        q0 = 0.0
        cluster_start = 0

        for i in range(1, len(means)):
            cand_w = cur_w + float(weights[i])
            q1 = min(1.0, (w_cum + cand_w) / total_w)
            if k(q1) - k(q0) <= 1.0:
                cur_mu = (cur_mu * cur_w + float(means[i]) * float(weights[i])) / cand_w
                cur_w = cand_w
            else:
                new_means.append(cur_mu)
                new_weights.append(cur_w)
                if sorted_ex is not None:
                    new_exemplars.append(_pick_exemplars(cluster_start, i, cur_mu))
                w_cum += cur_w
                q0 = w_cum / total_w
                cur_mu = float(means[i])
                cur_w = float(weights[i])
                cluster_start = i

        new_means.append(cur_mu)
        new_weights.append(cur_w)
        if sorted_ex is not None:
            new_exemplars.append(_pick_exemplars(cluster_start, len(means), cur_mu))

        self.means = np.array(new_means, dtype=np.float64)
        self.weights = np.array(new_weights, dtype=np.float64)
        self.centroid_exemplars = new_exemplars if sorted_ex is not None else []
        self.total_weight = total_w

    def merge(self, other: QuantileAccumulator) -> QuantileAccumulator:
        """Merge another quantile accumulator into this one."""
        self.compress()
        other.compress()

        # Merge reservoir samples proportionally
        if self.sample_capacity > 0 and other.sample:
            if not self.sample:
                self.sample = list(other.sample)
            else:
                combined = self.sample + other.sample
                if len(combined) > self.sample_capacity:
                    w1 = max(1.0, float(self.count if self.count > 0 else self.total_weight))
                    w2 = max(1.0, float(other.count if other.count > 0 else other.total_weight))
                    prob_self = w1 / (w1 + w2)
                    n_self = int(round(self.sample_capacity * prob_self))
                    n_other = self.sample_capacity - n_self
                    p1 = np.random.choice(len(self.sample), size=min(len(self.sample), n_self), replace=False) if n_self > 0 else []
                    p2 = np.random.choice(len(other.sample), size=min(len(other.sample), n_other), replace=False) if n_other > 0 else []
                    self.sample = [self.sample[i] for i in p1] + [other.sample[j] for j in p2]
                else:
                    self.sample = combined

        self.count += other.count

        if other.total_weight == 0:
            return self
        if self.total_weight == 0:
            self.means = other.means.copy()
            self.weights = other.weights.copy()
            self.min_val = other.min_val
            self.max_val = other.max_val
            self.total_weight = other.total_weight
            self.centroid_exemplars = [list(c) for c in other.centroid_exemplars]
            self.min_exemplar = dict(other.min_exemplar) if other.min_exemplar is not None else None
            self.max_exemplar = dict(other.max_exemplar) if other.max_exemplar is not None else None
            return self

        if other.min_val < self.min_val or (
            other.min_val == self.min_val and self.min_exemplar is None and other.min_exemplar is not None
        ):
            self.min_exemplar = dict(other.min_exemplar) if other.min_exemplar is not None else None
        self.min_val = min(self.min_val, other.min_val)

        if other.max_val > self.max_val or (
            other.max_val == self.max_val and self.max_exemplar is None and other.max_exemplar is not None
        ):
            self.max_exemplar = dict(other.max_exemplar) if other.max_exemplar is not None else None
        self.max_val = max(self.max_val, other.max_val)

        if self.centroid_exemplars or other.centroid_exemplars:
            ex_self = (
                self.centroid_exemplars
                if len(self.centroid_exemplars) == len(self.means)
                else [[] for _ in range(len(self.means))]
            )
            ex_other = (
                other.centroid_exemplars
                if len(other.centroid_exemplars) == len(other.means)
                else [[] for _ in range(len(other.means))]
            )
            self.centroid_exemplars = ex_self + ex_other

        self.means = np.concatenate([self.means, other.means])
        self.weights = np.concatenate([self.weights, other.weights])
        self.buffer = []
        self.buffer_exemplars = []
        self.compress()
        return self

    def cdf(self, value: float) -> float | None:
        """Estimate the cumulative distribution function P(X <= value) in [0, 1]."""
        self.compress()
        if len(self.means) == 0:
            return None
        if value <= self.min_val:
            return 0.0
        if value >= self.max_val:
            return 1.0

        total_w = self.total_weight
        if total_w == 0:
            return None

        cum_weights = np.cumsum(self.weights) - self.weights / 2.0

        if value <= self.means[0]:
            if self.means[0] == self.min_val:
                return 0.0
            return float(cum_weights[0] * (value - self.min_val) / (self.means[0] - self.min_val)) / total_w

        if value >= self.means[-1]:
            if self.max_val == self.means[-1]:
                return 1.0
            return float(cum_weights[-1] + (total_w - cum_weights[-1]) * (value - self.means[-1]) / (self.max_val - self.means[-1])) / total_w

        idx = int(np.searchsorted(self.means, value))
        left = idx - 1
        right = idx
        span_x = self.means[right] - self.means[left]
        if span_x == 0:
            return float(cum_weights[left] / total_w)
        fraction = (value - self.means[left]) / span_x
        target_w = cum_weights[left] + fraction * (cum_weights[right] - cum_weights[left])
        return float(max(0.0, min(1.0, target_w / total_w)))

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

    def exemplar(self, q: float) -> dict[str, Any] | None:
        """Return the concrete exemplar record closest to quantile ``q`` in [0, 1]."""
        self.compress()
        if len(self.means) == 0:
            return None
        if q <= 0.0:
            if self.min_exemplar is not None:
                return dict(self.min_exemplar)
            return {"value": self.min_val}
        if q >= 1.0:
            if self.max_exemplar is not None:
                return dict(self.max_exemplar)
            return {"value": self.max_val}

        q_val = self.quantile(q)
        if q_val is None:
            return None

        if self.centroid_exemplars:
            best_ex: dict[str, Any] | None = None
            best_dist = math.inf
            for i, cluster_cands in enumerate(self.centroid_exemplars):
                for ex in cluster_cands:
                    val_i = float(ex.get("value", self.means[i]))
                    dist = abs(val_i - q_val)
                    if dist < best_dist:
                        best_dist = dist
                        best_ex = ex
            if best_ex is not None:
                return dict(best_ex)

        idx = int(np.argmin(np.abs(self.means - q_val)))
        return {"value": float(self.means[idx])}

    def result(
        self,
        quantiles: Sequence[float] | None = None,
        sla: float | None = None,
        with_exemplars: bool | None = None,
    ) -> dict[str, Any]:
        """Render count, min, max, iqr, percentiles, optional SLA metrics, and optional exemplars."""
        self.compress()
        if self.total_weight == 0:
            empty_res: dict[str, Any] = {
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
            if with_exemplars:
                empty_res["exemplars"] = {}
            return empty_res

        total_n = self.count if self.count > 0 else int(round(self.total_weight))
        target_quantiles = quantiles if quantiles is not None else [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        res: dict[str, Any] = {
            "count": total_n,
            "min": self.min_val,
            "max": self.max_val,
        }

        sorted_qs = sorted(target_quantiles)
        q_keys: list[tuple[float, str]] = []
        for q in sorted_qs:
            val = self.quantile(q)
            pct = q * 100
            key = f"p{int(pct):02d}" if pct.is_integer() else f"p{pct:g}"
            res[key] = val
            q_keys.append((q, key))

        p25 = self.quantile(0.25)
        p75 = self.quantile(0.75)
        res["iqr"] = (p75 - p25) if (p75 is not None and p25 is not None) else None

        if sla is not None:
            sla_cdf = self.cdf(sla)
            if sla_cdf is not None:
                sla_exceeded_pct = max(0.0, min(100.0, (1.0 - sla_cdf) * 100.0))
                sla_count = int(round(total_n * (1.0 - sla_cdf)))
                res["sla"] = sla
                res["sla_exceeded_pct"] = round(sla_exceeded_pct, 2)
                res["sla_exceeded_count"] = sla_count
                res["sla_compliant_pct"] = round(100.0 - sla_exceeded_pct, 2)

        if self.sample:
            res["sample"] = list(self.sample)

        include_ex = with_exemplars if with_exemplars is not None else bool(self.centroid_exemplars)
        if include_ex:
            exemplars_map: dict[str, Any] = {
                "min": self.exemplar(0.0),
            }
            for q, key in q_keys:
                exemplars_map[key] = self.exemplar(q)
            exemplars_map["max"] = self.exemplar(1.0)
            res["exemplars"] = exemplars_map

        return res

