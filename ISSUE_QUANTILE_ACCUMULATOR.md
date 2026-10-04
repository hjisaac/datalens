# Feature: Streaming Quantile / Percentile Accumulator

**Type:** Feature Request / Enhancement  
**Component:** `core/accumulators.py`, `core/registry.py`, `core/config.py`  
**Suggested Labels:** `enhancement`, `core`, `algorithms`

---

## 1. Summary & Problem Statement

Currently, numeric columns are summarized using `NumericAccumulator`, which computes:
- `count`
- `mean`
- `std` (via Welford's algorithm)
- `min`
- `max`

While Welford's algorithm provides an exact, numerically stable $O(1)$-space summary, real-world datasets (e.g., token lengths, file sizes, query latencies, precursor m/z, peptide scores) are frequently **skewed, bimodal, or heavy-tailed with extreme outliers**.

In these distributions:
- `mean` and `std` can be severely distorted by a few outliers.
- Users cannot determine typical values (**median / P50**), robust dispersion (**IQR: P75 - P25**), or tail behavior (**P95 / P99**).

We need a dedicated **Quantile / Percentile Accumulator** that fits seamlessly into the streaming Map-Reduce engine.

---

## 2. Technical Challenge in Map-Reduce

Unlike mean and variance, exact quantiles cannot be calculated in $O(1)$ memory without storing and sorting every value in the dataset. Storing all raw values per worker causes high memory consumption and large IPC serialization overhead during the shuffle phase.

### Design Options

| Approach | Space / Worker | Shuffle Payload | Accuracy | Dependencies |
| :--- | :--- | :--- | :--- | :--- |
| **A. Exact (Full Buffer)** | $O(N)$ (risks OOM on 100M+ rows) | High | 100% exact | None (pure Python / NumPy) |
| **B. Reservoir Sampling** | $O(K)$ bounded (e.g., $K=10{,}000$) | Small & constant | High for P25–P75; lower at extreme tails (P99.9) | None (pure Python / NumPy) |
| **C. T-Digest** | $O(C)$ small cluster count (~a few KB) | Minimal | Extremely accurate across all percentiles & tails | Lightweight library (e.g. `tdigest`) or custom sketch |
| **D. Adaptive / Hybrid** | Exact if $N \le 50{,}000$; Reservoir / T-Digest if $N > 50{,}000$ | Adaptive | Exact for small datasets; scalable for large | None or optional dependency |

**Recommendation:** Start with **Option B (Reservoir Sampling)** or **Option D (Adaptive)** for zero-dependency baseline, or support a pluggable **T-Digest** sketch for production-scale telemetry.

---

## 3. Proposed User Experience & Configuration

### `config.yaml`
```yaml
root: /path/to/dataset
file_pattern: "*.parquet"

columns:
  latency_ms: quantile      # or "percentiles"
  request_size: quantile
  charge: categorical
  score: numeric

# Optional: customize quantiles/percentiles to compute
quantiles: [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

# Optional: reservoir sample buffer size per partition (default: 10000)
quantile_buffer_size: 10000
```

---

## 4. Expected Output Format

```json
{
  "partition": ["region_us", "cluster_1"],
  "rows": 1500000,
  "latency_ms": {
    "count": 1500000,
    "min": 0.42,
    "max": 1420.5,
    "p01": 1.1,
    "p05": 2.5,
    "p25": 8.4,
    "p50": 18.2,
    "p75": 39.7,
    "p95": 94.3,
    "p99": 240.1,
    "iqr": 31.3
  }
}
```

---

## 5. Proposed Implementation Sketch

```python
import numpy as np
from dataclasses import dataclass, field
from typing import ClassVar, Any
from core.registry import Accumulator

@dataclass
class QuantileAccumulator(Accumulator):
    """Streaming quantile accumulator using bounded reservoir sampling."""

    kind: ClassVar[str] = "quantile"
    capacity: int = 10_000
    seen_count: int = 0
    samples: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def update(self, values: np.ndarray) -> None:
        """Fold incoming batch into the reservoir sample."""
        # Filter NaNs and update reservoir using Algorithm R or stream sampling
        ...

    def merge(self, other: "QuantileAccumulator") -> "QuantileAccumulator":
        """Merge another reservoir into this one using weighted sampling."""
        ...

    def result(self, quantiles: list[float] | None = None) -> dict[str, Any]:
        """Compute percentiles from sorted sample buffer."""
        ...
```

---

## 6. Implementation Checklist

- [ ] Add `QuantileAccumulator` to `core/accumulators.py`.
- [ ] Auto-register under `kind: "quantile"` (and alias `"percentiles"`).
- [ ] Add config options (`quantiles`, `quantile_buffer_size`) in `core/config.py`.
- [ ] Support batch folding in `core/mapper.py`.
- [ ] Add unit tests in `t/test_accumulators.py` (or `t/test_pipeline.py`):
  - Known distribution verification (e.g. uniform, normal).
  - Empty / all-NaN batch handling.
  - Multi-partition merge correctness.
- [ ] Document in `README.md` and `config.yaml.example`.
