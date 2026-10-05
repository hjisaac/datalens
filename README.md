# DataLens

**Fast, streaming map-reduce statistics for tabular datasets.**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DataLens is a lightweight, high-performance library for computing corpus-level statistics across tabular data (Parquet, CSV, TSV, JSONL, and in-memory structures). Built on a streaming map-reduce architecture, it processes massive datasets in bounded memory using numerically stable online algorithms.

---

## Features

- **Streaming & Numerically Stable**: Uses Welford's algorithm for $O(1)$-memory online mean and variance calculation without catastrophic precision loss.
- **Multimodal Formats**: Native support for **Parquet**, **CSV**, **TSV**, **JSONL / NDJSON**, and direct **in-memory data** (lists of dicts, columnar arrays, PyArrow Tables, Pandas/Polars DataFrames).
- **Parallel Multi-Processing**: Transparently scales across all CPU cores for disk-bound corpora, while running zero-overhead in-process for in-memory data.
- **Unified Architecture**: A single, elegant pipeline (`Task -> Map -> Shuffle -> Reduce`) with zero branching between file and in-memory workflows.
- **First-Class Python & CLI Interfaces**: Strongly typed Python dataclasses (`AnalysisConfig`, `DataTask`, `FileTask`) alongside a modern **Typer** CLI with Rich output formatting.
- **Featherweight Base**: Base installation requires only pure Python and NumPy. Heavy dependencies like `pyarrow` are completely optional.

---

## Supported Metric Accumulators

| Metric Kind | Description | Outputs |
| :--- | :--- | :--- |
| `numeric` | Streaming summary via Welford's algorithm | `count`, `mean`, `std`, `min`, `max` |
| `quantile` / `tdigest` | Streaming percentiles via Ted Dunning's T-Digest | `count`, `min`, `max`, `p01`–`p99`, `iqr` |
| `categorical` | Exact frequency counter with optional top-N truncation | `unique`, `counts: {value: frequency}` |
| `cardinality` | Exact distinct-value tracking | `unique` |
| `count` | Row / event counter | `count` |

---

## Installation

### From GitHub
```bash
pip install git+https://github.com/hjisaac/datalens.git

# With optional Parquet support:
pip install "datalens[parquet] @ git+https://github.com/hjisaac/datalens.git"
```

### Local Development Installation
```bash
# Clone the repository
git clone git@github.com:hjisaac/datalens.git
cd datalens

# Install in editable mode
pip install -e ".[dev]"
```

---

## Quickstart

### 1. Python API

#### In-Memory Data (Single Process)
```python
from datalens import AnalysisConfig, run_analysis

# Define metrics to compute
config = AnalysisConfig(
    columns={
        "latency_ms": "numeric",
        "status_code": "categorical",
        "user_id": "cardinality",
    }
)

# Any in-memory records (list of dicts, columnar dict, or PyArrow Table)
records = [
    {"latency_ms": 12.5, "status_code": 200, "user_id": "u1"},
    {"latency_ms": 45.0, "status_code": 500, "user_id": "u2"},
    {"latency_ms": 18.2, "status_code": 200, "user_id": "u1"},
]

result = run_analysis(config, source=records)

# Save or inspect results
result.to_json("stats.json")
print(result.to_dict())
```

#### Scanning a Directory Dataset (Parallel Map-Reduce)
```python
from datalens import AnalysisConfig, run_analysis

config = AnalysisConfig(
    root="/data/events",
    file_pattern="*.parquet",      # or "*.csv", "*.jsonl"
    columns={
        "price": "numeric",
        "category": "categorical",
    },
    partition_depth=1,             # use the first subfolder as partition key
    workers=8,                     # parallel worker processes (None = all CPUs)
    top_categories=10,             # keep top 10 categories
)

result = run_analysis(config)
result.to_json("corpus_stats.json")
```

#### Using Streaming Accumulators Directly
```python
import numpy as np
from datalens import NumericAccumulator, CategoricalAccumulator

# Welford online numeric summary:
acc = NumericAccumulator()
acc.update(np.array([10.0, 20.0, 30.0]))
acc.update(np.array([40.0, 50.0]))
print(acc.result())
# {'count': 5, 'mean': 30.0, 'std': 14.14..., 'min': 10.0, 'max': 50.0}
```

---

### 2. Command Line Interface (CLI)

DataLens provides a built-in Typer command:

```bash
# Run analysis and print formatted JSON to stdout:
datalens config.yaml

# Save to output file with custom worker processes:
datalens config.yaml --output stats.json --workers 8

# Silent mode (suppress logs):
datalens config.yaml -o stats.json -q
```

---

## Configuration (`config.yaml`)

```yaml
# Dataset root directory
root: /path/to/dataset

# Physical column name → accumulator kind
columns:
  latency_ms: numeric
  status_code: categorical
  user_id: cardinality

# File pattern glob
file_pattern: "*.parquet"

# Number of worker processes (null = all CPUs; 1 = serial)
workers: null

# Rows read per batch during mapping
batch_size: 65536

# Truncate partition key depth from directory tree
partition_depth: null

# Keep only top-N categories (null = all)
top_categories: 10
```

---

## Testing

Run the test suite using `pytest`:

```bash
pytest
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
