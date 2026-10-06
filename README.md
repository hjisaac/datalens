# DataLens

**Fast, streaming map-reduce statistics and publication-grade visualizations for tabular datasets.**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DataLens is a lightweight library for computing corpus-level statistics and generating publication figures across tabular data (Parquet, CSV, TSV, JSONL, and in-memory structures) in strictly bounded memory using numerically stable online algorithms.

---

## Features

- **Streaming & $O(1)$ Memory**: Welford's algorithm for online mean/variance and T-Digest for percentiles without loading full datasets.
- **Multimodal Formats**: Native support for **Parquet**, **CSV**, **TSV**, **JSONL**, and **in-memory data** (dicts, arrays, Arrow, DataFrames).
- **Parallel Multi-Processing**: Transparently scales across CPU cores for disk corpora with zero branching between file and in-memory workflows.
- **Publication Visualizations (`datalens.viz`)**: High-resolution figures (dual-panel quantile fans, beeswarm + box plots, partition comparisons) with automated SLA threshold overlays.
- **Featherweight**: Core engine depends only on pure Python and NumPy. `pyarrow`, `matplotlib`, and `mcp` are optional extras.

---

## Installation

```bash
# Core engine:
pip install git+https://github.com/hjisaac/datalens.git

# With optional extras:
pip install "datalens[parquet,viz,mcp] @ git+https://github.com/hjisaac/datalens.git"
```

---

## Quickstart

### Python API

```python
from datalens import AnalysisConfig, run_analysis

config = AnalysisConfig(
    root="/data/events",
    file_pattern="*.parquet",
    columns={
        "latency_ms": "quantile",   # percentiles + sample reservoir for plotting
        "price": "numeric",         # streaming mean, std, min, max
        "status": "categorical",    # frequency counters
    },
    plots=True,                     # auto-generate publication figures
    plot_dir="./plots",
    sla={"latency_ms": 100.0},      # SLA threshold line & violation rate
)

result = run_analysis(config)
result.to_json("stats.json")

# Or render plots on demand:
result.plot(output_dir="./plots", style="paper")
result.plot_beeswarm("latency_ms", output_path="latency.png", sla=100.0)
```

### CLI

```bash
# Run analysis:
datalens config.yaml -o stats.json

# Run with publication plots and SLA threshold:
datalens config.yaml --plots --sla 100.0 --plot-style paper
```

---

## Visualizations & SLA Overlays

DataLens generates publication-ready figures directly to PNG, SVG, or PDF:

- **Dual-Panel Quantile Distribution**: Percentile fan chart (IQR & P05–P95 shading) with Box & Whisker summary and Empirical CDF.
- **Beeswarm + Box Plot**: Box plot overlaid with jittered raw sample points for true distribution density without binning artifacts.
- **SLA Threshold Overlays**: Reference threshold lines with callout badges and truncation violation metrics (`sla_exceeded_pct`).
- **Visual Styles**: `datalens` (corporate modern), `paper` (academic print/LaTeX), and `dark`. Use `captions=False` to strip titles for paper captions.

<details>
<summary><b>Click to expand full configuration options (<code>config.yaml</code>)</b></summary>

```yaml
root: /path/to/dataset
file_pattern: "*.parquet"
columns:
  latency_ms: quantile
  price: numeric
  status_code: categorical
  user_id: cardinality
workers: null             # null = all CPUs, 1 = serial
batch_size: 65536
partition_depth: null
top_categories: 10

# Plotting & SLA
plots: true
plot_dir: "./plots"
plot_style: "datalens"    # "datalens" | "paper" | "dark"
plot_format: "png"        # "png" | "svg" | "pdf"
plot_captions: true
sla:
  latency_ms: 100.0
```
</details>

---

## Real-World Examples

<details>
<summary><b>Click to expand production examples (LLM Token Truncation, Microservices SLA, Multi-Partition Shards)</b></summary>

### 1. LLM Pre-training & Fine-Tuning: Context Window SLA & Truncation Analytics
In NLP dataset preparation, setting a sequence length cutoff (e.g. 512, 1024, or 4096 tokens) requires knowing the exact distribution tail and truncation impact across splits.

```python
from datalens import AnalysisConfig, run_analysis

# Analyze token lengths across train/val/test parquet shards
config = AnalysisConfig(
    root="/data/nlp/tokenized_corpus",
    file_pattern="*.parquet",
    columns={
        "token_count": "quantile",      # T-Digest percentiles + sample reservoir
        "language": "categorical",
        "doc_id": "cardinality",
    },
    partition_depth=1,                  # partitions: train, val, test
    sla={"token_count": 512.0},         # Context window cutoff
    plots=True,
    plot_dir="./figures/token_dist",
    plot_style="paper",                 # High-contrast, print-ready for academic papers
    plot_captions=False,                # Strips in-figure titles for LaTeX figure captions
)

result = run_analysis(config)

# Generate publication-grade beeswarm + box plot overlay for the paper:
result.plot_beeswarm(
    column="token_count",
    output_path="./figures/token_beeswarm.pdf",
    sla=512.0,
    style="paper",
    captions=False,
)

# Inspect exact truncation statistics:
metrics = result.to_dict()["global"]["token_count"]
print(f"P95: {metrics['p95']:.1f} tokens | P99: {metrics['p99']:.1f} tokens")
print(f"Truncated Documents: {metrics['sla_exceeded_count']:,} ({metrics['sla_exceeded_pct']:.2f}%)")
```

---

### 2. Microservices API Observability: Streaming Latency SLA Diagnostics
Process millions of streaming JSONL / NDJSON access logs without memory spikes, tracking p95/p99 tail latency and HTTP status code distributions across worker processes.

```python
from datalens import AnalysisConfig, run_analysis

config = AnalysisConfig(
    root="/var/log/api_gateway",
    file_pattern="access_*.jsonl",
    columns={
        "latency_ms": "quantile",
        "status_code": "categorical",
        "client_ip": "cardinality",
        "payload_bytes": "numeric",
    },
    workers=16,                         # Multi-core streaming map-reduce
    sla={"latency_ms": 200.0},          # 200ms latency SLA threshold
    top_categories=10,
    plots=True,
    plot_dir="./dashboard_plots",
    plot_style="dark",                  # Sleek dark theme for dashboard previews
)

result = run_analysis(config)
result.to_json("api_sla_report.json")
```

---

### 3. Multi-Partition Comparative Analysis across Shards
Compare numerical distributions across multiple shards or geographic partitions using automated side-by-side grouped comparisons.

```python
from datalens import AnalysisConfig, run_analysis

config = AnalysisConfig(
    root="/datasets/user_events",
    file_pattern="*.parquet",
    columns={
        "session_duration_sec": "quantile",
        "conversion_value": "numeric",
    },
    partition_depth=1,                  # partitions by subfolder (e.g. region=EU, region=US)
    sla={"session_duration_sec": 300.0},
    plots=True,
    plot_dir="./regional_comparison",
    plot_style="datalens",
)

result = run_analysis(config)
```

</details>

---

## Accumulators & Performance

| Kind | Description | Outputs |
| :--- | :--- | :--- |
| `numeric` | Streaming summary via Welford's algorithm | `count`, `mean`, `std`, `min`, `max` |
| `quantile` | Streaming percentiles (T-Digest + reservoir sampling) | `count`, `min`, `max`, `p01`–`p99`, `iqr`, `samples` |
| `categorical` | Exact frequency counter with optional top-N truncation | `unique`, `counts: {value: frequency}` |
| `cardinality` | Exact distinct-value tracking | `unique` |
| `count` | Row / event counter | `count` |

<details>
<summary><b>Click to view throughput benchmarks (up to 24M+ rows/sec)</b></summary>

Empirical measurements on a standard CPU core (1,000,000 rows, batch size 50,000):

| Component / Workload | Throughput | Latency (1M rows) | Memory Complexity |
| :--- | :--- | :--- | :--- |
| **`NumericAccumulator` (Welford)** | **~24.7M rows/s** | ~40.5 ms | $O(1)$ constant |
| **Full Pipeline (Columnar Map-Reduce)** | **~2.84M rows/s** | ~352 ms | $O(1)$ batch-bounded |
| **`QuantileAccumulator` (T-Digest)** | **~400K rows/s** | ~2.5 s | $O(C)$ centroid-bounded |
| **Multi-Column (4 mixed metrics, 100k rows)** | **~205K rows/s** | ~487 ms | $O(1)$ bounded |

> **Streaming Guarantee**: Because all statistics operate via online accumulators, memory usage remains strictly bounded regardless of dataset scale.

</details>

---

## Model Context Protocol (MCP) Server

DataLens includes an MCP server for AI assistants (Claude Desktop, Cursor, Gemini):

```bash
datalens mcp
```

<details>
<summary><b>Click to expand MCP tools & client configuration</b></summary>

### Claude Desktop / Cursor Configuration (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "datalens": {
      "command": "datalens",
      "args": ["mcp"]
    }
  }
}
```

### Available Tools
- `inspect_dataset`: Infers schema and null counts without reading into memory.
- `compute_statistics`: Runs distributed map-reduce statistics.
- `profile_dataset`: Auto-discovers schemas and returns full statistical profiles.
- `generate_dataset_plots`: Produces publication-grade charts directly from AI prompts.
</details>

---

## Testing & License

```bash
pytest
```

MIT License. See [LICENSE](LICENSE) for details.
