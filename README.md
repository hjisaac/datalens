# dataset-analyzer

Quick **sequence token statistics** for tabular datasets: read a CSV or Parquet column of strings, tokenize each row, and print a JSON report (`corpus_token_mass`, `sequence_presence`, `intra_sequence`, `meta`).

## Setup

```bash
uv sync
```

## Layout

- `src/sequence_token_stats/` — token statistics (`core.py`, `tests/` beside the module).
- `src/core_lib/` — small helpers (e.g. repo paths).
- `app/main.py` — CLI `dataset-analyzer`.
- `notebooks/` — exploratory work.

## Run

```bash
uv run dataset-analyzer                           # demo sequences
uv run dataset-analyzer data.parquet --column sequence
uv run dataset-analyzer data.csv --column tokens --sep '|'
```

Use as a library:

```python
from src.sequence_token_stats import analyze_strings, compute_token_statistics
```

## Tests

```bash
uv run python -m unittest discover -s src/sequence_token_stats/tests -v
```

## Adding dependencies

Edit root `pyproject.toml` only (no per-tool dependency files).
