# Architecture

## Scope

Single analysis tool: **`sequence_token_stats`** — statistics over tokenized sequences (corpus mass, per-sequence presence, intra-sequence entropy summaries). No ML training stack; no generic “analyzers” layer.

## Layout

- **`src.sequence_token_stats`** — `compute_token_statistics` / `analyze_strings` (`core.py`); **unit tests** live in `src/sequence_token_stats/tests/` next to the module.
- **`src.core_lib`** — optional tiny shared helpers.
- **`app`** — `dataset-analyzer` CLI: reads a column of strings, applies a simple tokenizer, prints JSON.

## Dependencies

- **Base:** `numpy`, `pandas`, `scipy` (SciPy for entropy; pandas for CLI I/O).

## Execution

- **Library:** `from src.sequence_token_stats import compute_token_statistics` (pre-tokenized rows) or `analyze_strings` (strings + `tokenize`).
- **CLI:** `uv run dataset-analyzer [path] [--column NAME] [--sep CHAR]`.

## Non-goals

- Nested repos, submodules, duplicate dependency manifests.
- Optional “extras” split across many stacks (keep the footprint small unless you add a real second tool later).
