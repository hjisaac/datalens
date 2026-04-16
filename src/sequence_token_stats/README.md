# `sequence_token_stats`

Small library for **summarizing token usage** over many sequences. You pass either **pre-tokenized rows** (lists of token strings) or **raw strings plus a `tokenize` function**. It returns **one JSON-serializable `dict`** per run.

Typical uses: audit NLP or bioinformatics token vocabularies, compare train/validation slices, or sanity-check that a few symbols are not dominating every row.

## What you get

The report has four top-level keys:

| Block | Question it helps answer |
|-------|---------------------------|
| **`meta`** | Row count, total tokens, number of distinct token types, log base used for entropies. |
| **`corpus_token_mass`** | Pooled over all positions: counts, fraction of all tokens per type, corpus Shannon entropy, max single-token mass. Includes **`bounds`** (valid ranges for fractions and entropy). |
| **`sequence_presence`** | Per type: in what **fraction of sequences** does it appear at least once? Includes **`bounds`**. |
| **`intra_sequence`** | Per sequence: entropy of the within-row distribution, normalized entropy when \(K>1\), plus per-token summaries (mean mass when present, mean mass counting absent rows as zero). Approximate quantiles use a **fixed-size reservoir** so memory stays bounded on large corpora (`quantile_sample_cap`). |

Quantiles are **approximate** on very large data; raise or lower `quantile_sample_cap` on `compute_token_statistics` / `analyze_strings` as you like.

## Dependencies

- **NumPy**, **SciPy** (Shannon entropy and numeric helpers).

## Install / reuse

- **Copy-paste:** copy the `sequence_token_stats` package directory into your project, declare `numpy` and `scipy`, then import (no other runtime deps).
- **Vendored inside another repo** (e.g. under a parent package named `src`): use `from src.sequence_token_stats import ...` instead of the imports below.

## API (two entry points)

- **`compute_token_statistics(tokenized_sequences, *, log_base=math.e, quantile_sample_cap=50_000)`** — each item is an ordered sequence of token strings (list, tuple, etc.).
- **`analyze_strings(raw_sequences, tokenize, **kwargs)`** — each item is a string; `tokenize(s)` must return a sequence of token strings.

Full parameter and field documentation: **`core.py`** docstrings.

---

## Examples

### 1. Pre-tokenized rows (no tokenizer)

```python
import math

from sequence_token_stats import compute_token_statistics

rows = [
    ["the", "cat", "sat"],
    ["the", "dog", "sat"],
    ["the", "cat"],
]
report = compute_token_statistics(rows, log_base=math.e, quantile_sample_cap=10_000)

print(report["meta"])
# e.g. total_sequences, total_tokens, num_token_types

top = report["corpus_token_mass"]["per_token"][:5]
for row in top:
    print(row["token"], row["fraction_of_tokens"])
```

### 2. Raw strings + whitespace split

```python
from sequence_token_stats import analyze_strings

lines = [
    "BEGIN /foo/bar END",
    "BEGIN /foo/baz END",
    "BEGIN END",
]

def tokenize(s: str) -> list[str]:
    return s.split()

report = analyze_strings(lines, tokenize)
print(report["sequence_presence"]["per_token"][:3])
```

### 3. Pipe-delimited sequences (single-character separator)

```python
from sequence_token_stats import analyze_strings

def tokenize(s: str) -> list[str]:
    return [t for t in s.split("|") if t]  # drop empties if needed

report = analyze_strings(["A|B|C", "A|B", "B|C"], tokenize)
```

### 4. Custom tokenizer (e.g. alnum words with `regex`)

```python
import re

from sequence_token_stats import analyze_strings

_token_re = re.compile(r"[A-Za-z0-9]+|[.,;:]")

def tokenize(s: str) -> list[str]:
    return _token_re.findall(s)

report = analyze_strings(["Hello, world!  Hello again."], tokenize)
```

### 5. Tabular file: one sequence column (pandas)

```python
import json

import pandas as pd

from sequence_token_stats import analyze_strings

df = pd.read_parquet("dataset.parquet")  # or read_csv("dataset.csv")
col = "sequence"  # adjust to your column name

def tokenize(s: str) -> list[str]:
    return str(s).split()  # or .split("|"), etc.

report = analyze_strings(df[col].dropna().astype(str), tokenize)

with open("token_stats.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
```

### 6. Stream lines from disk (memory-friendly input)

You still build **one** report over the full stream (the library aggregates everything in one pass).

```python
from sequence_token_stats import compute_token_statistics

def token_rows(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line.split()  # one sequence per non-empty line

report = compute_token_statistics(token_rows("sequences.txt"))
```

### 7. Write JSON and read back a slice in another tool

```python
import json

from sequence_token_stats import compute_token_statistics

report = compute_token_statistics([["x", "y"], ["y", "z"]])
path = "out.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

with open(path, encoding="utf-8") as f:
    loaded = json.load(f)
assert loaded["meta"]["total_sequences"] == report["meta"]["total_sequences"]
```

---

## Tests (this monorepo)

From the repository root, with dependencies installed (`uv sync`):

```bash
python -m unittest discover -s src/sequence_token_stats/tests -v
```

---

## Design note

The three blocks (**corpus mass**, **row presence**, **within-row spread**) are complementary: a token can be rare globally but appear in every row, or frequent globally but concentrated in a few long sequences. The report keeps all three views in one structure so you can load a single JSON file in a notebook or dashboard.
