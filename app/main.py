from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import pandas as pd

from src.sequence_token_stats import analyze_strings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dataset-analyzer",
        description="Token statistics on a sequence column (CSV/Parquet) via sequence_token_stats.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to CSV or Parquet. If omitted, runs a tiny built-in demo.",
    )
    parser.add_argument(
        "--column",
        default="sequence",
        help="Column name containing one string per row (default: sequence).",
    )
    parser.add_argument(
        "--sep",
        default=None,
        metavar="CHAR",
        help="Single-character delimiter for tokens; default is whitespace split.",
    )
    parser.add_argument(
        "--quantile-sample-cap",
        type=int,
        default=50_000,
        help="Reservoir cap for per-sequence entropy quantiles (default: 50000).",
    )
    args = parser.parse_args(argv)

    if args.sep is not None:
        if len(args.sep) != 1:
            print("--sep must be a single character", file=sys.stderr)
            return 2

        def tokenize(s: str) -> Sequence[str]:
            return s.split(args.sep)

    else:

        def tokenize(s: str) -> Sequence[str]:
            return s.split()

    if args.path:
        path = args.path
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        if args.column not in df.columns:
            print(f"Column {args.column!r} not in: {list(df.columns)}", file=sys.stderr)
            return 2
        series = df[args.column].dropna().astype(str)
    else:
        series = pd.Series(["A B C", "A A D", "B C"])

    report = analyze_strings(series, tokenize, quantile_sample_cap=args.quantile_sample_cap)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
