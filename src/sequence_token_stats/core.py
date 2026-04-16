"""Core statistics over tokenized sequences (NumPy / SciPy)."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from scipy.stats import entropy as scipy_entropy


def _log_scalar(x: float, base: float) -> float:
    if x <= 0.0:
        return float("-inf")
    return float(np.log(x) / np.log(base))


def _quantiles(values: Sequence[float], qs: tuple[float, ...]) -> dict[str, float]:
    """Linear interpolated quantiles; ``values`` non-empty."""
    arr = np.asarray(values, dtype=np.float64)
    arr.sort()
    qarr = np.quantile(arr, list(qs), method="linear")
    return {f"q{q}": float(v) for q, v in zip(qs, np.atleast_1d(qarr), strict=True)}


class _Welford:
    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        d2 = x - self.mean
        self.m2 += d * d2

    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return float(np.sqrt(self.m2 / (self.n - 1)))


class _Reservoir:
    """Reservoir sample for approximate quantiles (uniform over stream)."""

    __slots__ = ("cap", "data", "seen")

    def __init__(self, cap: int) -> None:
        self.cap = max(1, cap)
        self.data: list[float] = []
        self.seen = 0

    def add(self, x: float) -> None:
        self.seen += 1
        if len(self.data) < self.cap:
            self.data.append(x)
            return
        j = random.randint(0, self.seen - 1)
        if j < self.cap:
            self.data[j] = x

    def quantiles(self, qs: tuple[float, ...]) -> dict[str, float] | None:
        if not self.data:
            return None
        return _quantiles(self.data, qs)


def compute_token_statistics(
    tokenized_sequences: Iterable[Sequence[str]],
    *,
    log_base: float = math.e,
    quantile_sample_cap: int = 50_000,
) -> dict[str, Any]:
    """Aggregate three complementary views of token usage.

    Parameters
    ----------
    tokenized_sequences
        Each element is one sequence as an ordered list of token strings
        (already split the way your model counts them).
    log_base
        Base for all entropies and ``log(K)`` normalization (default natural log).
    quantile_sample_cap
        Max stored samples for approximate quantiles of per-sequence entropy
        summaries (reservoir sampling).

    Returns
    -------
    dict
        JSON-serializable structure with ``corpus_token_mass``, ``sequence_presence``,
        ``intra_sequence``, and ``meta`` fields. Each block includes a ``bounds`` object
        with theoretical extrema where applicable.
    """
    corpus: Counter[str] = Counter()
    presence: Counter[str] = Counter()
    w_when_present: dict[str, _Welford] = {}
    sum_q_all: Counter[str] = Counter()
    total_sequences = 0

    w_H = _Welford()
    w_J = _Welford()
    min_H = math.inf
    max_H = -math.inf
    min_J = math.inf
    max_J = -math.inf
    res_H = _Reservoir(quantile_sample_cap)
    res_J = _Reservoir(quantile_sample_cap)

    for tokens in tokenized_sequences:
        if not tokens:
            continue
        total_sequences += 1
        row: Counter[str] = Counter(tokens)
        L = sum(row.values())
        if L <= 0:
            continue

        for t in row:
            corpus[t] += row[t]
            presence[t] += 1
            q = row[t] / L
            sum_q_all[t] += q
            if t not in w_when_present:
                w_when_present[t] = _Welford()
            w_when_present[t].update(q)

        distinct = len(row)
        probs = [c / L for c in row.values()]
        H = float(scipy_entropy(probs, base=log_base))
        w_H.update(H)
        min_H = min(min_H, H)
        max_H = max(max_H, H)
        res_H.add(H)

        if distinct > 1:
            denom = _log_scalar(float(distinct), log_base)
            J = H / denom if denom > 0.0 and math.isfinite(denom) else 0.0
            w_J.update(J)
            min_J = min(min_J, J)
            max_J = max(max_J, J)
            res_J.add(J)

    total_tokens = sum(corpus.values())
    num_types = len(corpus)

    corpus_fractions: dict[str, float] = {}
    if total_tokens > 0:
        for t, c in corpus.items():
            corpus_fractions[t] = c / total_tokens

    entropy_corpus = 0.0
    if num_types > 0 and total_tokens > 0:
        entropy_corpus = float(scipy_entropy(list(corpus_fractions.values()), base=log_base))

    max_frac = max(corpus_fractions.values()) if corpus_fractions else 0.0

    K = num_types
    qs = (0.0, 0.25, 0.5, 0.75, 1.0)

    meta = {
        "total_sequences": total_sequences,
        "total_tokens": total_tokens,
        "num_token_types": num_types,
        "log_base": "e" if log_base == math.e else str(log_base),
    }

    corpus_block: dict[str, Any] = {
        "bounds": {
            "per_token_fraction": {"min": 0.0, "max": 1.0},
            "corpus_entropy": {"min": 0.0, "max": _log_scalar(float(K), log_base) if K > 0 else 0.0},
            "max_token_fraction": {"min": 1.0 / K if K > 0 else 0.0, "max": 1.0},
        },
        "total_tokens": total_tokens,
        "num_types": num_types,
        "corpus_entropy": entropy_corpus,
        "max_token_fraction": max_frac,
        "per_token": sorted(
            (
                {
                    "token": t,
                    "count": int(corpus[t]),
                    "fraction_of_tokens": corpus_fractions.get(t, 0.0),
                }
                for t in corpus
            ),
            key=lambda d: d["fraction_of_tokens"],
            reverse=True,
        ),
    }

    presence_block: dict[str, Any] = {
        "bounds": {
            "fraction_of_sequences": {"best": 1.0, "worst": 0.0},
        },
        "total_sequences": total_sequences,
        "per_token": sorted(
            (
                {
                    "token": t,
                    "sequences_with_token": int(presence[t]),
                    "fraction_of_sequences": (presence[t] / total_sequences) if total_sequences else 0.0,
                }
                for t in presence
            ),
            key=lambda d: d["fraction_of_sequences"],
            reverse=True,
        ),
    }

    intra_summary_H = {
        "mean": w_H.mean if w_H.n else None,
        "std": w_H.std() if w_H.n else None,
        "min": None if min_H is math.inf else min_H,
        "max": None if max_H is -math.inf else max_H,
        "quantiles_reservoir": res_H.quantiles(qs),
        "reservoir_size_cap": quantile_sample_cap,
        "reservoir_seen": res_H.seen,
    }
    intra_summary_J = {
        "mean": w_J.mean if w_J.n else None,
        "std": w_J.std() if w_J.n else None,
        "min": None if min_J is math.inf else min_J,
        "max": None if max_J is -math.inf else max_J,
        "quantiles_reservoir": res_J.quantiles(qs),
        "reservoir_size_cap": quantile_sample_cap,
        "reservoir_seen": res_J.seen,
    }

    per_token_intra = []
    for t in corpus:
        wp = w_when_present.get(t)
        mean_wp = wp.mean if wp and wp.n else None
        std_wp = wp.std() if wp and wp.n else None
        per_token_intra.append(
            {
                "token": t,
                "mean_q_when_present": mean_wp,
                "std_q_when_present": std_wp,
                "mean_q_across_all_sequences": (sum_q_all[t] / total_sequences) if total_sequences else 0.0,
                "sequences_with_token": int(presence[t]),
            }
        )
    per_token_intra.sort(key=lambda d: d["mean_q_across_all_sequences"], reverse=True)

    intra_block: dict[str, Any] = {
        "bounds": {
            "per_sequence_entropy_H": {
                "min": 0.0,
                "max_note": "At most log(K_s) for K_s distinct types in that sequence (same log base as analysis).",
            },
            "normalized_entropy_J": {"min": 0.0, "max": 1.0},
        },
        "sequence_entropy_H": intra_summary_H,
        "normalized_entropy_J": intra_summary_J,
        "per_token": per_token_intra,
    }

    return {
        "meta": meta,
        "corpus_token_mass": corpus_block,
        "sequence_presence": presence_block,
        "intra_sequence": intra_block,
    }


def analyze_strings(
    raw_sequences: Iterable[str],
    tokenize: Callable[[str], Sequence[str]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Tokenize each string then run :func:`compute_token_statistics`."""

    def gen() -> Iterable[list[str]]:
        for s in raw_sequences:
            if s is None:
                continue
            toks = tokenize(s)
            yield list(toks)

    return compute_token_statistics(gen(), **kwargs)
