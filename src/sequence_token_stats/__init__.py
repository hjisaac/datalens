"""Tokenizer-agnostic token distribution statistics for sequence datasets.

Depends on **NumPy** and **SciPy** (for Shannon entropy and quantiles).

Public API
----------
compute_token_statistics
    Aggregate corpus mass, sequence presence, and intra-sequence dominance
    from an iterable of token lists.
analyze_strings
    Convenience wrapper: raw strings + a ``tokenize(text) -> Sequence[str]`` callable.
"""

from .core import analyze_strings, compute_token_statistics

__all__ = ["analyze_strings", "compute_token_statistics"]
