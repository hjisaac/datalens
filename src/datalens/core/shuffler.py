"""Shuffle phase: group map outputs by partition key."""

from __future__ import annotations

from collections import defaultdict

from .types import PartialStats


def shuffle(
    mapped: list[tuple[tuple[str, ...], PartialStats]],
) -> dict[tuple[str, ...], list[PartialStats]]:
    """Group ``(partition_key, partial)`` pairs by partition key."""
    groups: dict[tuple[str, ...], list[PartialStats]] = defaultdict(list)
    for key, partial in mapped:
        groups[key].append(partial)
    return dict(groups)
