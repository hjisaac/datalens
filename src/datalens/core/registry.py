"""Auto-registering factory for accumulator implementations.

Accumulators self-register: subclassing :class:`Accumulator` and setting a
``kind`` is enough to make a new kind available everywhere (
:func:`create_accumulator`, :func:`build_partial_stats`, and column-kind
validation when a config is loaded) with no separate list to keep in sync.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .types import AccKind, PartialStats


class Accumulator(ABC):
    """Base for auto-registering, mergeable streaming accumulators.

    Subclasses set a class-level ``kind`` and implement ``update``/``merge``/
    ``result``; defining the subclass is enough to register it under
    ``kind`` — no separate registration call or dict entry needed.
    """

    kind: ClassVar[AccKind]
    _registry: ClassVar[dict[AccKind, type["Accumulator"]]] = {}

    @abstractmethod
    def update(self, values: Any) -> None:
        """Fold a chunk of values into the running summary."""

    @abstractmethod
    def merge(self, other: Accumulator) -> Accumulator:
        """Merge another accumulator of the same kind into this one."""

    @abstractmethod
    def result(self, **kwargs: Any) -> dict[str, Any]:
        """Render a JSON-serialisable summary."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return  # still-abstract intermediate subclass; nothing to register yet
        if "kind" not in vars(cls):
            raise TypeError(f"{cls.__name__} must define 'kind' to auto-register")
        if cls.kind in Accumulator._registry:
            existing = Accumulator._registry[cls.kind].__name__
            raise ValueError(f"Accumulator kind {cls.kind!r} is already registered to {existing}")
        Accumulator._registry[cls.kind] = cls
        for alias in getattr(cls, "aliases", ()):
            Accumulator._registry[alias] = cls


def create_accumulator(kind: AccKind) -> Accumulator:
    """Instantiate a fresh accumulator for ``kind``."""
    try:
        return Accumulator._registry[kind]()
    except KeyError as exc:
        known = ", ".join(sorted(Accumulator._registry))
        raise ValueError(f"Unknown accumulator kind {kind!r}. Known kinds: {known}") from exc


def known_kinds() -> frozenset[AccKind]:
    """Return the set of currently-registered accumulator kinds."""
    return frozenset(Accumulator._registry)


def build_partial_stats(columns: dict[str, AccKind]) -> PartialStats:
    """Create an empty :class:`PartialStats` from a column configuration."""
    metrics: dict[str, Any] = {}
    for column, kind in columns.items():
        metrics[column] = create_accumulator(kind)
    return PartialStats(metrics=metrics)


# Imported for its side effect: defining the concrete accumulator classes in
# `accumulators.py` triggers `Accumulator.__init_subclass__` above, which
# registers each one. Placed last because `accumulators.py` imports
# `Accumulator` from this module, so `Accumulator` must exist first.
from . import accumulators as _accumulators  # noqa: E402,F401
