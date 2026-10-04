"""Configuration objects for the dataset statistics module.

These dataclasses are intentionally free of any InstaNovo-internal imports so the
whole :mod:`instanovo.dataset_stats` package stays portable and can be dropped into
any codebase that stores MS2 training data as per-project parquet files.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from instanovo.dataset_stats.identity import IdentityConfig

# Logical field names accepted by ``--map logical=physical``.
MAPPABLE_SCALAR_FIELDS: frozenset[str] = frozenset(
    {
        "sequence",
        "unmodified_peptide",
        "fragmentation",
        "acquisition",
        "detector",
        "enzyme",
        "collision_energy",
        "charge",
        "precursor_mz",
        "header",
        "protein",
        "run",
        "usi",
        "project",
        "filepath",
    }
)


@dataclass(frozen=True)
class FieldMap:
    """Maps standard logical fields to physical parquet column names.

    The module always uses logical names internally (``run``, ``project``,
    ``sequence``, …). Defaults match the InstaNovo-FM corpus; override any
    name with ``--map logical=physical`` when a client schema differs.

    Note: logical ``run`` is the sample/run identifier used for the *runs*
    metric and optional row-level identity — it is not tied to any particular
    client column name such as ``experiment_name``.
    """

    sequence: str = "sequence"
    unmodified_peptide: str = "unmodified_peptide"
    fragmentation: str = "frag_type"
    acquisition: str = "acquisition"
    detector: str = "detector"
    enzyme: str = "enzyme"
    collision_energy: str = "collision_energy"
    charge: str = "precursor_charge"
    precursor_mz: str = "precursor_mz"
    header: str = "header"
    protein: str = "protein"
    run: str = "experiment_name"
    usi: str = "usi"
    project: str = "project"
    filepath: str = "filepath"
    score_columns: tuple[str, ...] = (
        "hyperscore",
        "expectation",
        "probability",
        "nextscore",
    )


def field_map_from_pairs(pairs: list[str]) -> FieldMap:
    """Build a :class:`FieldMap` from ``logical=physical`` CLI pairs."""
    overrides: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"Invalid --map entry (expected logical=physical): {pair!r}")
        logical, physical = pair.split("=", 1)
        logical = logical.strip()
        physical = physical.strip()
        if not logical or not physical:
            raise SystemExit(f"Invalid --map entry: {pair!r}")
        if logical not in MAPPABLE_SCALAR_FIELDS:
            known = ", ".join(sorted(MAPPABLE_SCALAR_FIELDS))
            raise SystemExit(f"Unknown logical field {logical!r}. Known fields: {known}")
        overrides[logical] = physical
    return dataclasses.replace(FieldMap(), **overrides)


@dataclass(frozen=True)
class MetricToggles:
    """Per-metric on/off switches.

    Disabling a metric means its source column is never read, so turning metrics
    off is also the primary way to speed up a run on very large corpora.
    """

    spectra: bool = True
    projects: bool = True
    runs: bool = True
    unique_sequences: bool = True
    unique_unmodified: bool = True
    peptide_length: bool = True
    modifications: bool = True
    fragmentation: bool = True
    acquisition: bool = True
    collision_energy: bool = True
    charge: bool = True
    precursor_mz: bool = True
    analyzer_class: bool = True
    organism: bool = True
    instrument: bool = True
    detector: bool = True
    enzyme: bool = True
    confidence_scores: bool = True

    @classmethod
    def all_disabled(cls) -> "MetricToggles":
        """Return a toggle set with every metric disabled (handy as a base to opt in)."""
        return cls(**{f.name: False for f in fields(cls)})

    def enabled_names(self) -> tuple[str, ...]:
        """Return the names of the metrics that are currently enabled."""
        return tuple(f.name for f in fields(self) if getattr(self, f.name))


@dataclass(frozen=True)
class StatsConfig:
    """Top-level configuration for a statistics run.

    Args:
        root: Directory that contains the per-tier folders.
        tiers: Tier sub-directory names to process, each holding ``PXD*`` projects.
        workers: Number of worker processes. ``None`` uses all available CPUs;
            ``1`` runs serially in-process (useful for debugging/profiling).
        per_project: When ``True`` the result keeps a per-project breakdown in
            addition to the per-tier totals.
        progress: When ``True`` (the default) live progress is logged during the
            run, reporting both project-based and file-based completion.
        batch_size: Number of rows pulled per parquet record batch. Bounds the peak
            memory of a single worker independently of file size.
        keep_top_categories: Cap on how many categories are returned for high
            cardinality categorical metrics (``None`` keeps them all).
        catalog: Optional path to the run-level metadata CSV (``search_data.csv``).
            When provided, ``organism`` and ``instrument`` are sourced from the
            catalog via a ``(project, run)`` join instead of being derived from
            the ``protein`` / ``header`` parquet columns.
        fields: Logical-to-physical column mapping.
        identity: Rules for file discovery and project/run resolution.
        metrics: Which metrics to compute.
    """

    root: Path
    tiers: tuple[str, ...] = ("lcfm", "mcfm", "hcfm")
    workers: int | None = None
    per_project: bool = True
    progress: bool = True
    catalog: Path | None = None
    batch_size: int = 65_536
    keep_top_categories: int | None = None
    fields: FieldMap = field(default_factory=FieldMap)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    metrics: MetricToggles = field(default_factory=MetricToggles)

    def resolved_workers(self) -> int:
        """Return the concrete worker count, expanding ``None`` to the CPU count."""
        if self.workers is not None:
            return max(1, self.workers)
        return max(1, os.cpu_count() or 1)
