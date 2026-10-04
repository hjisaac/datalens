"""Per-file collection: turn one parquet file into a mergeable bundle of statistics.

:func:`collect_file` is the unit of parallel work. It is a top-level function (so it
pickles cleanly into worker processes), reads a single parquet file in bounded record
batches, and returns a :class:`TierAccumulator` that the runner reduces into
per-project and per-tier totals.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from instanovo.dataset_stats.accumulators import (
    CardinalityAccumulator,
    CategoricalAccumulator,
    NumericAccumulator,
)
from instanovo.dataset_stats.catalog import CatalogEntry, RunKey, normalize_run_name, parquet_run_name
from instanovo.dataset_stats.config import FieldMap, MetricToggles
from instanovo.dataset_stats.identity import IdentityConfig, resolve_project, resolve_run
from instanovo.dataset_stats.parsing import (
    analyzer_class_from_header,
    modifications_from_sequence,
    organism_from_protein,
)

# Module-level catalog, set once per process via init_worker_catalog.
# Using a global avoids pickling the dict for every task submission when
# running with a ProcessPoolExecutor (it is set by the pool initializer or
# directly before serial execution).
_catalog: dict[RunKey, CatalogEntry] = {}


def init_worker_catalog(catalog: dict[RunKey, CatalogEntry]) -> None:
    """Set the module-level catalog in this process.

    Called once per worker via the pool initializer (and once in the main
    process before serial execution) so the catalog is never pickled per task.
    """
    global _catalog
    _catalog = catalog

Accumulator = Union[NumericAccumulator, CategoricalAccumulator, CardinalityAccumulator]
_SCORE_PREFIX = "score:"
# Metrics that can be filled from search_data.csv when parquet columns are absent.
_CATALOG_FALLBACK_METRICS = frozenset(
    {"organism", "instrument", "acquisition", "fragmentation", "detector", "enzyme"}
)
# Metrics with no expected parquet column in InstaNovo-FM; catalog unless --map adds one.
_CATALOG_ONLY_METRICS = frozenset({"organism", "instrument", "detector", "enzyme"})
CATALOG_FALLBACK_METRICS = _CATALOG_FALLBACK_METRICS


@dataclass(frozen=True)
class FileTask:
    """A single parquet file to process, tagged with its tier and project."""

    tier: str
    project: str
    path: str


@dataclass
class TierAccumulator:
    """A bundle of enabled accumulators plus the running spectrum count.

    The same class is used at every level of the reduction (file, project, tier);
    merging two instances simply merges each contained accumulator.
    """

    metrics: dict[str, Accumulator] = field(default_factory=dict)
    spectra: int = 0
    file_columns: frozenset[str] = frozenset()
    catalog_files_matched: int = 0
    catalog_files_missed: int = 0
    catalog_run_spectra: dict[RunKey, int] = field(default_factory=dict)
    # Per-metric spectrum counts by source (parquet / catalog / protein_proxy).
    source_counts: dict[str, Counter[str]] = field(default_factory=dict)

    def record_source(self, metric: str, source: str, count: int) -> None:
        """Record how many spectra contributed to ``metric`` via ``source``."""
        if count > 0:
            self.source_counts.setdefault(metric, Counter())[source] += count

    def merge(self, other: "TierAccumulator") -> "TierAccumulator":
        """Merge another accumulator bundle into this one and return self."""
        self.spectra += other.spectra
        self.catalog_files_matched += other.catalog_files_matched
        self.catalog_files_missed += other.catalog_files_missed
        for key, count in other.catalog_run_spectra.items():
            self.catalog_run_spectra[key] = self.catalog_run_spectra.get(key, 0) + count
        for metric, counts in other.source_counts.items():
            self.source_counts.setdefault(metric, Counter()).update(counts)
        for name, accumulator in other.metrics.items():
            if name in self.metrics:
                self.metrics[name].merge(accumulator)  # type: ignore[arg-type]
            else:
                self.metrics[name] = accumulator
        return self


def build_accumulator(metrics: MetricToggles, fields_: FieldMap) -> TierAccumulator:
    """Create an empty :class:`TierAccumulator` holding only the enabled metrics."""
    bundle: dict[str, Accumulator] = {}
    if metrics.projects:
        bundle["projects"] = CardinalityAccumulator()
    if metrics.runs:
        bundle["runs"] = CardinalityAccumulator()
    if metrics.unique_sequences:
        bundle["unique_sequences"] = CardinalityAccumulator()
    if metrics.unique_unmodified:
        bundle["unique_unmodified"] = CardinalityAccumulator()
    if metrics.peptide_length:
        bundle["peptide_length"] = CategoricalAccumulator()
    if metrics.modifications:
        bundle["modifications"] = CategoricalAccumulator()
    if metrics.fragmentation:
        bundle["fragmentation"] = CategoricalAccumulator()
    if metrics.acquisition:
        bundle["acquisition"] = CategoricalAccumulator()
    if metrics.collision_energy:
        bundle["collision_energy"] = NumericAccumulator()
    if metrics.charge:
        bundle["charge"] = CategoricalAccumulator()
    if metrics.precursor_mz:
        bundle["precursor_mz"] = NumericAccumulator()
    if metrics.analyzer_class:
        bundle["analyzer_class"] = CategoricalAccumulator()
    if metrics.organism:
        bundle["organism"] = CategoricalAccumulator()
    if metrics.instrument:
        bundle["instrument"] = CategoricalAccumulator()
    if metrics.detector:
        bundle["detector"] = CategoricalAccumulator()
    if metrics.enzyme:
        bundle["enzyme"] = CategoricalAccumulator()
    if metrics.confidence_scores:
        for column in fields_.score_columns:
            bundle[f"{_SCORE_PREFIX}{column}"] = NumericAccumulator()
    return TierAccumulator(metrics=bundle)


def required_columns(
    metrics: MetricToggles,
    fields_: FieldMap,
    identity: IdentityConfig,
    catalog_loaded: bool = False,
) -> set[str]:
    """Return the parquet columns that must be read to satisfy the enabled metrics.

    Args:
        metrics: Which metrics are enabled.
        fields_: Logical-to-physical column mapping.
        identity: Project/run resolution rules (may require extra columns).
        catalog_loaded: When ``True``, run-level metadata (organism, instrument,
            acquisition, fragmentation) may come from the catalog when parquet
            columns are absent.
    """
    columns: set[str] = set()
    if metrics.unique_sequences or metrics.modifications:
        columns.add(fields_.sequence)
    if metrics.unique_unmodified or metrics.peptide_length:
        columns.add(fields_.unmodified_peptide)
    if metrics.runs and identity.run_source in ("column", "auto"):
        columns.add(fields_.run)
    if metrics.fragmentation:
        columns.add(fields_.fragmentation)
    if metrics.acquisition:
        columns.add(fields_.acquisition)
    if metrics.detector:
        columns.add(fields_.detector)
    if metrics.enzyme:
        columns.add(fields_.enzyme)
    if metrics.collision_energy:
        columns.add(fields_.collision_energy)
    if metrics.charge:
        columns.add(fields_.charge)
    if metrics.precursor_mz:
        columns.add(fields_.precursor_mz)
    if metrics.analyzer_class:
        columns.add(fields_.header)
    if metrics.organism and not catalog_loaded:
        columns.add(fields_.protein)
    if metrics.confidence_scores:
        columns.update(fields_.score_columns)
    if catalog_loaded and _catalog_metrics_enabled(metrics):
        columns.add(fields_.run)
    columns.update(identity.row_identity_columns(fields_))
    return columns


def _catalog_metrics_enabled(metrics: MetricToggles) -> bool:
    """Return whether any catalog-backed metadata metric is enabled."""
    return any(getattr(metrics, name) for name in _CATALOG_FALLBACK_METRICS)


def _uses_catalog_for_metric(metric: str, fields_: FieldMap, file_columns: frozenset[str]) -> bool:
    """Return whether ``metric`` should be filled from the catalog for this file."""
    if metric in {"organism", "instrument"}:
        return True
    if metric in {"detector", "enzyme", "acquisition", "fragmentation"}:
        column = getattr(fields_, metric)
        return column not in file_columns
    return False


def _to_float(array: pa.Array) -> np.ndarray:
    """Cast an arrow array to a float64 numpy array (nulls become ``NaN``)."""
    return pc.cast(array, pa.float64()).to_numpy(zero_copy_only=False)


def _update_runs(
    metrics: dict[str, Accumulator],
    batch: pa.RecordBatch,
    present: set[str],
    fields_: FieldMap,
    identity: IdentityConfig,
    file_path: str,
) -> None:
    """Update the runs metric using identity rules (column, path, or filename)."""
    if "runs" not in metrics:
        return
    source = identity.run_source
    if source == "column" or (source == "auto" and fields_.run in present):
        if fields_.run in present:
            values = batch.column(fields_.run).to_pylist()
            metrics["runs"].update(value for value in values if value is not None)
        return
    if source in ("path", "auto"):
        run_col = identity.run_field(fields_)
        if run_col and run_col in present:
            for value in batch.column(run_col).to_pylist():
                run = resolve_run(value, identity, file_path=file_path)
                if run:
                    metrics["runs"].update([run])
            return
    run = resolve_run(None, identity, file_path=file_path)
    if run:
        metrics["runs"].update([run])


def _catalog_run_keys_in_batch(
    batch: pa.RecordBatch,
    fields_: FieldMap,
    file_path: str,
    task_project: str,
) -> Counter[RunKey]:
    """Count spectra per ``(project, run)`` catalog key inside a record batch."""
    present = set(batch.schema.names)
    if fields_.run in present:
        runs = batch.column(fields_.run).to_pylist()
    else:
        fallback = parquet_run_name(file_path)
        runs = [fallback] * batch.num_rows
    counts: Counter[RunKey] = Counter()
    for run in runs:
        if run is None:
            continue
        text = normalize_run_name(str(run))
        if text:
            counts[(task_project, text)] += 1
    return counts


def _track_catalog_runs(
    bundle: TierAccumulator,
    batch: pa.RecordBatch,
    fields_: FieldMap,
    file_path: str,
    task_project: str,
) -> None:
    """Record per-run spectrum counts for a later catalog metadata join."""
    if not _catalog or not _catalog_metrics_enabled_from_bundle(bundle):
        return
    for key, count in _catalog_run_keys_in_batch(batch, fields_, file_path, task_project).items():
        bundle.catalog_run_spectra[key] = bundle.catalog_run_spectra.get(key, 0) + count


def _catalog_metrics_enabled_from_bundle(bundle: TierAccumulator) -> bool:
    return any(name in bundle.metrics for name in _CATALOG_FALLBACK_METRICS)


def _update_from_batch(
    bundle: TierAccumulator,
    batch: pa.RecordBatch,
    fields_: FieldMap,
    identity: IdentityConfig,
    file_path: str,
    task_project: str,
) -> None:
    """Fold a single record batch into the accumulator bundle."""
    present = set(batch.schema.names)
    metrics = bundle.metrics
    bundle.spectra += batch.num_rows
    _track_catalog_runs(bundle, batch, fields_, file_path, task_project)

    if ("unique_sequences" in metrics or "modifications" in metrics) and fields_.sequence in present:
        sequences = batch.column(fields_.sequence).to_pylist()
        if "unique_sequences" in metrics:
            metrics["unique_sequences"].update(sequences)
        if "modifications" in metrics:
            mods: list[str] = []
            for sequence in sequences:
                mods.extend(modifications_from_sequence(sequence))
            metrics["modifications"].update(mods)

    if ("unique_unmodified" in metrics or "peptide_length" in metrics) and fields_.unmodified_peptide in present:
        peptides = batch.column(fields_.unmodified_peptide).to_pylist()
        if "unique_unmodified" in metrics:
            metrics["unique_unmodified"].update(peptides)
        if "peptide_length" in metrics:
            metrics["peptide_length"].update(len(peptide) for peptide in peptides if peptide)

    _update_runs(metrics, batch, present, fields_, identity, file_path)
    _update_categorical(metrics, "fragmentation", batch, present, fields_.fragmentation, bundle)
    _update_categorical(metrics, "acquisition", batch, present, fields_.acquisition, bundle)
    _update_categorical(metrics, "detector", batch, present, fields_.detector, bundle)
    _update_categorical(metrics, "enzyme", batch, present, fields_.enzyme, bundle)
    _update_categorical(metrics, "charge", batch, present, fields_.charge, bundle)

    if "analyzer_class" in metrics and fields_.header in present:
        headers = batch.column(fields_.header).to_pylist()
        metrics["analyzer_class"].update(analyzer_class_from_header(header) for header in headers)
    # Organism proxy from protein column: only used when the catalog is not loaded.
    # When the catalog is present, organism is populated once per file in collect_file.
    if "organism" in metrics and not _catalog and fields_.protein in present:
        proteins = batch.column(fields_.protein).to_pylist()
        proxy_values = [organism_from_protein(protein) for protein in proteins]
        metrics["organism"].update(proxy_values)
        bundle.record_source("organism", "protein_proxy", sum(1 for protein in proteins if protein is not None))

    if "collision_energy" in metrics and fields_.collision_energy in present:
        metrics["collision_energy"].update(_to_float(batch.column(fields_.collision_energy)))
    if "precursor_mz" in metrics and fields_.precursor_mz in present:
        metrics["precursor_mz"].update(_to_float(batch.column(fields_.precursor_mz)))

    for column in fields_.score_columns:
        key = f"{_SCORE_PREFIX}{column}"
        if key in metrics and column in present:
            metrics[key].update(_to_float(batch.column(column)))


def _update_categorical(
    metrics: dict[str, Accumulator],
    name: str,
    batch: pa.RecordBatch,
    present: set[str],
    column: str,
    bundle: TierAccumulator,
) -> None:
    """Update a categorical metric from ``column`` if it is enabled and present."""
    if name in metrics and column in present:
        values = batch.column(column).to_pylist()
        non_null = [value for value in values if value is not None]
        if non_null:
            metrics[name].update(non_null)
            bundle.record_source(name, "parquet", len(non_null))


def collect_file(
    task: FileTask,
    metrics: MetricToggles,
    fields_: FieldMap,
    identity: IdentityConfig,
    batch_size: int,
) -> list[tuple[str, str, TierAccumulator]]:
    """Read one parquet file and return statistics keyed by ``(tier, project)``.

    Organism and instrument are sourced from the module-level ``_catalog`` global
    (set by :func:`init_worker_catalog` before this function is called). When the
    catalog is empty, organism falls back to the ``protein``-suffix proxy.

    Args:
        task: The file to process.
        metrics: Which metrics to compute.
        fields_: Logical-to-physical column mapping.
        identity: Project/run resolution rules.
        batch_size: Rows per record batch (bounds peak memory).

    Returns:
        One or more ``(tier, project, accumulator)`` tuples. Row-level project
        resolution may return several tuples when a file mixes projects.
    """
    parquet_file = pq.ParquetFile(task.path)
    available = set(parquet_file.schema_arrow.names)
    if not _is_spectrum_like_schema(available, fields_):
        return []

    if identity.resolves_project_per_row():
        return _collect_by_row_project(task, metrics, fields_, identity, batch_size, parquet_file)
    return [_collect_single_project_file(task, metrics, fields_, identity, batch_size, parquet_file)]


def _collect_single_project_file(
    task: FileTask,
    metrics: MetricToggles,
    fields_: FieldMap,
    identity: IdentityConfig,
    batch_size: int,
    parquet_file: pq.ParquetFile,
) -> tuple[str, str, TierAccumulator]:
    """Collect one file whose project is fixed for all rows."""
    bundle = build_accumulator(metrics, fields_)
    available = frozenset(parquet_file.schema_arrow.names)
    bundle.file_columns = available
    if "projects" in bundle.metrics:
        bundle.metrics["projects"].update([task.project])

    if parquet_file.metadata.num_rows > 0:
        columns = sorted(required_columns(metrics, fields_, identity, catalog_loaded=bool(_catalog)) & set(available))
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            _update_from_batch(bundle, batch, fields_, identity, task.path, task.project)

    _apply_catalog_from_runs(bundle, fields_)
    return task.tier, task.project, bundle


def _collect_by_row_project(
    task: FileTask,
    metrics: MetricToggles,
    fields_: FieldMap,
    identity: IdentityConfig,
    batch_size: int,
    parquet_file: pq.ParquetFile,
) -> list[tuple[str, str, TierAccumulator]]:
    """Collect one file and split accumulators by row-level project identity."""
    project_col = identity.project_field(fields_)
    if project_col is None:
        raise ValueError(
            "Row-level project resolution requires --project-source column|path "
            "and a matching --map project=... or --map filepath=... entry."
        )

    bundles: dict[str, TierAccumulator] = {}
    available = frozenset(parquet_file.schema_arrow.names)
    if parquet_file.metadata.num_rows > 0:
        columns = sorted(required_columns(metrics, fields_, identity, catalog_loaded=bool(_catalog)) & set(available))
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            if project_col not in batch.schema.names:
                bundle = bundles.setdefault(task.project, build_accumulator(metrics, fields_))
                bundle.file_columns = available
                if "projects" in bundle.metrics and bundle.spectra == 0:
                    bundle.metrics["projects"].update([task.project])
                _update_from_batch(bundle, batch, fields_, identity, task.path, task.project)
                continue

            project_values = batch.column(project_col).to_pylist()
            project_rows: dict[str, list[int]] = {}
            for row_idx, value in enumerate(project_values):
                project = resolve_project(value, identity, file_path=task.path)
                project_rows.setdefault(project, []).append(row_idx)

            for project, rows in project_rows.items():
                bundle = bundles.setdefault(project, build_accumulator(metrics, fields_))
                bundle.file_columns = available
                if "projects" in bundle.metrics and bundle.spectra == 0:
                    bundle.metrics["projects"].update([project])
                batch_slice = batch.take(pa.array(rows, type=pa.int64()))
                _update_from_batch(bundle, batch_slice, fields_, identity, task.path, project)

    if not bundles:
        bundle = build_accumulator(metrics, fields_)
        bundle.file_columns = available
        if "projects" in bundle.metrics:
            bundle.metrics["projects"].update([task.project])
        bundles[task.project] = bundle

    out: list[tuple[str, str, TierAccumulator]] = []
    for project, bundle in bundles.items():
        _apply_catalog_from_runs(bundle, fields_)
        out.append((task.tier, project, bundle))
    out.sort(key=lambda item: item[1])
    return out


def _is_spectrum_like_schema(columns: set[str], fields_: FieldMap) -> bool:
    """Skip helper parquet tables that do not contain spectrum rows."""
    markers = {
        fields_.run,
        fields_.usi,
        fields_.header,
        fields_.precursor_mz,
        fields_.charge,
        fields_.fragmentation,
        fields_.sequence,
        fields_.unmodified_peptide,
    }
    return bool(columns & markers)


def _apply_catalog_from_runs(bundle: TierAccumulator, fields_: FieldMap) -> None:
    """Join catalog metadata using per-run ``(project, run)`` keys.

    Parquet columns take precedence when present in the file schema; the catalog
    fills organism, instrument, acquisition, fragmentation, detector, and enzyme otherwise.
    """
    if not _catalog or bundle.spectra == 0 or not bundle.catalog_run_spectra:
        return
    metrics = bundle.metrics
    for key, spectra in bundle.catalog_run_spectra.items():
        entry = _catalog.get(key)
        if entry is None:
            bundle.catalog_files_missed += 1
            continue
        bundle.catalog_files_matched += 1
        if "organism" in metrics and entry.organisms and _uses_catalog_for_metric("organism", fields_, bundle.file_columns):
            metrics["organism"].update({org: spectra for org in entry.organisms})
            bundle.record_source("organism", "catalog", spectra * len(entry.organisms))
        if "instrument" in metrics and entry.instrument and _uses_catalog_for_metric("instrument", fields_, bundle.file_columns):
            metrics["instrument"].update({entry.instrument: spectra})
            bundle.record_source("instrument", "catalog", spectra)
        if "acquisition" in metrics and entry.acquisition and _uses_catalog_for_metric("acquisition", fields_, bundle.file_columns):
            metrics["acquisition"].update({entry.acquisition: spectra})
            bundle.record_source("acquisition", "catalog", spectra)
        if "fragmentation" in metrics and entry.fragmentations and _uses_catalog_for_metric("fragmentation", fields_, bundle.file_columns):
            metrics["fragmentation"].update({frag: spectra for frag in entry.fragmentations})
            bundle.record_source("fragmentation", "catalog", spectra * len(entry.fragmentations))
        if "detector" in metrics and entry.detectors and _uses_catalog_for_metric("detector", fields_, bundle.file_columns):
            metrics["detector"].update({det: spectra for det in entry.detectors})
            bundle.record_source("detector", "catalog", spectra * len(entry.detectors))
        if "enzyme" in metrics and entry.enzyme and _uses_catalog_for_metric("enzyme", fields_, bundle.file_columns):
            metrics["enzyme"].update({entry.enzyme: spectra})
            bundle.record_source("enzyme", "catalog", spectra)


def _length_summary(accumulator: CategoricalAccumulator) -> dict[str, Any]:
    """Summarise a peptide-length frequency table (count/mean/std/min/max + histogram)."""
    counts = accumulator.counts
    total_n = sum(counts.values())
    if total_n == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "histogram": {}}
    weighted_sum = sum(length * freq for length, freq in counts.items())
    mean = weighted_sum / total_n
    variance = sum(freq * (length - mean) ** 2 for length, freq in counts.items()) / total_n
    return {
        "count": total_n,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(counts),
        "max": max(counts),
        "histogram": {str(length): int(counts[length]) for length in sorted(counts)},
    }


def render(bundle: TierAccumulator, keep_top: int | None) -> dict[str, Any]:
    """Render an accumulator bundle into a plain, JSON-serialisable dictionary."""
    metrics = bundle.metrics
    out: dict[str, Any] = {"spectra": bundle.spectra}

    for name in ("projects", "runs", "unique_sequences", "unique_unmodified"):
        if name in metrics:
            out[name] = metrics[name].count  # type: ignore[union-attr]
    if "peptide_length" in metrics:
        out["peptide_length"] = _length_summary(metrics["peptide_length"])  # type: ignore[arg-type]
    for name in (
        "modifications",
        "fragmentation",
        "acquisition",
        "charge",
        "analyzer_class",
        "organism",
        "instrument",
        "detector",
        "enzyme",
    ):
        if name in metrics:
            out[name] = metrics[name].result(keep_top)  # type: ignore[call-arg]
    for name in ("collision_energy", "precursor_mz"):
        if name in metrics:
            out[name] = metrics[name].result()

    scores = {
        name[len(_SCORE_PREFIX):]: metrics[name].result()
        for name in metrics
        if name.startswith(_SCORE_PREFIX)
    }
    if scores:
        out["confidence_scores"] = scores
    return out
