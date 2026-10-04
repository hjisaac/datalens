"""Discovery and parallel orchestration of a dataset statistics run.

The runner discovers every parquet file under the configured tiers, fans the files
out across a process pool (one task per file), and reduces the returned partial
accumulators into per-project and per-tier totals (a classic map-reduce).

Process-based parallelism is deliberate: the heavy per-file work (building distinct
sequence sets, counting modifications, tallying categories) happens in Python and is
therefore bound by the GIL, so threads would serialise. Separate processes sidestep
the GIL and scale across cores. Set ``workers=1`` to run serially in-process.

The orchestration is intentionally verbose at ``INFO`` level: it echoes the resolved
configuration, the per-tier discovery breakdown, live progress (throughput, running
spectrum count and ETA), per-tier completion, and a rich per-tier summary at the end,
so a reader can follow exactly what happened from the logs alone.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from instanovo.dataset_stats.catalog import CatalogEntry, RunKey, load_catalog
from instanovo.dataset_stats.collector import (
    CATALOG_FALLBACK_METRICS,
    FileTask,
    TierAccumulator,
    build_accumulator,
    collect_file,
    init_worker_catalog,
    render,
    required_columns,
)
from instanovo.dataset_stats.config import FieldMap, MAPPABLE_SCALAR_FIELDS, StatsConfig
from instanovo.dataset_stats.identity import UNRESOLVED, discovery_project

logger = logging.getLogger(__name__)

# Minimum seconds between live progress lines (also logged on each 10% boundary).
_PROGRESS_INTERVAL_S = 60.0


@dataclass
class StatisticsResult:
    """The outcome of a statistics run.

    Attributes:
        tiers: Mapping of tier name to its rendered statistics (including a
            ``projects_detail`` breakdown when ``per_project`` was enabled).
        files_processed: Number of parquet files successfully read.
        files_failed: Number of parquet files that raised an error.
        config_summary: A JSON-friendly echo of the configuration used.
    """

    tiers: dict[str, Any]
    files_processed: int
    files_failed: int
    config_summary: dict[str, Any]
    notes: dict[str, str] | None = None
    field_sources: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dictionary."""
        out: dict[str, Any] = {
            "config": self.config_summary,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "tiers": self.tiers,
        }
        if self.field_sources:
            out["field_sources"] = self.field_sources
        if self.notes:
            out["notes"] = self.notes
        return out

    def to_json(self, path: str | os.PathLike[str], indent: int = 2) -> None:
        """Write the result to ``path`` as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")


def discover_tasks(config: StatsConfig) -> list[FileTask]:
    """Enumerate every parquet file under the configured tiers.

    Walks each tier directory recursively — layout depth does not matter; every
    ``.parquet`` file found is included.

    Args:
        config: The run configuration.

    Returns:
        A deterministically ordered list of :class:`FileTask` entries.
    """
    tasks: list[FileTask] = []
    identity = config.identity
    for tier in config.tiers:
        tier_dir = config.root / tier
        if not tier_dir.is_dir():
            logger.warning("Tier directory not found, skipping: %s", tier_dir)
            continue
        for root, _, files in os.walk(tier_dir):
            for name in sorted(files):
                if not name.endswith(".parquet"):
                    continue
                path = str(Path(root) / name)
                project = discovery_project(path, identity)
                tasks.append(FileTask(tier=tier, project=project, path=path))
    return tasks


def run_dataset_statistics(config: StatsConfig) -> StatisticsResult:
    """Compute corpus statistics for the configured tiers.

    Args:
        config: The run configuration.

    Returns:
        A :class:`StatisticsResult` with per-tier (and optionally per-project) stats.
    """
    workers = config.resolved_workers()
    _log_config(config, workers)

    logger.info("Discovering parquet files under %s ...", config.root)
    discovery_start = time.monotonic()
    tasks = discover_tasks(config)
    tier_totals = _log_discovery(tasks, time.monotonic() - discovery_start)

    if not tasks:
        logger.warning("No parquet files found; nothing to do.")
        return StatisticsResult({}, 0, 0, _config_summary(config, workers))

    catalog: dict[RunKey, CatalogEntry] = {}
    if config.catalog is not None:
        logger.info("Loading metadata catalog from %s ...", config.catalog)
        catalog = load_catalog(config.catalog)
        logger.info(
            "Catalog loaded: %s run entries covering %s projects. "
            "Run-level metadata (organism, instrument, acquisition, fragmentation, "
            "detector, enzyme) will use parquet columns when present, otherwise the catalog.",
            _fmt_int(len(catalog)),
            _fmt_int(len({p for p, _ in catalog})),
        )
    else:
        logger.info(
            "No catalog provided — organism uses protein-suffix proxy when available; "
            "instrument and catalog-only fields will be empty."
        )

    _warn_missing_columns(config, tasks, catalog_loaded=bool(catalog))
    project_totals = _project_file_totals(tasks)
    progress = _Progress(
        total=len(tasks),
        tier_totals=tier_totals,
        project_totals=project_totals,
        enabled=config.progress,
    )
    logger.info(
        "Processing %s files across %s projects with %d worker(s) ...",
        _fmt_int(len(tasks)),
        _fmt_int(len(project_totals)),
        workers,
    )

    results: list[tuple[str, str, TierAccumulator]] = []
    if workers == 1:
        # Serial path: set the catalog global directly in this process.
        init_worker_catalog(catalog)
        for task in tasks:
            try:
                outcomes = collect_file(task, config.metrics, config.fields, config.identity, config.batch_size)
            except Exception:
                logger.exception("Failed to process %s", task.path)
                progress.record(task.tier, task.project, spectra=None)
                continue
            if not outcomes:
                progress.record(task.tier, task.project, spectra=0)
                continue
            results.extend(outcomes)
            progress.record(task.tier, task.project, spectra=sum(bundle.spectra for _, _, bundle in outcomes))
    else:
        # Parallel path: use the pool initializer so the catalog is set once per
        # worker process rather than being pickled with every task submission.
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker_catalog, initargs=(catalog,)) as executor:
            futures = {
                executor.submit(
                    collect_file, task, config.metrics, config.fields, config.identity, config.batch_size
                ): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    outcomes = future.result()
                except Exception:
                    logger.exception("Failed to process %s", task.path)
                    progress.record(task.tier, task.project, spectra=None)
                    continue
                if not outcomes:
                    progress.record(task.tier, task.project, spectra=0)
                    continue
                results.extend(outcomes)
                progress.record(task.tier, task.project, spectra=sum(bundle.spectra for _, _, bundle in outcomes))

    logger.info("Reducing per-file results into per-tier totals ...")
    tiers = _reduce(config, results)

    progress.log_final()
    _log_summary(config, tiers)
    notes = _build_notes(config, results, catalog_loaded=bool(catalog))
    field_sources = _build_field_sources(config, results, catalog_loaded=bool(catalog))
    if notes:
        for key, value in notes.items():
            logger.info("note: %s = %s", key, value)
    if field_sources:
        for key, value in field_sources.items():
            logger.info("field_source: %s = %s", key, value)
    return StatisticsResult(
        tiers=tiers,
        files_processed=progress.processed,
        files_failed=progress.failed,
        config_summary=_config_summary(config, workers),
        notes=notes or None,
        field_sources=field_sources or None,
    )


def _reduce(
    config: StatsConfig,
    results: list[tuple[str, str, TierAccumulator]],
) -> dict[str, Any]:
    """Reduce per-file accumulators into rendered per-tier (and per-project) statistics."""
    project_acc: dict[tuple[str, str], TierAccumulator] = {}
    for tier, project, bundle in results:
        key = (tier, project)
        if key not in project_acc:
            project_acc[key] = build_accumulator(config.metrics, config.fields)
        project_acc[key].merge(bundle)

    tier_acc: dict[str, TierAccumulator] = {}
    tier_projects: dict[str, dict[str, TierAccumulator]] = {}
    for (tier, project), bundle in project_acc.items():
        tier_acc.setdefault(tier, build_accumulator(config.metrics, config.fields)).merge(bundle)
        tier_projects.setdefault(tier, {})[project] = bundle

    rendered: dict[str, Any] = {}
    for tier in config.tiers:
        if tier not in tier_acc:
            continue
        tier_view = render(tier_acc[tier], config.keep_top_categories)
        if config.per_project:
            tier_view["projects_detail"] = {
                project: render(bundle, config.keep_top_categories)
                for project, bundle in sorted(tier_projects[tier].items())
            }
        rendered[tier] = tier_view
    return rendered


def _project_file_totals(tasks: list[FileTask]) -> dict[tuple[str, str], int]:
    """Count how many files each ``(tier, project)`` contains (for project progress)."""
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for task in tasks:
        totals[(task.tier, task.project)] += 1
    return dict(totals)


class _Progress:
    """Tracks and logs live progress: project/file completion, throughput and ETA.

    Progress is reported on two axes: file-based (one unit per parquet file) and
    project-based (a project counts as done once all of its files are attempted).
    When ``enabled`` is ``False`` the live lines are suppressed while the underlying
    counters are still maintained for the final summary.
    """

    def __init__(
        self,
        total: int,
        tier_totals: dict[str, int],
        project_totals: dict[tuple[str, str], int],
        enabled: bool = True,
    ) -> None:
        self.total = total
        self.tier_totals = tier_totals
        self.enabled = enabled
        self.processed = 0
        self.failed = 0
        self.spectra = 0
        self.tier_done: dict[str, int] = defaultdict(int)
        self._project_remaining = {
            key: count for key, count in project_totals.items() if key[1] != UNRESOLVED
        }
        self.total_projects = len(self._project_remaining)
        self.projects_done = 0
        self.start = time.monotonic()
        self._last_log = self.start
        self._next_pct = 10

    def record(self, tier: str, project: str, spectra: int | None) -> None:
        """Record one finished file (``spectra=None`` marks a failure) and maybe log."""
        if spectra is None:
            self.failed += 1
        else:
            self.processed += 1
            self.spectra += spectra
            self.tier_done[tier] += 1
            if self.tier_done[tier] == self.tier_totals.get(tier):
                logger.info("Tier '%s' complete: %s files.", tier, _fmt_int(self.tier_totals[tier]))
        key = (tier, project)
        if project != UNRESOLVED and key in self._project_remaining:
            self._project_remaining[key] -= 1
            if self._project_remaining[key] == 0:
                self.projects_done += 1
                logger.info(
                    "Project complete [%s/%s] %s/%s — %s spectra so far.",
                    tier,
                    project,
                    _fmt_int(self.projects_done),
                    _fmt_int(self.total_projects),
                    _humanize(self.spectra),
                )
        if spectra is None:
            logger.warning(
                "File failed (%d total failures so far) — %s/%s files attempted.",
                self.failed,
                _fmt_int(self.processed + self.failed),
                _fmt_int(self.total),
            )
        self._maybe_log()

    def _maybe_log(self) -> None:
        """Emit a progress line on a time interval or a 5%% boundary."""
        if not self.enabled:
            return
        attempted = self.processed + self.failed
        now = time.monotonic()
        pct = 100.0 * attempted / max(1, self.total)
        crossed_pct = pct >= self._next_pct
        if not (crossed_pct or now - self._last_log >= _PROGRESS_INTERVAL_S or attempted == self.total):
            return
        self._last_log = now
        if crossed_pct:
            self._next_pct = int(pct // 10) * 10 + 10
        elapsed = now - self.start
        rate = attempted / elapsed if elapsed > 0 else 0.0
        eta = (self.total - attempted) / rate if rate > 0 else 0.0
        project_pct = 100.0 * self.projects_done / max(1, self.total_projects)
        logger.info(
            "Progress | projects %s/%s (%.0f%%) | files %s/%s (%.0f%%) | %s spectra | %.0f files/s | elapsed %s | ETA %s%s",
            _fmt_int(self.projects_done),
            _fmt_int(self.total_projects),
            project_pct,
            _fmt_int(attempted),
            _fmt_int(self.total),
            pct,
            _humanize(self.spectra),
            rate,
            _dur(elapsed),
            _dur(eta),
            f" | {self.failed} failed" if self.failed else "",
        )

    def log_final(self) -> None:
        """Log the end-of-run throughput line."""
        elapsed = time.monotonic() - self.start
        rate = self.processed / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Done: %s files processed, %s failed, %s spectra in %s (%.0f files/s).",
            _fmt_int(self.processed),
            _fmt_int(self.failed),
            _fmt_int(self.spectra),
            _dur(elapsed),
            rate,
        )


def _log_config(config: StatsConfig, workers: int) -> None:
    """Log the resolved configuration that is about to be used."""
    enabled = config.metrics.enabled_names()
    disabled = tuple(name for name in vars(config.metrics) if name not in enabled)
    logger.info("=== Dataset statistics run ===")
    logger.info("root=%s | tiers=%s | workers=%d", config.root, ", ".join(config.tiers), workers)
    logger.info("per_project=%s | progress=%s | batch_size=%s | keep_top_categories=%s", config.per_project, config.progress, _fmt_int(config.batch_size), config.keep_top_categories)
    logger.info("catalog=%s", config.catalog or "none (organism/analyzer_class will use data-derived proxies)")
    logger.info(
        "identity | project=%s | run=%s",
        _identity_summary(config.identity, config.fields, "project"),
        _identity_summary(config.identity, config.fields, "run"),
    )
    logger.info("field_map overrides: %s", _field_map_summary(config.fields))
    logger.info("metrics enabled (%d): %s", len(enabled), ", ".join(enabled))
    if disabled:
        logger.info("metrics disabled (%d): %s", len(disabled), ", ".join(disabled))
    if config.metrics.unique_sequences or config.metrics.unique_unmodified:
        logger.info("note: unique_sequences/unmodified use in-memory distinct sets (high RAM on large corpora).")
    if not config.catalog and (config.metrics.organism or config.metrics.analyzer_class):
        logger.info("note: without a catalog, 'organism' and 'analyzer_class' are data-derived proxies (protein suffix / header).")


def _build_notes(
    config: StatsConfig,
    results: list[tuple[str, str, TierAccumulator]],
    *,
    catalog_loaded: bool,
) -> dict[str, str]:
    """Short interpretive notes for the JSON output."""
    notes: dict[str, str] = {}
    identity = config.identity
    if config.metrics.runs:
        if identity.run_source in ("column", "auto"):
            notes["runs"] = f"column:{config.fields.run}"
        elif identity.run_source == "filename":
            notes["runs"] = "parquet filename stem"
        else:
            notes["runs"] = f"path via {identity.run_field(config.fields) or 'file path'}"
    if catalog_loaded and (config.metrics.organism or config.metrics.instrument):
        matched = sum(bundle.catalog_files_matched for _, _, bundle in results)
        missed = sum(bundle.catalog_files_missed for _, _, bundle in results)
        total = matched + missed
        if total:
            notes["catalog_match"] = f"{matched}/{total}"
        notes["organism_counts"] = "per-run via catalog; multi-organism may exceed spectra"
    return notes


def _build_field_sources(
    config: StatsConfig,
    results: list[tuple[str, str, TierAccumulator]],
    *,
    catalog_loaded: bool,
) -> dict[str, str]:
    """Describe where each enabled metric's values came from (parquet vs catalog)."""
    fields = config.fields
    metrics = config.metrics
    catalog_label = str(config.catalog) if config.catalog else "search_data.csv"

    aggregated: dict[str, Counter[str]] = defaultdict(Counter)
    for _, _, bundle in results:
        for metric, counts in bundle.source_counts.items():
            aggregated[metric].update(counts)

    def _describe(metric: str, column: str | None = None) -> str | None:
        if not getattr(metrics, metric, False):
            return None
        counts = aggregated.get(metric, Counter())
        parts: list[str] = []
        parquet_n = counts.get("parquet", 0)
        catalog_n = counts.get("catalog", 0)
        proxy_n = counts.get("protein_proxy", 0)
        total = parquet_n + catalog_n + proxy_n
        if parquet_n:
            label = f"parquet (column: {column or metric})"
            if catalog_n or proxy_n:
                label += f", {100.0 * parquet_n / total:.0f}% of attributed spectra"
            parts.append(label)
        if catalog_n:
            label = f"catalog ({catalog_label})"
            if parquet_n or proxy_n:
                label += f", {100.0 * catalog_n / total:.0f}% of attributed spectra"
            parts.append(label)
        if proxy_n:
            label = "parquet (protein suffix proxy)"
            if parquet_n or catalog_n:
                label += f", {100.0 * proxy_n / total:.0f}% of attributed spectra"
            parts.append(label)
        if parts:
            return " + ".join(parts)
        if catalog_loaded and metric in CATALOG_FALLBACK_METRICS:
            return f"catalog ({catalog_label}) when parquet column absent; no data matched"
        if metric == "organism":
            return "parquet (protein suffix proxy) when catalog absent"
        if column:
            return f"parquet (column: {column})"
        return "unavailable"

    parquet_columns: dict[str, str | None] = {
        "acquisition": fields.acquisition,
        "fragmentation": fields.fragmentation,
        "detector": fields.detector,
        "enzyme": fields.enzyme,
        "charge": fields.charge,
        "collision_energy": fields.collision_energy,
        "precursor_mz": fields.precursor_mz,
        "analyzer_class": fields.header,
        "unique_sequences": fields.sequence,
        "unique_unmodified": fields.unmodified_peptide,
        "modifications": fields.sequence,
        "peptide_length": fields.unmodified_peptide,
        "runs": fields.run,
    }

    sources: dict[str, str] = {}
    for metric, column in parquet_columns.items():
        desc = _describe(metric, column)
        if desc:
            sources[metric] = desc
    for metric in ("organism", "instrument", "detector", "enzyme"):
        desc = _describe(metric)
        if desc:
            sources[metric] = desc
    if metrics.confidence_scores:
        score_cols = ", ".join(fields.score_columns)
        sources["confidence_scores"] = f"parquet (columns: {score_cols})"
    if metrics.spectra:
        sources["spectra"] = "parquet (row count)"
    if metrics.projects:
        sources["projects"] = "path or column identity rules"
    return sources


def _warn_missing_columns(
    config: StatsConfig,
    tasks: list[FileTask],
    *,
    catalog_loaded: bool = False,
) -> None:
    """Warn (once per tier) if an enabled metric needs a column absent from the data.

    The schema is sampled from the first file of each tier (a cheap footer-only read).
    This surfaces genuine data problems instead of silently reporting zero for a metric
    whose source column is missing.
    """
    required = required_columns(config.metrics, config.fields, config.identity, catalog_loaded=catalog_loaded)
    if catalog_loaded:
        # acquisition / fragmentation may come from the catalog when absent in parquet.
        fields = config.fields
        if config.metrics.acquisition:
            required.discard(fields.acquisition)
        if config.metrics.fragmentation:
            required.discard(fields.fragmentation)
        if config.metrics.detector:
            required.discard(fields.detector)
        if config.metrics.enzyme:
            required.discard(fields.enzyme)
    if not required:
        return
    first_task: dict[str, FileTask] = {}
    for task in tasks:
        first_task.setdefault(task.tier, task)
    for tier, task in first_task.items():
        try:
            available = set(pq.read_schema(task.path).names)
        except Exception:
            logger.exception("Could not read schema of sample file %s", task.path)
            continue
        missing = sorted(required - available)
        if missing:
            logger.warning(
                "Tier '%s': enabled metrics reference column(s) absent from sample file '%s': %s "
                "(these metrics will be empty for files sharing this schema).",
                tier,
                Path(task.path).name,
                ", ".join(missing),
            )


def _log_discovery(tasks: list[FileTask], elapsed: float) -> dict[str, int]:
    """Log the per-tier file/project breakdown and return per-tier file counts."""
    tier_files: dict[str, int] = defaultdict(int)
    tier_projects: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        tier_files[task.tier] += 1
        tier_projects[task.tier].add(task.project)
    logger.info("Discovered %s files in %s:", _fmt_int(len(tasks)), _dur(elapsed))
    for tier in tier_files:
        logger.info("  %-6s %s projects | %s files", tier, _fmt_int(len(tier_projects[tier])), _fmt_int(tier_files[tier]))
    return dict(tier_files)


def _log_summary(config: StatsConfig, tiers: dict[str, Any]) -> None:
    """Log a compact, information-dense per-tier summary of the computed statistics."""
    logger.info("=== Summary ===")
    for tier, view in tiers.items():
        logger.info("[%s]", tier)
        headline = [f"spectra={_fmt_int(view['spectra'])}"]
        for key in ("projects", "runs", "unique_sequences", "unique_unmodified"):
            if key in view:
                headline.append(f"{key}={_fmt_int(view[key])}")
        logger.info("  %s", " | ".join(headline))
        if "peptide_length" in view and view["peptide_length"]["count"]:
            length = view["peptide_length"]
            logger.info("  peptide_length: mean=%.1f std=%.1f range=[%s-%s]", length["mean"], length["std"], length["min"], length["max"])
        for key in (
            "fragmentation",
            "acquisition",
            "analyzer_class",
            "charge",
            "organism",
            "instrument",
            "detector",
            "enzyme",
            "modifications",
        ):
            if key in view:
                logger.info("  %-13s unique=%s top=%s", key, _fmt_int(view[key]["unique"]), _top(view[key]["counts"]))
        if "confidence_scores" in view:
            parts = [f"{name}(mean={stat['mean']:.2f})" for name, stat in view["confidence_scores"].items() if stat["count"]]
            if parts:
                logger.info("  scores: %s", " | ".join(parts))


def _top(counts: dict[str, int], limit: int = 5) -> str:
    """Render the top categories of a frequency dict as a compact string."""
    if not counts:
        return "(none)"
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    rendered = ", ".join(f"{key}={_humanize(value)}" for key, value in items)
    return rendered + (" ..." if len(counts) > limit else "")


def _config_summary(config: StatsConfig, workers: int) -> dict[str, Any]:
    """Build a JSON-friendly echo of the configuration that produced a result."""
    return {
        "root": str(config.root),
        "tiers": list(config.tiers),
        "workers": workers,
        "per_project": config.per_project,
        "progress": config.progress,
        "catalog": str(config.catalog) if config.catalog else None,
        "batch_size": config.batch_size,
        "keep_top_categories": config.keep_top_categories,
        "enabled_metrics": list(config.metrics.enabled_names()),
        "identity": {
            "project_source": config.identity.project_source,
            "run_source": config.identity.run_source,
            "project_path_index": config.identity.project_path_index,
            "run_path_index": config.identity.run_path_index,
            "project_pattern": config.identity.project_pattern,
            "run_pattern": config.identity.run_pattern,
        },
        "field_map": {
            name: getattr(config.fields, name)
            for name in MAPPABLE_SCALAR_FIELDS
            if getattr(config.fields, name) != getattr(FieldMap(), name)
        },
    }


def _identity_summary(identity, fields: FieldMap, kind: str) -> str:
    """Render a compact identity rule for logging."""
    if kind == "project":
        source = identity.project_source
        column = identity.project_field(fields)
        index = identity.project_path_index
    else:
        source = identity.run_source
        column = identity.run_field(fields)
        index = identity.run_path_index
    parts = [f"source={source}"]
    if column:
        parts.append(f"column={column}")
    if index is not None:
        parts.append(f"path_index={index}")
    return " ".join(parts)


def _field_map_summary(fields: FieldMap) -> str:
    """Render non-default logical→physical column overrides."""
    overrides = {
        name: getattr(fields, name)
        for name in MAPPABLE_SCALAR_FIELDS
        if getattr(fields, name) != getattr(FieldMap(), name)
    }
    return ", ".join(f"{logical}={physical}" for logical, physical in sorted(overrides.items())) or "(defaults)"


def _fmt_int(value: int) -> str:
    """Format an integer with thousands separators."""
    return f"{value:,}"


def _humanize(value: float) -> str:
    """Format a large count compactly (e.g. ``8.2M``)."""
    number = float(value)
    for unit in ("", "K", "M", "B"):
        if abs(number) < 1000:
            return f"{int(number)}{unit}" if unit == "" else f"{number:.1f}{unit}"
        number /= 1000
    return f"{number:.1f}T"


def _dur(seconds: float) -> str:
    """Format a duration compactly (``s`` / ``m`` / ``h``)."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds // 60:.0f}m{seconds % 60:.0f}s"
    return f"{seconds // 3600:.0f}h{(seconds % 3600) // 60:.0f}m"
