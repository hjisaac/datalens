#!/usr/bin/env python
"""Independent row-by-row reference verification for instanovo.dataset_stats.

This script does NOT import collector/runner internals for computing stats.
It re-implements the metric logic by reading every parquet row directly,
then compares against the module output for workers=1,2,4,8.

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from instanovo.dataset_stats.catalog import (
    load_catalog,
    normalize_run_name,
    parquet_run_name,
)
from instanovo.dataset_stats.config import FieldMap, MetricToggles, StatsConfig
from instanovo.dataset_stats.identity import (
    IdentityConfig,
    UNRESOLVED,
    discovery_project,
    resolve_project,
    resolve_run,
)
from instanovo.dataset_stats.parsing import (
    analyzer_class_from_header,
    modifications_from_sequence,
    organism_from_protein,
)
from instanovo.dataset_stats.runner import discover_tasks, run_dataset_statistics

FM = Path("/app/fm_mount")
CATALOG = Path("/app/search_data.csv")
SCORE_PREFIX = "score:"
SPECTRUM_MARKERS = {
    "experiment_name",
    "usi",
    "header",
    "precursor_mz",
    "precursor_charge",
    "frag_type",
    "sequence",
    "unmodified_peptide",
}


def _is_spectrum_like(columns: set[str], fields: FieldMap) -> bool:
    markers = {
        fields.run,
        fields.usi,
        fields.header,
        fields.precursor_mz,
        fields.charge,
        fields.fragmentation,
        fields.sequence,
        fields.unmodified_peptide,
    }
    return bool(columns & markers)


def _resolve_run_for_row(
    row: dict[str, Any],
    fields: FieldMap,
    identity: IdentityConfig,
    file_path: str,
) -> Any:
    """Mirror collector._update_runs row semantics exactly."""
    present = set(row.keys())
    source = identity.run_source
    # auto + run column: use raw column values (not resolve_run).
    if source == "column" or (source == "auto" and fields.run in present):
        if fields.run in present:
            return row.get(fields.run)
        return None
    if source in ("path", "auto"):
        run_col = identity.run_field(fields)
        if run_col and run_col in present:
            return resolve_run(row.get(run_col), identity, file_path=file_path)
    return resolve_run(None, identity, file_path=file_path)


def _catalog_key(
    project: str,
    row: dict[str, Any],
    fields: FieldMap,
    file_path: str,
) -> tuple[str, str] | None:
    if fields.run in row and row[fields.run] is not None:
        run = normalize_run_name(str(row[fields.run]))
    else:
        run = parquet_run_name(file_path)
    if not run:
        return None
    return (project, run)


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    finite = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.array(finite, dtype=np.float64)
    mean = float(arr.mean())
    return {
        "count": len(finite),
        "mean": mean,
        "std": float(np.sqrt(np.mean((arr - mean) ** 2))),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _length_summary(lengths: Counter[int]) -> dict[str, Any]:
    total_n = sum(lengths.values())
    if total_n == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "histogram": {}}
    weighted_sum = sum(length * freq for length, freq in lengths.items())
    mean = weighted_sum / total_n
    variance = sum(freq * (length - mean) ** 2 for length, freq in lengths.items()) / total_n
    return {
        "count": total_n,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(lengths),
        "max": max(lengths),
        "histogram": {str(k): int(lengths[k]) for k in sorted(lengths)},
    }


class ReferenceStats:
    """Row-by-row independent stats accumulator."""

    def __init__(
        self,
        metrics: MetricToggles,
        fields: FieldMap,
        identity: IdentityConfig,
        catalog: dict | None,
    ) -> None:
        self.metrics = metrics
        self.fields = fields
        self.identity = identity
        self.catalog = catalog or {}
        self.spectra = 0
        self.projects: set[str] = set()
        self.runs: set[Any] = set()
        self.unique_sequences: set[str] = set()
        self.unique_unmodified: set[str] = set()
        self.peptide_lengths: Counter[int] = Counter()
        self.modifications: Counter[str] = Counter()
        self.fragmentation: Counter[str] = Counter()
        self.acquisition: Counter[str] = Counter()
        self.charge: Counter[str] = Counter()
        self.analyzer_class: Counter[str] = Counter()
        self.organism: Counter[str] = Counter()
        self.instrument: Counter[str] = Counter()
        self.collision_energy: list[float] = []
        self.precursor_mz: list[float] = []
        self.scores: dict[str, list[float]] = {c: [] for c in fields.score_columns}
        self.catalog_run_spectra: Counter[tuple[str, str]] = Counter()

    def ingest_row(
        self,
        row: dict[str, Any],
        *,
        file_path: str,
        file_project: str,
    ) -> None:
        present = set(row.keys())
        if not _is_spectrum_like(present, self.fields):
            return

        if self.identity.resolves_project_per_row():
            project_col = self.identity.project_field(self.fields)
            if project_col and project_col in present:
                project = resolve_project(row.get(project_col), self.identity, file_path=file_path)
            else:
                project = file_project
        else:
            project = file_project

        self.spectra += 1
        if self.metrics.projects:
            self.projects.add(project)

        key = _catalog_key(project, row, self.fields, file_path)
        if key and (self.metrics.organism or self.metrics.instrument):
            self.catalog_run_spectra[key] += 1

        f = self.fields
        if self.metrics.unique_sequences and f.sequence in present:
            seq = row.get(f.sequence)
            if seq is not None:
                self.unique_sequences.add(seq)
        if self.metrics.modifications and f.sequence in present:
            self.modifications.update(modifications_from_sequence(row.get(f.sequence)))
        if self.metrics.unique_unmodified and f.unmodified_peptide in present:
            pep = row.get(f.unmodified_peptide)
            if pep is not None:
                self.unique_unmodified.add(pep)
        if self.metrics.peptide_length and f.unmodified_peptide in present:
            pep = row.get(f.unmodified_peptide)
            if pep:
                self.peptide_lengths[len(pep)] += 1

        if self.metrics.runs:
            run = _resolve_run_for_row(row, f, self.identity, file_path)
            if run is not None:
                self.runs.add(run)

        for name, col, bucket in (
            ("fragmentation", f.fragmentation, self.fragmentation),
            ("acquisition", f.acquisition, self.acquisition),
            ("charge", f.charge, self.charge),
        ):
            if getattr(self.metrics, name) and col in present:
                val = row.get(col)
                if val is not None:
                    bucket[str(val)] += 1

        if self.metrics.analyzer_class and f.header in present:
            self.analyzer_class[analyzer_class_from_header(row.get(f.header))] += 1

        if self.metrics.organism and not self.catalog and f.protein in present:
            self.organism[organism_from_protein(row.get(f.protein))] += 1

        def _float(val: Any) -> float | None:
            if val is None:
                return None
            try:
                x = float(val)
                return None if math.isnan(x) else x
            except (TypeError, ValueError):
                return None

        if self.metrics.collision_energy and f.collision_energy in present:
            v = _float(row.get(f.collision_energy))
            if v is not None:
                self.collision_energy.append(v)
        if self.metrics.precursor_mz and f.precursor_mz in present:
            v = _float(row.get(f.precursor_mz))
            if v is not None:
                self.precursor_mz.append(v)
        if self.metrics.confidence_scores:
            for col in f.score_columns:
                if col in present:
                    v = _float(row.get(col))
                    if v is not None:
                        self.scores[col].append(v)

    def finalize_catalog(self) -> None:
        if not self.catalog:
            return
        for (project, run), count in self.catalog_run_spectra.items():
            entry = self.catalog.get((project, run))
            if entry is None:
                continue
            if self.metrics.organism and entry.organisms:
                for org in entry.organisms:
                    self.organism[org] += count
            if self.metrics.instrument and entry.instrument:
                self.instrument[entry.instrument] += count

    def render(self) -> dict[str, Any]:
        self.finalize_catalog()
        out: dict[str, Any] = {"spectra": self.spectra}
        if self.metrics.projects:
            out["projects"] = len(self.projects)
        if self.metrics.runs:
            out["runs"] = len(self.runs)
        if self.metrics.unique_sequences:
            out["unique_sequences"] = len(self.unique_sequences)
        if self.metrics.unique_unmodified:
            out["unique_unmodified"] = len(self.unique_unmodified)
        if self.metrics.peptide_length:
            out["peptide_length"] = _length_summary(self.peptide_lengths)
        for name, counter in (
            ("modifications", self.modifications),
            ("fragmentation", self.fragmentation),
            ("acquisition", self.acquisition),
            ("charge", self.charge),
            ("analyzer_class", self.analyzer_class),
            ("organism", self.organism),
            ("instrument", self.instrument),
        ):
            if getattr(self.metrics, name):
                out[name] = {
                    "unique": len(counter),
                    "counts": {str(k): int(v) for k, v in counter.items()},
                }
        if self.metrics.collision_energy:
            out["collision_energy"] = _numeric_summary(self.collision_energy)
        if self.metrics.precursor_mz:
            out["precursor_mz"] = _numeric_summary(self.precursor_mz)
        if self.metrics.confidence_scores:
            scores = {col: _numeric_summary(vals) for col, vals in self.scores.items() if vals or True}
            out["confidence_scores"] = {k: v for k, v in scores.items() if v["count"] > 0}
            if not out["confidence_scores"]:
                del out["confidence_scores"]
        return out


def _read_rows_line_by_line(path: str, columns: list[str] | None = None) -> list[dict[str, Any]]:
    """Read every row individually via batch_size=1 (true row-at-a-time)."""
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    if not _is_spectrum_like(available, FieldMap()):
        return []
    use_cols = sorted(set(columns or []) & available) if columns else None
    rows: list[dict[str, Any]] = []
    for batch in pf.iter_batches(batch_size=1, columns=use_cols):
        for i in range(batch.num_rows):
            rows.append({name: batch.column(name)[i].as_py() for name in batch.schema.names})
    return rows


def _needed_columns(metrics: MetricToggles, fields: FieldMap, identity: IdentityConfig, catalog: bool) -> set[str]:
    from instanovo.dataset_stats.collector import required_columns

    return required_columns(metrics, fields, identity, catalog_loaded=catalog)


def compute_reference(
    tasks: list,
    metrics: MetricToggles,
    fields: FieldMap,
    identity: IdentityConfig,
    catalog_path: Path | None,
) -> dict[str, Any]:
    """Compute tier-level reference stats by reading every row of every file."""
    catalog = load_catalog(catalog_path) if catalog_path else {}
    tier_stats: dict[str, ReferenceStats] = {}
    tier_project_stats: dict[str, dict[str, ReferenceStats]] = defaultdict(dict)

    for task in tasks:
        tier = task.tier
        tier_stats.setdefault(tier, ReferenceStats(metrics, fields, identity, catalog))
        cols = sorted(_needed_columns(metrics, fields, identity, bool(catalog)))
        rows = _read_rows_line_by_line(task.path, cols)
        file_project = task.project

        if identity.resolves_project_per_row():
            for row in rows:
                project_col = identity.project_field(fields)
                if project_col and project_col in row:
                    proj = resolve_project(row.get(project_col), identity, file_path=task.path)
                else:
                    proj = file_project
                ps = tier_project_stats[tier].setdefault(
                    proj, ReferenceStats(metrics, fields, identity, catalog)
                )
                ps.ingest_row(row, file_path=task.path, file_project=proj)
                tier_stats[tier].ingest_row(row, file_path=task.path, file_project=proj)
        else:
            for row in rows:
                tier_stats[tier].ingest_row(row, file_path=task.path, file_project=file_project)
                ps = tier_project_stats[tier].setdefault(
                    file_project, ReferenceStats(metrics, fields, identity, catalog)
                )
                ps.ingest_row(row, file_path=task.path, file_project=file_project)

    rendered: dict[str, Any] = {}
    for tier, ref in tier_stats.items():
        view = ref.render()
        view["projects_detail"] = {
            proj: ps.render() for proj, ps in sorted(tier_project_stats[tier].items())
        }
        rendered[tier] = view
    return rendered


def _close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def _compare_numeric(ref: dict, got: dict, path: str, errors: list[str]) -> None:
    for key in ("count", "mean", "std", "min", "max"):
        if key not in ref and key not in got:
            continue
        rv, gv = ref.get(key), got.get(key)
        if key in ("mean", "std"):
            if not _close(rv, gv, tol=1e-6):
                errors.append(f"{path}.{key}: ref={rv} got={gv}")
        elif rv != gv:
            errors.append(f"{path}.{key}: ref={rv} got={gv}")


def _compare_categorical(ref: dict, got: dict, path: str, errors: list[str]) -> None:
    if ref.get("unique") != got.get("unique"):
        errors.append(f"{path}.unique: ref={ref.get('unique')} got={got.get('unique')}")
    rc, gc = ref.get("counts", {}), got.get("counts", {})
    if rc != gc:
        missing = set(rc) - set(gc)
        extra = set(gc) - set(rc)
        diff = {k: (rc[k], gc[k]) for k in set(rc) & set(gc) if rc[k] != gc[k]}
        errors.append(f"{path}.counts mismatch: missing={len(missing)} extra={len(extra)} diffs={len(diff)}")
        for k in sorted(diff)[:5]:
            errors.append(f"  {path}.counts[{k!r}]: ref={diff[k][0]} got={diff[k][1]}")


def compare_views(ref: dict, got: dict, prefix: str = "") -> list[str]:
    errors: list[str] = []
    if ref.get("spectra") != got.get("spectra"):
        errors.append(f"{prefix}spectra: ref={ref.get('spectra')} got={got.get('spectra')}")

    for key in ("projects", "runs", "unique_sequences", "unique_unmodified"):
        if key in ref or key in got:
            if ref.get(key) != got.get(key):
                errors.append(f"{prefix}{key}: ref={ref.get(key)} got={got.get(key)}")

    if "peptide_length" in ref or "peptide_length" in got:
        rl, gl = ref.get("peptide_length", {}), got.get("peptide_length", {})
        _compare_numeric(rl, gl, f"{prefix}peptide_length", errors)
        if rl.get("histogram") != gl.get("histogram"):
            errors.append(f"{prefix}peptide_length.histogram mismatch")

    for key in (
        "modifications",
        "fragmentation",
        "acquisition",
        "charge",
        "analyzer_class",
        "organism",
        "instrument",
    ):
        if key in ref or key in got:
            _compare_categorical(ref.get(key, {}), got.get(key, {}), f"{prefix}{key}", errors)

    for key in ("collision_energy", "precursor_mz"):
        if key in ref or key in got:
            _compare_numeric(ref.get(key, {}), got.get(key, {}), f"{prefix}{key}", errors)

    rs, gs = ref.get("confidence_scores", {}), got.get("confidence_scores", {})
    for col in set(rs) | set(gs):
        _compare_numeric(rs.get(col, {}), gs.get(col, {}), f"{prefix}confidence_scores.{col}", errors)

    rpd, gpd = ref.get("projects_detail", {}), got.get("projects_detail", {})
    if set(rpd) != set(gpd):
        errors.append(f"{prefix}projects_detail keys: ref={sorted(rpd)} got={sorted(gpd)}")
    for proj in set(rpd) & set(gpd):
        errors.extend(compare_views(rpd[proj], gpd[proj], prefix=f"{prefix}projects_detail[{proj}]."))

    return errors


def build_mini_mount(root: Path) -> None:
    acfm = root / "acfm_splits"
    acfm.mkdir(parents=True)
    for name in ("test_0.parquet", "train_0.parquet", "val_0.parquet"):
        (acfm / name).symlink_to(FM / "acfm_splits" / name)

    lcfm = root / "lcfm" / "PXD000561"
    lcfm.mkdir(parents=True)
    for src in sorted((FM / "lcfm" / "PXD000561").glob("*.parquet"))[:8]:
        (lcfm / src.name).symlink_to(src)


def run_scenario(
    label: str,
    root: Path,
    tiers: tuple[str, ...],
    identity: IdentityConfig,
    metrics: MetricToggles,
    errors: list[str],
) -> None:
    print(f"\n=== Scenario: {label} ===")
    config = StatsConfig(
        root=root,
        tiers=tiers,
        catalog=CATALOG,
        workers=1,
        per_project=True,
        progress=False,
        identity=identity,
        metrics=metrics,
        batch_size=1,  # force row-by-row in module too
    )
    tasks = discover_tasks(config)
    print(f"  files: {len(tasks)}")

    print("  computing independent row-by-row reference ...")
    ref_tiers = compute_reference(tasks, metrics, config.fields, identity, CATALOG)

    worker_results: dict[int, dict] = {}
    for workers in (1, 2, 4, 8):
        cfg = replace(config, workers=workers)
        result = run_dataset_statistics(cfg)
        if result.files_failed:
            errors.append(f"{label}: workers={workers} had {result.files_failed} failures")
        worker_results[workers] = result.tiers
        print(f"  workers={workers}: spectra={sum(v['spectra'] for v in result.tiers.values()):,}")

    for tier in config.tiers:
        if tier not in ref_tiers:
            errors.append(f"{label}: tier {tier} missing from reference")
            continue
        ref = ref_tiers[tier]
        for workers, tiers_data in worker_results.items():
            if tier not in tiers_data:
                errors.append(f"{label}: tier {tier} missing from workers={workers}")
                continue
            got = tiers_data[tier]
            mismatches = compare_views(ref, got, prefix=f"{label}/{tier}/w{workers}/")
            if mismatches:
                errors.append(f"{label}: REF vs workers={workers} on tier {tier}: {len(mismatches)} mismatches")
                errors.extend(mismatches[:20])

    # cross-worker consistency (1 vs 2,4,8) — means may differ at float ULP from merge order
    base = worker_results[1]
    for workers in (2, 4, 8):
        for tier, view in base.items():
            other = worker_results[workers][tier]
            mismatches = compare_views(view, other, prefix=f"{label}/{tier}/w1-vs-w{workers}/")
            if mismatches:
                errors.append(f"{label}: workers=1 vs workers={workers} differ on tier {tier}")
                errors.extend(mismatches[:10])


def main() -> int:
    if not FM.is_dir() or not CATALOG.is_file():
        print("FAIL: dataset or catalog not available")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="verify_dataset_stats_"))
    errors: list[str] = []
    try:
        build_mini_mount(tmp)

        metrics_lcfm = replace(
            MetricToggles.all_disabled(),
            spectra=True,
            projects=True,
            runs=True,
            unique_sequences=True,
            unique_unmodified=True,
            peptide_length=True,
            modifications=True,
            fragmentation=True,
            acquisition=True,
            collision_energy=True,
            charge=True,
            precursor_mz=True,
            analyzer_class=True,
            organism=True,
            instrument=True,
            confidence_scores=True,
        )
        metrics_acfm = replace(
            metrics_lcfm,
            unique_sequences=False,
            unique_unmodified=False,
            peptide_length=False,
            modifications=False,
            acquisition=False,
            confidence_scores=False,
        )

        run_scenario(
            "lcfm",
            tmp,
            ("lcfm",),
            IdentityConfig(project_source="auto"),
            metrics_lcfm,
            errors,
        )
        run_scenario(
            "acfm_splits",
            tmp,
            ("acfm_splits",),
            IdentityConfig(project_source="path"),
            metrics_acfm,
            errors,
        )

        if errors:
            print("\n" + "=" * 60)
            print(f"FAIL: {len(errors)} verification error(s)")
            for err in errors[:50]:
                print(f"  - {err}")
            if len(errors) > 50:
                print(f"  ... and {len(errors) - 50} more")
            return 1

        print("\n" + "=" * 60)
        print("PASS: row-by-line reference matches module for workers 1,2,4,8")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
