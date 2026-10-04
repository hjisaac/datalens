#!/usr/bin/env python
"""End-to-end smoke test for instanovo.dataset_stats on a small real-data slice."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from instanovo.dataset_stats.config import MetricToggles, StatsConfig
from instanovo.dataset_stats.identity import IdentityConfig
from instanovo.dataset_stats.runner import run_dataset_statistics

FM = Path("/app/fm_mount")
CATALOG = Path("/app/search_data.csv")
ACFM_FILES = ["test_0.parquet", "train_0.parquet", "val_0.parquet"]
LCFM_PROJECT = "PXD000561"
LCFM_FILE_LIMIT = 10


def _build_mini_mount(root: Path) -> dict[str, int]:
    """Symlink a small but representative slice of the real corpus."""
    acfm_dir = root / "acfm_splits"
    acfm_dir.mkdir(parents=True)
    for name in ACFM_FILES:
        src = FM / "acfm_splits" / name
        if not src.is_file():
            raise FileNotFoundError(src)
        (acfm_dir / name).symlink_to(src)

    lcfm_proj = root / "lcfm" / LCFM_PROJECT
    lcfm_proj.mkdir(parents=True)
    src_project = FM / "lcfm" / LCFM_PROJECT
    if not src_project.is_dir():
        raise FileNotFoundError(src_project)
    lcfm_files = sorted(src_project.glob("*.parquet"))[:LCFM_FILE_LIMIT]
    if not lcfm_files:
        raise FileNotFoundError(f"No parquet in {src_project}")
    for src in lcfm_files:
        (lcfm_proj / src.name).symlink_to(src)

    return {"acfm_splits": len(ACFM_FILES), "lcfm": len(lcfm_files)}


def _assert_tier(
    view: dict,
    *,
    expect_organism: bool,
    expect_sequences: bool,
    expect_acquisition: bool = True,
) -> list[str]:
    errors: list[str] = []
    if view.get("spectra", 0) <= 0:
        errors.append("spectra is zero")
    if view.get("projects", 0) <= 0:
        errors.append("projects is zero")
    if view.get("runs", 0) <= 0:
        errors.append("runs is zero")
    if expect_sequences:
        if not view.get("unique_sequences"):
            errors.append("unique_sequences missing")
        if not view.get("unique_unmodified"):
            errors.append("unique_unmodified missing")
    if expect_organism:
        org = view.get("organism", {})
        inst = view.get("instrument", {})
        if not org.get("counts"):
            errors.append("organism counts empty")
        if not inst.get("counts"):
            errors.append("instrument counts empty")
        if sum(org.get("counts", {}).values()) != view["spectra"]:
            errors.append(
                f"organism total {sum(org['counts'].values())} != spectra {view['spectra']}"
            )
        if sum(inst.get("counts", {}).values()) != view["spectra"]:
            errors.append(
                f"instrument total {sum(inst['counts'].values())} != spectra {view['spectra']}"
            )
    for key in ("fragmentation", "charge"):
        if not view.get(key, {}).get("counts"):
            errors.append(f"{key} counts empty")
    if expect_acquisition and not view.get("acquisition", {}).get("counts"):
        errors.append("acquisition counts empty")
    if not view.get("projects_detail"):
        errors.append("projects_detail missing")
    return errors


def main() -> int:
    if not FM.is_dir():
        print(f"FAIL: fm_mount not found at {FM}")
        return 1
    if not CATALOG.is_file():
        print(f"FAIL: catalog not found at {CATALOG}")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="dataset_stats_e2e_"))
    try:
        expected_files = _build_mini_mount(tmp)
        print("Mini mount:", expected_files)

        metrics_full = replace(
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
            confidence_scores=False,
        )
        metrics_acfm = replace(
            metrics_full,
            unique_sequences=False,
            unique_unmodified=False,
            peptide_length=False,
            modifications=False,
            confidence_scores=False,
        )

        errors: list[str] = []
        all_results: list[tuple[str, object]] = []

        runs = [
            ("acfm_splits", ("acfm_splits",), IdentityConfig(project_source="path"), metrics_acfm),
            ("lcfm", ("lcfm",), IdentityConfig(project_source="auto"), metrics_full),
        ]
        for label, tiers, identity, metrics in runs:
            config = StatsConfig(
                root=tmp,
                tiers=tiers,
                catalog=CATALOG,
                workers=4,
                per_project=True,
                progress=False,
                identity=identity,
                metrics=metrics,
            )
            result = run_dataset_statistics(config)
            out_path = tmp / f"e2e_{label}.json"
            result.to_json(out_path)
            all_results.append((label, result))

            expected = expected_files[label]
            if result.files_processed != expected:
                errors.append(f"{label}: files_processed {result.files_processed} != {expected}")
            if result.files_failed:
                errors.append(f"{label}: files_failed={result.files_failed}")
            if not result.notes:
                errors.append(f"{label}: notes missing")
            else:
                for key in ("runs", "catalog_match", "organism_counts"):
                    if key not in result.notes:
                        errors.append(f"{label}: notes.{key} missing")

            tier = tiers[0]
            if tier not in result.tiers:
                errors.append(f"{label}: tier {tier} missing")
            else:
                errors.extend(
                    _assert_tier(
                        result.tiers[tier],
                        expect_organism=True,
                        expect_sequences=label == "lcfm",
                        expect_acquisition=label == "lcfm",
                    )
                )

            payload = result.to_dict()
            required_top = {"config", "files_processed", "files_failed", "tiers", "notes"}
            if not required_top.issubset(payload):
                errors.append(f"{label}: missing top-level keys: {required_top - set(payload)}")

        print("\n=== Summary ===")
        for label, result in all_results:
            tier = label if label != "acfm_splits" else "acfm_splits"
            view = result.tiers[tier]
            print(
                f"[{label}] spectra={view['spectra']:,} projects={view.get('projects')} "
                f"runs={view.get('runs')} catalog={result.notes.get('catalog_match') if result.notes else 'n/a'} "
                f"failed={result.files_failed}"
            )
            if result.notes:
                print(f"  notes: {json.dumps(result.notes)}")

        if errors:
            print("\nFAIL:")
            for err in errors:
                print(f"  - {err}")
            return 1

        print("\nE2E PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
