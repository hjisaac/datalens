from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from instanovo.dataset_stats.catalog import load_catalog, normalize_run_name
from instanovo.dataset_stats.collector import (
    FileTask,
    build_accumulator,
    collect_file,
    init_worker_catalog,
    render,
)
from instanovo.dataset_stats.config import FieldMap, MetricToggles
from instanovo.dataset_stats.identity import IdentityConfig

CATALOG = Path("/app/search_data.csv")
LCFM_SAMPLE = Path("/app/fm_mount/lcfm/PXD000561/Fetal_Heart_bRP_Elite_19_f21.parquet")
ACFM_SAMPLE = Path("/app/fm_mount/acfm_splits/test_0.parquet")


def _organism_instrument_metrics() -> MetricToggles:
    return replace(
        MetricToggles.all_disabled(),
        spectra=True,
        organism=True,
        instrument=True,
        runs=True,
    )


@pytest.fixture(autouse=True)
def _reset_catalog() -> None:
    init_worker_catalog({})
    yield
    init_worker_catalog({})


def test_normalize_run_name_strips_mzml_suffix() -> None:
    assert normalize_run_name("Adult_Adrenalgland_Gel_Elite_49_f03.mzML") == "Adult_Adrenalgland_Gel_Elite_49_f03"


@pytest.mark.skipif(not CATALOG.is_file(), reason="search_data.csv not available")
@pytest.mark.skipif(not LCFM_SAMPLE.is_file(), reason="LCFM sample parquet not available")
def test_lcfm_catalog_join_uses_project_and_experiment_name() -> None:
    init_worker_catalog(load_catalog(CATALOG))
    metrics = _organism_instrument_metrics()
    task = FileTask(tier="lcfm", project="PXD000561", path=str(LCFM_SAMPLE))
    outcomes = collect_file(task, metrics, FieldMap(), IdentityConfig(), batch_size=4096)
    assert len(outcomes) == 1
    _, project, bundle = outcomes[0]
    assert project == "PXD000561"
    assert bundle.spectra > 0
    assert bundle.catalog_files_matched == 1
    assert bundle.catalog_files_missed == 0

    view = render(bundle, keep_top=None)
    assert sum(view["organism"]["counts"].values()) == bundle.spectra
    assert sum(view["instrument"]["counts"].values()) == bundle.spectra


@pytest.mark.skipif(not CATALOG.is_file(), reason="search_data.csv not available")
@pytest.mark.skipif(not ACFM_SAMPLE.is_file(), reason="ACFM sample parquet not available")
def test_acfm_catalog_join_uses_row_level_project_and_run() -> None:
    init_worker_catalog(load_catalog(CATALOG))
    metrics = _organism_instrument_metrics()
    identity = IdentityConfig(project_source="path")
    task = FileTask(tier="acfm_splits", project="__UNRESOLVED__", path=str(ACFM_SAMPLE))
    outcomes = collect_file(task, metrics, FieldMap(), identity, batch_size=8192)
    assert outcomes
    total_spectra = sum(bundle.spectra for _, _, bundle in outcomes)
    assert total_spectra == 199_970

    matched = sum(bundle.catalog_files_matched for _, _, bundle in outcomes)
    missed = sum(bundle.catalog_files_missed for _, _, bundle in outcomes)
    assert matched == 46
    assert missed == 0

    organism_total = 0
    instrument_total = 0
    for _, _, bundle in outcomes:
        view = render(bundle, keep_top=None)
        organism_total += sum(view["organism"]["counts"].values())
        instrument_total += sum(view["instrument"]["counts"].values())
    assert organism_total == total_spectra
    assert instrument_total == total_spectra


@pytest.mark.skipif(not CATALOG.is_file(), reason="search_data.csv not available")
@pytest.mark.skipif(not LCFM_SAMPLE.is_file(), reason="LCFM sample parquet not available")
def test_lcfm_without_run_column_falls_back_to_filename() -> None:
    init_worker_catalog(load_catalog(CATALOG))
    metrics = _organism_instrument_metrics()
    task = FileTask(tier="lcfm", project="PXD000561", path=str(LCFM_SAMPLE))
    fields = FieldMap(run="__missing_run__")
    outcomes = collect_file(task, metrics, fields, IdentityConfig(), batch_size=4096)
    bundle = outcomes[0][2]
    assert bundle.catalog_files_matched == 1
    view = render(bundle, keep_top=None)
    assert view["organism"]["counts"]
