from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from datalens.core.config import boot_config, build_config
from datalens.core.mapper import map_task
from datalens.core.reducer import reduce_group
from datalens.core.shuffler import shuffle
from datalens.core.types import FileTask


def _write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_map_shuffle_reduce_parquet(tmp_path: Path) -> None:
    table_a = pa.table({"charge": [2, 2, 3], "mz": [400.0, 410.0, 500.0]})
    table_b = pa.table({"charge": [2, 4], "mz": [420.0, 600.0]})
    path_a = tmp_path / "tier_a" / "proj1" / "run1.parquet"
    path_b = tmp_path / "tier_a" / "proj1" / "run2.parquet"
    _write_parquet(path_a, table_a)
    _write_parquet(path_b, table_b)

    config = build_config(
        {
            "root": tmp_path,
            "columns": {"charge": "categorical", "mz": "numeric"},
            "workers": 1,
        }
    )
    boot_config(config)

    mapped = [
        map_task(FileTask(path=str(path_a), partition=("tier_a", "proj1"))),
        map_task(FileTask(path=str(path_b), partition=("tier_a", "proj1"))),
    ]
    groups = shuffle(mapped)
    result = reduce_group(("tier_a", "proj1"), groups[("tier_a", "proj1")])

    assert result["rows"] == 5
    assert result["charge"]["counts"]["2"] == 3
    assert result["charge"]["counts"]["3"] == 1
    assert result["charge"]["counts"]["4"] == 1
    assert result["mz"]["count"] == 5
    assert result["mz"]["min"] == 400.0
    assert result["mz"]["max"] == 600.0


def test_run_analysis_multiple_partitions(tmp_path: Path) -> None:
    _write_parquet(
        tmp_path / "east" / "file.parquet",
        pa.table({"region": ["east", "east"], "value": [1.0, 2.0]}),
    )
    _write_parquet(
        tmp_path / "west" / "file.parquet",
        pa.table({"region": ["west"], "value": [3.0]}),
    )

    config = build_config(
        {
            "root": tmp_path,
            "columns": {"region": "categorical", "value": "numeric"},
            "workers": 1,
        }
    )

    from datalens.core.orchestrator import run_analysis

    result = run_analysis(config)
    assert result.files_processed == 2
    assert result.files_failed == 0
    assert result.groups[("east",)]["rows"] == 2
    assert result.groups[("west",)]["rows"] == 1
    assert result.groups[("east",)]["region"]["counts"]["east"] == 2
    assert result.groups[("west",)]["value"]["mean"] == 3.0

    # Test to_dict()
    as_dict = result.to_dict()
    assert as_dict["files_processed"] == 2
    assert as_dict["files_failed"] == 0
    assert "east" in as_dict["groups"]
    assert "west" in as_dict["groups"]

    # Test to_json()
    out_json = tmp_path / "summary.json"
    result.to_json(out_json)
    import json
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded == as_dict


def test_public_package_api_import(tmp_path: Path) -> None:
    from datalens import (
        AnalysisConfig,
        AnalysisResult,
        CardinalityAccumulator,
        CategoricalAccumulator,
        CountAccumulator,
        DataTask,
        FileTask,
        NumericAccumulator,
        PartialStats,
        QuantileAccumulator,
        StatsConfig,
        TDigestAccumulator,
        Task,
        build_config,
        discover_tasks,
        load_config,
        resolve_tasks,
        run_analysis,
    )

    assert callable(run_analysis)
    assert callable(load_config)
    assert callable(build_config)
    assert callable(discover_tasks)
    assert callable(resolve_tasks)
    assert AnalysisConfig is not None
    assert StatsConfig is AnalysisConfig
    assert AnalysisResult is not None
    assert DataTask is not None
    assert FileTask is not None
    assert NumericAccumulator is not None
    assert QuantileAccumulator is not None
    assert TDigestAccumulator is QuantileAccumulator


def test_run_analysis_in_memory_records() -> None:
    from datalens import AnalysisConfig, run_analysis

    config = AnalysisConfig(columns={"score": "numeric", "status": "categorical"})
    records = [
        {"score": 10.0, "status": "ok"},
        {"score": 20.0, "status": "fail"},
        {"score": 30.0, "status": "ok"},
    ]
    result = run_analysis(config, source=records)
    assert result.groups[()]["rows"] == 3
    assert result.groups[()]["score"]["mean"] == 20.0
    assert result.groups[()]["status"]["counts"]["ok"] == 2


def test_run_analysis_with_data_tasks() -> None:
    from datalens import AnalysisConfig, DataTask, run_analysis

    config = AnalysisConfig(columns={"value": "numeric"})
    tasks = [
        DataTask(data=[{"value": 1.0}, {"value": 2.0}], partition=("group_1",)),
        DataTask(data=[{"value": 3.0}], partition=("group_2",)),
    ]
    result = run_analysis(config, source=tasks)
    assert result.groups[("group_1",)]["rows"] == 2
    assert result.groups[("group_1",)]["value"]["mean"] == 1.5
    assert result.groups[("group_2",)]["rows"] == 1
    assert result.groups[("group_2",)]["value"]["mean"] == 3.0


def test_direct_accumulator_classes() -> None:
    import numpy as np
    from datalens import (
        CardinalityAccumulator,
        CategoricalAccumulator,
        CountAccumulator,
        NumericAccumulator,
    )

    num = NumericAccumulator()
    num.update(np.array([10.0, 30.0]))
    assert num.result()["mean"] == 20.0

    cat = CategoricalAccumulator()
    cat.update(["a", "b", "a"])
    assert cat.result()["counts"]["a"] == 2

    card = CardinalityAccumulator()
    card.update(["x", "y", "x"])
    assert card.result()["unique"] == 2

    cnt = CountAccumulator()
    cnt.update(10)
    assert cnt.result()["count"] == 10


