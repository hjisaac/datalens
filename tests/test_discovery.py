from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from datalens.core.config import build_config
from datalens.core.discovery import discover_tasks
from datalens.core.types import FileTask


def _write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_discover_tasks_uses_folder_segments_as_partition(tmp_path: Path) -> None:
    table = pa.table({"charge": [2, 3], "mz": [400.0, 500.0]})
    _write_parquet(tmp_path / "lcfm" / "PXD000561" / "run1.parquet", table)
    _write_parquet(tmp_path / "lcfm" / "PXD000865" / "run2.parquet", table)
    _write_parquet(tmp_path / "flat.parquet", table)

    config = build_config(
        {
            "root": tmp_path,
            "columns": {"charge": "categorical"},
            "file_pattern": "*.parquet",
        }
    )
    tasks = discover_tasks(config)

    assert len(tasks) == 3
    by_path = {task.path: task.partition for task in tasks}
    assert by_path[str(tmp_path / "flat.parquet")] == ()
    assert by_path[str(tmp_path / "lcfm" / "PXD000561" / "run1.parquet")] == ("lcfm", "PXD000561")
    assert by_path[str(tmp_path / "lcfm" / "PXD000865" / "run2.parquet")] == ("lcfm", "PXD000865")


def test_discover_tasks_respects_partition_depth(tmp_path: Path) -> None:
    table = pa.table({"charge": [2]})
    _write_parquet(tmp_path / "lcfm" / "PXD000561" / "run1.parquet", table)

    config = build_config(
        {
            "root": tmp_path,
            "columns": {"charge": "categorical"},
            "partition_depth": 1,
        }
    )
    tasks = discover_tasks(config)
    assert tasks[0].partition == ("lcfm",)
