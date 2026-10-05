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


def test_resolve_tasks_with_directory_source(tmp_path: Path) -> None:
    from datalens.core.discovery import resolve_tasks

    table = pa.table({"x": [1, 2]})
    data_dir = tmp_path / "custom_data"
    _write_parquet(data_dir / "part1.parquet", table)
    _write_parquet(data_dir / "sub" / "part2.parquet", table)

    # Config with no root
    config = build_config({"columns": {"x": "numeric"}})
    tasks = resolve_tasks(config, source=data_dir)
    assert len(tasks) == 2
    paths = [t.path for t in tasks]
    assert str(data_dir / "part1.parquet") in paths
    assert str(data_dir / "sub" / "part2.parquet") in paths


def test_resolve_tasks_with_single_file_and_list(tmp_path: Path) -> None:
    from datalens.core.discovery import resolve_tasks

    table = pa.table({"x": [1]})
    f1 = tmp_path / "f1.parquet"
    f2 = tmp_path / "f2.parquet"
    _write_parquet(f1, table)
    _write_parquet(f2, table)

    config = build_config({"columns": {"x": "numeric"}})

    # Single file string
    single = resolve_tasks(config, source=str(f1))
    assert len(single) == 1
    assert single[0].path == str(f1)

    # List of file paths
    multiple = resolve_tasks(config, source=[f1, f2])
    assert len(multiple) == 2
    assert multiple[0].path == str(f1)
    assert multiple[1].path == str(f2)


def test_resolve_tasks_nonexistent_source(tmp_path: Path) -> None:
    import pytest
    from datalens.core.discovery import resolve_tasks

    config = build_config({"columns": {"x": "numeric"}})
    with pytest.raises(FileNotFoundError):
        resolve_tasks(config, source=tmp_path / "does_not_exist")

