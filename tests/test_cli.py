from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from datalens.__main__ import app

runner = CliRunner()


def test_cli_execution_with_output_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "p1"
    data_dir.mkdir(parents=True)
    table = pa.table({"score": [10.0, 20.0, 30.0], "label": ["a", "b", "a"]})
    pq.write_table(table, data_dir / "test.parquet")

    cfg_path = tmp_path / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            {
                "root": str(tmp_path / "data"),
                "columns": {"score": "numeric", "label": "categorical"},
                "workers": 1,
            },
            f,
        )

    out_json = tmp_path / "out.json"
    result = runner.invoke(app, [str(cfg_path), "-o", str(out_json), "-q"])
    assert result.exit_code == 0
    assert out_json.is_file()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["files_processed"] == 1
    assert payload["files_failed"] == 0
    assert payload["groups"]["p1"]["rows"] == 3
    assert payload["groups"]["p1"]["score"]["mean"] == 20.0
    assert payload["groups"]["p1"]["label"]["counts"]["a"] == 2


def test_cli_missing_config() -> None:
    result = runner.invoke(app, ["/nonexistent/path/config.yaml"])
    assert result.exit_code != 0


def test_cli_explicit_run_subcommand(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    table = pa.table({"val": [5.0, 15.0]})
    pq.write_table(table, data_dir / "test.parquet")

    cfg_path = tmp_path / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            {
                "root": str(data_dir),
                "columns": {"val": "numeric"},
                "workers": 1,
            },
            f,
        )

    result = runner.invoke(app, ["run", str(cfg_path)])
    assert result.exit_code == 0

