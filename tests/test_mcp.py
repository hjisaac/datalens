from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from datalens import AnalysisConfig
from datalens.__main__ import app
from datalens.mcp import compute_statistics, create_mcp_server, inspect_dataset, profile_dataset

runner = CliRunner()


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "id,latency,category,is_active\n"
        "id_1,12.5,electronics,true\n"
        "id_2,45.0,books,false\n"
        "id_3,8.2,electronics,true\n"
        "id_4,19.1,clothing,true\n",
        encoding="utf-8",
    )
    return csv_file


@pytest.fixture
def sample_partitioned_dir(tmp_path: Path) -> Path:
    base = tmp_path / "dataset"
    (base / "region=us").mkdir(parents=True)
    (base / "region=eu").mkdir(parents=True)

    (base / "region=us" / "part1.csv").write_text(
        "score,label\n10.0,alpha\n20.0,beta\n",
        encoding="utf-8",
    )
    (base / "region=eu" / "part2.csv").write_text(
        "score,label\n30.0,alpha\n40.0,gamma\n",
        encoding="utf-8",
    )
    return base


def test_inspect_dataset_file(sample_csv: Path) -> None:
    result = inspect_dataset(str(sample_csv))
    assert result["file_count"] == 1
    assert result["format"] == "csv"
    assert result["sample_rows"] == 4

    cols = result["columns"]
    assert "latency" in cols
    assert cols["latency"]["type"] == "float"
    assert cols["latency"]["recommended_kind"] == "quantile"

    assert "category" in cols
    assert cols["category"]["type"] == "string"
    assert cols["category"]["recommended_kind"] == "categorical"

    assert "is_active" in cols
    assert cols["is_active"]["type"] == "boolean"
    assert cols["is_active"]["recommended_kind"] == "categorical"

    rec_cols = result["recommended_config"]["columns"]
    assert rec_cols["latency"] == "quantile"
    assert rec_cols["category"] == "categorical"


def test_inspect_dataset_directory(sample_partitioned_dir: Path) -> None:
    result = inspect_dataset(str(sample_partitioned_dir))
    assert result["file_count"] == 2
    assert len(result["partitions_detected"]) == 2
    assert "score" in result["columns"]
    assert result["columns"]["score"]["type"] == "float"


def test_inspect_dataset_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        inspect_dataset("/nonexistent/path/12345")


def test_compute_statistics_tool(sample_csv: Path) -> None:
    stats = compute_statistics(
        path=str(sample_csv),
        columns={"latency": "quantile", "category": "categorical"},
        quantiles=[0.5, 0.99],
    )
    assert "groups" in stats
    group = stats["groups"]["_all"]
    assert group["rows"] == 4
    assert group["latency"]["count"] == 4
    assert "p50" in group["latency"]
    assert group["category"]["counts"]["electronics"] == 2


def test_profile_dataset_tool(sample_csv: Path) -> None:
    profile = profile_dataset(str(sample_csv))
    assert "dataset" in profile
    assert "schema" in profile
    assert "statistics" in profile
    assert profile["dataset"]["file_count"] == 1
    assert profile["statistics"]["groups"]["_all"]["rows"] == 4


def test_mcp_server_registry() -> None:
    server = create_mcp_server()

    # Test tools registration
    tools = asyncio.run(server.list_tools())
    tool_names = {t.name for t in tools}
    assert "inspect_dataset" in tool_names
    assert "compute_statistics" in tool_names
    assert "profile_dataset" in tool_names

    # Test resources registration
    resources = asyncio.run(server.list_resources())
    resource_uris = {r.uri for r in resources}
    assert "datalens://workspace/datasets" in resource_uris

    # Test prompts registration
    prompts = asyncio.run(server.list_prompts())
    prompt_names = {p.name for p in prompts}
    assert "profile_and_analyze" in prompt_names


def test_mcp_server_call_tool(sample_csv: Path) -> None:
    server = create_mcp_server()
    res = asyncio.run(server.call_tool("inspect_dataset", {"path": str(sample_csv)}))
    assert not res.is_error
    payload = json.loads(res.content[0].text)
    assert payload["file_count"] == 1
    assert "latency" in payload["columns"]


def test_cli_mcp_help() -> None:
    res = runner.invoke(app, ["mcp", "--help"])
    assert res.exit_code == 0
    assert "Model Context Protocol" in res.output or "mcp" in res.output


def test_inspect_dataset_empty_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No supported tabular dataset files found"):
        inspect_dataset(str(empty_dir))


def test_mcp_profile_jsonl(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "stream.jsonl"
    jsonl_file.write_text(
        '{"speed": 100.5, "device": "sensor_1"}\n'
        '{"speed": 120.0, "device": "sensor_2"}\n'
        '{"speed": 98.2, "device": "sensor_1"}\n',
        encoding="utf-8",
    )
    profile = profile_dataset(str(jsonl_file))
    assert profile["dataset"]["format"] == "jsonl"
    assert profile["dataset"]["file_count"] == 1
    assert "speed" in profile["schema"]
    assert profile["schema"]["speed"]["type"] == "float"
    assert "statistics" in profile
    group = profile["statistics"]["groups"]["_all"]
    assert group["rows"] == 3
    assert group["speed"]["count"] == 3
    assert "p50" in group["speed"]
    assert group["device"]["counts"]["sensor_1"] == 2


def test_mcp_resources_and_prompts_execution(sample_csv: Path) -> None:
    server = create_mcp_server()

    # Test reading resource
    resources = asyncio.run(server.list_resources())
    uri = resources[0].uri
    res_contents = asyncio.run(server.read_resource(uri))
    assert len(res_contents) > 0

    # Test getting prompt
    prompt = asyncio.run(server.get_prompt("profile_and_analyze", {"path": str(sample_csv)}))
    assert prompt.messages
    assert "inspect_dataset" in prompt.messages[0].content.text

