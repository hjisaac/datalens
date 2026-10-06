from __future__ import annotations

from pathlib import Path

import pytest

from datalens.core.config import build_config


def test_config_rejects_empty_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one column"):
        build_config({"root": tmp_path, "columns": {}})


def test_config_rejects_unknown_accumulator(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown accumulator"):
        build_config({"root": tmp_path, "columns": {"x": "histogram"}})


def test_resolve_column_sla() -> None:
    from datalens import resolve_column_sla

    # 1. Scalar SLA
    assert resolve_column_sla(50.0, "latency") == 50.0
    assert resolve_column_sla(100, "tokens") == 100.0

    # 2. Dictionary SLA
    sla_map = {"latency": 50.0, "tokens": 512.0}
    assert resolve_column_sla(sla_map, "latency") == 50.0
    assert resolve_column_sla(sla_map, "tokens") == 512.0
    assert resolve_column_sla(sla_map, "unknown") is None

    # 3. Fallback to stats dict
    stats = {"sla": 75.0, "p50": 40.0}
    assert resolve_column_sla(None, "latency", stats=stats) == 75.0
    assert resolve_column_sla(None, "latency", stats={}) is None
    assert resolve_column_sla(None, "latency", stats=None) is None

