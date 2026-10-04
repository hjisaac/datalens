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
