from __future__ import annotations

from pathlib import Path

import pytest

from datalens.core.config import build_config
from datalens.core.orchestrator import run_analysis


def test_run_analysis_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("color,count\nred,1\nblue,2\nred,3\n", encoding="utf-8")

    config = build_config(
        {
            "root": tmp_path,
            "columns": {"color": "categorical", "count": "numeric"},
            "file_pattern": "*.csv",
            "workers": 1,
        }
    )

    result = run_analysis(config)
    group = result.groups[()]
    assert group["rows"] == 3
    assert group["color"]["counts"]["red"] == 2
    assert group["color"]["counts"]["blue"] == 1
    assert group["count"]["mean"] == 2.0


def test_run_analysis_jsonl(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "events.jsonl"
    jsonl_path.write_text(
        '{"event":"click","user":"a"}\n{"event":"view","user":"b"}\n{"event":"click","user":"a"}\n',
        encoding="utf-8",
    )

    config = build_config(
        {
            "root": tmp_path,
            "columns": {"event": "categorical", "user": "cardinality"},
            "file_pattern": "*.jsonl",
            "workers": 1,
        }
    )

    result = run_analysis(config)
    group = result.groups[()]
    assert group["rows"] == 3
    assert group["event"]["counts"]["click"] == 2
    assert group["user"]["unique"] == 2


def test_numeric_values_pure_numpy() -> None:
    import numpy as np
    from datalens.core.readers import numeric_values

    # Test numbers, None, strings, and floats
    res = numeric_values([10, None, 20.5, "30.0", "invalid"])
    assert res[0] == 10.0
    assert np.isnan(res[1])
    assert res[2] == 20.5
    assert res[3] == 30.0
    assert np.isnan(res[4])


def test_parquet_without_pyarrow_raises_importerror(tmp_path: Path, monkeypatch) -> None:
    import sys
    import pytest
    from datalens.core.readers import read_batches

    dummy = tmp_path / "test.parquet"
    dummy.write_text("dummy")

    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(ImportError, match="pyarrow is required to read Parquet files"):
        list(read_batches(dummy, columns=["a"], batch_size=10))


def test_run_analysis_tsv(tmp_path: Path) -> None:
    tsv_path = tmp_path / "data.tsv"
    tsv_path.write_text("city\tpopulation\nParis\t2.1\nLyon\t0.5\nParis\t2.1\n", encoding="utf-8")

    config = build_config(
        {
            "root": tmp_path,
            "columns": {"city": "categorical", "population": "numeric"},
            "file_pattern": "*.tsv",
            "workers": 1,
        }
    )

    result = run_analysis(config)
    group = result.groups[()]
    assert group["rows"] == 3
    assert group["city"]["counts"]["Paris"] == 2
    assert group["population"]["mean"] == pytest.approx((2.1 + 0.5 + 2.1) / 3, rel=1e-4)


def test_missing_column_handled_gracefully(tmp_path: Path) -> None:
    csv_path = tmp_path / "partial.csv"
    csv_path.write_text("existing_col\n100\n200\n", encoding="utf-8")

    config = build_config(
        {
            "root": tmp_path,
            "columns": {"existing_col": "numeric", "missing_col": "numeric"},
            "file_pattern": "*.csv",
            "workers": 1,
        }
    )

    result = run_analysis(config)
    group = result.groups[()]
    assert group["rows"] == 2
    assert group["existing_col"]["count"] == 2
    assert group["missing_col"]["count"] == 0


