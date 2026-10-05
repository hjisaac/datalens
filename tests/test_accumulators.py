from __future__ import annotations

import math
import numpy as np
import pytest

from datalens import AnalysisConfig, QuantileAccumulator, TDigestAccumulator, run_analysis


def test_quantile_accumulator_empty() -> None:
    acc = QuantileAccumulator()
    res = acc.result()
    assert res["count"] == 0
    assert res["min"] is None
    assert res["max"] is None
    assert res["p50"] is None
    assert res["iqr"] is None
    assert acc.quantile(0.5) is None


def test_quantile_accumulator_single_value() -> None:
    acc = QuantileAccumulator()
    acc.update(np.array([42.0]))
    res = acc.result()
    assert res["count"] == 1
    assert res["min"] == 42.0
    assert res["max"] == 42.0
    assert res["p50"] == 42.0
    assert res["p99"] == 42.0
    assert res["iqr"] == 0.0


def test_quantile_accumulator_distribution_accuracy() -> None:
    np.random.seed(42)
    # Generate 10,000 normal random samples
    data = np.random.normal(loc=100.0, scale=15.0, size=10_000)
    acc = QuantileAccumulator(delta=100.0)
    acc.update(data)

    res = acc.result()
    assert res["count"] == 10_000
    assert res["min"] == pytest.approx(float(data.min()), rel=1e-5)
    assert res["max"] == pytest.approx(float(data.max()), rel=1e-5)

    # Median / P50 should be very close to 100.0 (< 0.5% error)
    exact_p50 = float(np.percentile(data, 50))
    assert res["p50"] == pytest.approx(exact_p50, rel=0.01)

    # P25 and P75 and IQR
    exact_p25 = float(np.percentile(data, 25))
    exact_p75 = float(np.percentile(data, 75))
    assert res["p25"] == pytest.approx(exact_p25, rel=0.01)
    assert res["p75"] == pytest.approx(exact_p75, rel=0.01)
    assert res["iqr"] == pytest.approx(exact_p75 - exact_p25, rel=0.02)

    # Extreme tails: P01 and P99
    exact_p01 = float(np.percentile(data, 1))
    exact_p99 = float(np.percentile(data, 99))
    assert res["p01"] == pytest.approx(exact_p01, rel=0.02)
    assert res["p99"] == pytest.approx(exact_p99, rel=0.02)


def test_quantile_accumulator_custom_quantiles() -> None:
    acc = QuantileAccumulator()
    acc.update(np.arange(1, 101, dtype=np.float64))  # 1 to 100
    res = acc.result(quantiles=[0.10, 0.50, 0.90, 0.999])

    assert "p10" in res
    assert "p50" in res
    assert "p90" in res
    assert "p99.9" in res
    assert res["p50"] == pytest.approx(50.5, abs=1.0)


def test_quantile_accumulator_merge() -> None:
    np.random.seed(123)
    chunk1 = np.random.uniform(0.0, 1000.0, size=5000)
    chunk2 = np.random.uniform(0.0, 1000.0, size=5000)
    full = np.concatenate([chunk1, chunk2])

    acc1 = QuantileAccumulator()
    acc1.update(chunk1)

    acc2 = QuantileAccumulator()
    acc2.update(chunk2)

    acc1.merge(acc2)
    res = acc1.result()

    assert res["count"] == 10_000
    assert res["p50"] == pytest.approx(float(np.percentile(full, 50)), rel=0.02)
    assert res["p99"] == pytest.approx(float(np.percentile(full, 99)), rel=0.02)


def test_quantile_accumulator_nan_and_empty() -> None:
    acc = QuantileAccumulator()
    acc.update(np.array([]))
    acc.update(np.array([np.nan, np.nan]))
    assert acc.total_weight == 0
    assert acc.result()["count"] == 0

    acc.update(np.array([10.0, np.nan, 20.0]))
    assert acc.result()["count"] == 2
    assert acc.result()["p50"] == pytest.approx(15.0, abs=1.0)


def test_pipeline_with_quantile_accumulator() -> None:
    # Test through the full map-reduce pipeline with in-memory records
    config = AnalysisConfig(
        columns={"latency": "quantile", "status": "categorical"},
        quantiles=[0.5, 0.95, 0.99],
    )
    records = [
        {"latency": 10.0, "status": "ok"},
        {"latency": 20.0, "status": "ok"},
        {"latency": 30.0, "status": "warn"},
        {"latency": 100.0, "status": "err"},
    ]

    result = run_analysis(config, source=records)
    group = result.groups[()]

    assert group["rows"] == 4
    lat = group["latency"]
    assert lat["count"] == 4
    assert lat["min"] == 10.0
    assert lat["max"] == 100.0
    assert "p50" in lat
    assert "p95" in lat
    assert "p99" in lat


def test_tdigest_alias_in_pipeline() -> None:
    config = AnalysisConfig(columns={"score": "tdigest"})
    records = [{"score": i} for i in range(1, 101)]
    result = run_analysis(config, source=records)
    assert result.groups[()]["score"]["count"] == 100
    assert result.groups[()]["score"]["p50"] == pytest.approx(50.5, abs=1.0)
