from __future__ import annotations

import math
import numpy as np
import pytest

from datalens import AnalysisConfig, QuantileAccumulator, run_analysis


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


def test_quantile_pipeline_execution() -> None:
    config = AnalysisConfig(columns={"score": "quantile"}, quantiles=[0.0, 0.5, 1.0])
    records = [{"score": i} for i in range(1, 101)]
    result = run_analysis(config, source=records)
    score_res = result.groups[()]["score"]
    assert score_res["count"] == 100
    assert score_res["p00"] == 1.0
    assert score_res["p50"] == pytest.approx(50.5, abs=1.0)
    assert score_res["p100"] == 100.0


def test_numeric_accumulator_comprehensive() -> None:
    from datalens import NumericAccumulator

    acc = NumericAccumulator()
    # Empty state
    empty_res = acc.result()
    assert empty_res["count"] == 0
    assert empty_res["mean"] is None
    assert empty_res["std"] is None
    assert empty_res["min"] is None
    assert empty_res["max"] is None

    # Empty array update
    acc.update(np.array([]))
    assert acc.result()["count"] == 0

    # All-NaN update
    acc.update(np.array([np.nan, np.nan]))
    assert acc.result()["count"] == 0

    # Normal update
    acc.update(np.array([10.0, 20.0, 30.0]))
    assert acc.result()["count"] == 3
    assert acc.result()["mean"] == 20.0
    assert acc.result()["min"] == 10.0
    assert acc.result()["max"] == 30.0
    assert acc.result()["std"] == pytest.approx(float(np.std([10.0, 20.0, 30.0])), rel=1e-5)

    # Merge with empty
    empty_acc = NumericAccumulator()
    acc.merge(empty_acc)
    assert acc.result()["count"] == 3

    # Merge into empty
    new_empty = NumericAccumulator()
    new_empty.merge(acc)
    assert new_empty.result()["count"] == 3
    assert new_empty.result()["mean"] == 20.0

    # Merge two non-empty
    other = NumericAccumulator()
    other.update(np.array([40.0, 50.0]))
    acc.merge(other)
    assert acc.result()["count"] == 5
    assert acc.result()["mean"] == 30.0
    assert acc.result()["min"] == 10.0
    assert acc.result()["max"] == 50.0
    assert acc.result()["std"] == pytest.approx(float(np.std([10.0, 20.0, 30.0, 40.0, 50.0])), rel=1e-5)


def test_categorical_accumulator_comprehensive() -> None:
    from datalens import CategoricalAccumulator

    acc = CategoricalAccumulator()
    assert acc.result()["unique"] == 0
    assert acc.result()["counts"] == {}

    acc.update(["apple", "banana", "apple", "cherry", "apple", "banana"])
    res = acc.result()
    assert res["unique"] == 3
    assert res["counts"]["apple"] == 3
    assert res["counts"]["banana"] == 2
    assert res["counts"]["cherry"] == 1

    # Test top truncation
    res_top = acc.result(top=2)
    assert res_top["unique"] == 3
    assert len(res_top["counts"]) == 2
    assert "cherry" not in res_top["counts"]

    # Merge
    other = CategoricalAccumulator()
    other.update(["date", "banana"])
    acc.merge(other)
    assert acc.result()["unique"] == 4
    assert acc.result()["counts"]["banana"] == 3
    assert acc.result()["counts"]["date"] == 1


def test_cardinality_accumulator_comprehensive() -> None:
    from datalens import CardinalityAccumulator

    acc = CardinalityAccumulator()
    assert acc.result()["unique"] == 0

    acc.update(["u1", "u2", "u1", None, "u3"])
    assert acc.result()["unique"] == 3

    other = CardinalityAccumulator()
    other.update(["u3", "u4", "u5"])
    acc.merge(other)
    assert acc.result()["unique"] == 5


def test_count_accumulator_comprehensive() -> None:
    from datalens import CountAccumulator

    acc = CountAccumulator()
    assert acc.result()["count"] == 0

    acc.update(15)
    assert acc.result()["count"] == 15

    other = CountAccumulator(count=25)
    acc.merge(other)
    assert acc.result()["count"] == 40


def test_quantile_accumulator_cdf() -> None:
    acc = QuantileAccumulator()
    assert acc.cdf(50.0) is None

    # Test uniform distribution [0, 1000]
    np.random.seed(42)
    data = np.random.uniform(0.0, 1000.0, size=10_000)
    acc.update(data)

    assert acc.cdf(-10.0) == 0.0
    assert acc.cdf(1100.0) == 1.0

    # CDF at median ~ 0.50
    p50 = acc.quantile(0.5)
    assert p50 is not None
    assert acc.cdf(p50) == pytest.approx(0.50, abs=0.02)

    # CDF at 250 ~ 0.25, at 750 ~ 0.75
    assert acc.cdf(250.0) == pytest.approx(0.25, abs=0.03)
    assert acc.cdf(750.0) == pytest.approx(0.75, abs=0.03)


def test_quantile_accumulator_reservoir_sampling() -> None:
    acc = QuantileAccumulator(sample_capacity=100)
    data = np.arange(1000, dtype=np.float64)
    acc.update(data)

    assert len(acc.sample) == 100
    assert all(0.0 <= x < 1000.0 for x in acc.sample)

    # Merge reservoir samples
    other = QuantileAccumulator(sample_capacity=100)
    other.update(np.arange(1000, 2000, dtype=np.float64))
    acc.merge(other)

    assert len(acc.sample) <= 100
    res = acc.result()
    assert "sample" in res
    assert len(res["sample"]) == len(acc.sample)


def test_quantile_accumulator_sla_metrics() -> None:
    acc = QuantileAccumulator()
    # 800 items <= 100, 200 items > 100
    part1 = np.random.uniform(0.0, 100.0, size=800)
    part2 = np.random.uniform(101.0, 200.0, size=200)
    data = np.concatenate([part1, part2])
    acc.update(data)

    res = acc.result(sla=100.0)
    assert "sla" in res
    assert res["sla"] == 100.0
    assert "sla_exceeded_pct" in res
    assert "sla_exceeded_count" in res
    assert "sla_compliant_pct" in res

    # Exceeded should be ~20%
    assert res["sla_exceeded_pct"] == pytest.approx(20.0, abs=2.5)
    assert res["sla_compliant_pct"] == pytest.approx(80.0, abs=2.5)
    assert res["sla_exceeded_count"] == pytest.approx(200, abs=25)

