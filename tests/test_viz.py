from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from datalens import (
    AnalysisConfig,
    AnalysisResult,
    DataTask,
    apply_style,
    generate_plots,
    get_palette,
    get_style_rc,
    plot_beeswarm_box,
    plot_categorical_frequency,
    plot_numeric_summary,
    plot_partition_comparison,
    plot_quantile_distribution,
    run_analysis,
)
from datalens.__main__ import app

runner = CliRunner()


@pytest.fixture
def sample_analysis_result() -> AnalysisResult:
    config = AnalysisConfig(
        columns={"latency": "quantile", "status": "categorical", "score": "numeric"},
        quantiles=[0.01, 0.25, 0.5, 0.75, 0.99],
    )
    records = [
        {"latency": 10.0, "status": "ok", "score": 100.0},
        {"latency": 25.0, "status": "ok", "score": 80.0},
        {"latency": 45.0, "status": "warn", "score": 60.0},
        {"latency": 99.0, "status": "error", "score": 40.0},
    ]
    return run_analysis(config, source=records)


def test_styles_presets() -> None:
    for name in ["datalens", "paper", "dark"]:
        rc = get_style_rc(name)
        assert "figure.facecolor" in rc
        palette = get_palette(name)
        assert len(palette) >= 5

    # Context manager execution
    with apply_style("paper"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.close(fig)


def test_plot_quantile_distribution(tmp_path: Path) -> None:
    stats = {
        "count": 1000,
        "min": 1.0,
        "max": 100.0,
        "p01": 2.0,
        "p05": 5.0,
        "p25": 25.0,
        "p50": 50.0,
        "p75": 75.0,
        "p90": 90.0,
        "p95": 95.0,
        "p99": 99.0,
        "iqr": 50.0,
    }
    out_png = tmp_path / "quantile.png"
    result_path = plot_quantile_distribution("response_time", stats, out_png, captions=True, style="datalens")
    assert result_path.is_file()
    assert result_path.stat().st_size > 0

    # Test paper mode without captions
    out_svg = tmp_path / "quantile_paper.svg"
    paper_path = plot_quantile_distribution("response_time", stats, out_svg, captions=False, style="paper")
    assert paper_path.is_file()
    assert paper_path.stat().st_size > 0


def test_plot_categorical_frequency(tmp_path: Path) -> None:
    stats = {
        "unique": 3,
        "counts": {"alpha": 50, "beta": 30, "gamma": 10},
    }
    out_png = tmp_path / "categories.png"
    result_path = plot_categorical_frequency("device_type", stats, out_png, captions=True, top=5)
    assert result_path.is_file()
    assert result_path.stat().st_size > 0


def test_plot_numeric_summary(tmp_path: Path) -> None:
    stats = {
        "count": 500,
        "mean": 42.5,
        "std": 8.2,
        "min": 10.0,
        "max": 75.0,
    }
    out_png = tmp_path / "numeric.png"
    result_path = plot_numeric_summary("temperature", stats, out_png, captions=True)
    assert result_path.is_file()
    assert result_path.stat().st_size > 0


def test_plot_partition_comparison(tmp_path: Path) -> None:
    groups = {
        "us-east": {"latency": {"p25": 10.0, "p50": 20.0, "p75": 30.0, "p01": 5.0, "p99": 45.0, "min": 2.0, "max": 50.0}},
        "eu-west": {"latency": {"p25": 15.0, "p50": 30.0, "p75": 45.0, "p01": 8.0, "p99": 60.0, "min": 5.0, "max": 70.0}},
    }
    out_png = tmp_path / "partition_comp.png"
    result_path = plot_partition_comparison("latency", groups, out_png, kind="quantile")
    assert result_path.is_file()
    assert result_path.stat().st_size > 0


def test_generate_plots_and_result_method(sample_analysis_result: AnalysisResult, tmp_path: Path) -> None:
    plot_dir = tmp_path / "my_plots"
    plots = sample_analysis_result.plot(out_dir=plot_dir, format="png", captions=True)
    assert "latency_quantile" in plots
    assert "status_categorical" in plots
    assert "score_numeric" in plots
    for path in plots.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_orchestrator_auto_plots(tmp_path: Path) -> None:
    plot_dir = tmp_path / "auto_plots"
    config = AnalysisConfig(
        columns={"latency": "quantile"},
        plots=True,
        plot_dir=plot_dir,
    )
    records = [{"latency": 10.0}, {"latency": 20.0}]
    run_analysis(config, source=records)
    assert (plot_dir / "latency_quantile.png").is_file()


def test_cli_plots_flag_and_deactivate(tmp_path: Path) -> None:
    import yaml

    cfg_file = tmp_path / "job.yaml"
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("x,y\n1.0,a\n2.0,b\n3.0,a\n", encoding="utf-8")

    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.dump(
            {
                "root": str(tmp_path),
                "columns": {"x": "quantile", "y": "categorical"},
                "file_pattern": "*.csv",
                "workers": 1,
            },
            f,
        )

    # Run with default plots enabled
    out_dir = tmp_path / "cli_plots"
    res = runner.invoke(app, ["run", str(cfg_file), "--plot-dir", str(out_dir)])
    assert res.exit_code == 0
    assert (out_dir / "x_quantile.png").is_file()
    assert (out_dir / "y_categorical.png").is_file()

    # Run with --no-plots to deactivate
    no_plot_dir = tmp_path / "no_plots"
    res_no = runner.invoke(app, ["run", str(cfg_file), "--no-plots", "--plot-dir", str(no_plot_dir)])
    assert res_no.exit_code == 0
    assert not no_plot_dir.exists()


def test_mcp_generate_dataset_plots_tool(tmp_path: Path) -> None:
    from datalens.mcp import generate_dataset_plots

    data_file = tmp_path / "metrics.csv"
    data_file.write_text("response_ms,code\n12.5,200\n45.0,500\n8.2,200\n", encoding="utf-8")

    out_dir = tmp_path / "mcp_plots"
    result = generate_dataset_plots(str(data_file), out_dir=str(out_dir), format="svg", captions=False, style="paper")
    assert result["status"] == "success"
    assert result["style"] == "paper"
    assert not result["captions"]
    assert len(result["plots"]) > 0
    for file_path_str in result["plots"].values():
        p = Path(file_path_str)
        assert p.is_file()
        assert p.suffix == ".svg"


def test_plot_quantile_distribution_with_sla(tmp_path: Path) -> None:
    stats = {
        "count": 1000,
        "min": 1.0,
        "max": 100.0,
        "p01": 2.0,
        "p05": 5.0,
        "p25": 25.0,
        "p50": 50.0,
        "p75": 75.0,
        "p90": 90.0,
        "p95": 95.0,
        "p99": 99.0,
        "iqr": 50.0,
        "sla": 80.0,
        "sla_exceeded_count": 200,
        "sla_exceeded_pct": 20.0,
    }
    out_png = tmp_path / "quantile_sla.png"
    result_path = plot_quantile_distribution("latency", stats, out_png, sla=80.0, captions=True)
    assert result_path.is_file()
    assert result_path.stat().st_size > 0


def test_plot_beeswarm_box(tmp_path: Path) -> None:
    # 1. From raw list
    data_list = [10.0, 15.0, 20.0, 22.0, 30.0, 45.0, 50.0, 80.0]
    out1 = tmp_path / "beeswarm_raw.png"
    p1 = plot_beeswarm_box("token_count", data_list, out1, sla=40.0)
    assert p1.is_file()
    assert p1.stat().st_size > 0

    # 2. From partition dictionary
    part_data = {
        "Old Testament": [50.0, 60.0, 120.0, 150.0, 200.0, 520.0],
        "New Testament": [40.0, 55.0, 80.0, 95.0, 110.0, 250.0],
    }
    out2 = tmp_path / "beeswarm_parts.png"
    p2 = plot_beeswarm_box("token_count", part_data, out2, sla=512.0, captions=True)
    assert p2.is_file()
    assert p2.stat().st_size > 0

    # 3. Invalid data with no samples raises ValueError
    with pytest.raises(ValueError, match="No sample data found"):
        plot_beeswarm_box("col", {}, tmp_path / "err.png")


def test_plot_beeswarm_from_analysis_result(sample_analysis_result: AnalysisResult, tmp_path: Path) -> None:
    out = tmp_path / "result_beeswarm.png"
    res_path = sample_analysis_result.plot_beeswarm("latency", out, sla=50.0)
    assert res_path.is_file()
    assert res_path.stat().st_size > 0


def test_plot_partition_comparison_with_sla(tmp_path: Path) -> None:
    groups = {
        "us-east": {"latency": {"p25": 10.0, "p50": 20.0, "p75": 30.0, "p01": 5.0, "p99": 45.0, "min": 2.0, "max": 50.0}},
        "eu-west": {"latency": {"p25": 15.0, "p50": 30.0, "p75": 45.0, "p01": 8.0, "p99": 60.0, "min": 5.0, "max": 70.0}},
    }
    out_png = tmp_path / "comp_sla.png"
    result_path = plot_partition_comparison("latency", groups, out_png, kind="quantile", sla=25.0)
    assert result_path.is_file()
    assert result_path.stat().st_size > 0


def test_generate_plots_with_sla_and_beeswarm(tmp_path: Path) -> None:
    config = AnalysisConfig(
        columns={"tokens": "quantile"},
        sla=100.0,
    )
    records = [{"tokens": float(i)} for i in range(1, 201)]
    result = run_analysis(config, source=records)

    plots = result.plot(out_dir=tmp_path / "plots_sla", sla=100.0)
    assert "tokens_quantile" in plots
    assert "tokens_beeswarm" in plots
    assert plots["tokens_beeswarm"].is_file()
    assert plots["tokens_beeswarm"].stat().st_size > 0


def test_cli_sla_flag(tmp_path: Path) -> None:
    import yaml

    cfg_file = tmp_path / "job_sla.yaml"
    csv_file = tmp_path / "data_sla.csv"
    csv_file.write_text("tokens\n50.0\n120.0\n200.0\n", encoding="utf-8")

    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.dump(
            {
                "root": str(tmp_path),
                "columns": {"tokens": "quantile"},
                "file_pattern": "data_sla.csv",
                "workers": 1,
            },
            f,
        )

    out_dir = tmp_path / "cli_plots_sla"
    res = runner.invoke(app, ["run", str(cfg_file), "--plot-dir", str(out_dir), "--sla", "100.0"])
    assert res.exit_code == 0
    assert (out_dir / "tokens_quantile.png").is_file()
    assert (out_dir / "tokens_beeswarm.png").is_file()

