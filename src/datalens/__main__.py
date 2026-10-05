"""Typer CLI entry point for datalens.

Usage:
    datalens config.yaml
    datalens run config.yaml --output stats.json
    datalens mcp
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from typer.core import TyperGroup

from .core.config import load_config
from .core.orchestrator import run_analysis


class DefaultGroup(TyperGroup):
    """Custom TyperGroup that routes bare file path arguments to the 'run' command by default."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args.insert(0, "run")
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=DefaultGroup,
    name="datalens",
    help="DataLens — fast, streaming map-reduce statistics for tabular datasets (Parquet, CSV, TSV, JSONL).",
    add_completion=False,
)
console = Console()
logger = logging.getLogger("datalens")


@app.command(name="run")
def run_command(
    config_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to YAML configuration file (e.g. config.yaml).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Optional path to write the result as a JSON file.",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "-w",
        "--workers",
        help="Number of worker processes (overrides config; 1 = serial, None = all CPUs).",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "-b",
        "--batch-size",
        help="Record batch size (overrides config).",
    ),
    plots: bool = typer.Option(
        True,
        "--plots/--no-plots",
        help="Generate statistical distribution plots and figures (default: enabled).",
    ),
    plot_dir: Optional[Path] = typer.Option(
        None,
        "--plot-dir",
        help="Directory to save generated plots (default: 'plots').",
    ),
    plot_format: str = typer.Option(
        "png",
        "--plot-format",
        help="Plot image format ('png', 'svg', 'pdf').",
    ),
    plot_style: str = typer.Option(
        "datalens",
        "--plot-style",
        help="Plot style theme ('datalens', 'paper', 'dark').",
    ),
    captions: bool = typer.Option(
        True,
        "--captions/--no-captions",
        help="Include in-figure titles and statistical callout boxes. Set to --no-captions for academic papers.",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Enable debug-level logging.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress console output except errors.",
    ),
) -> None:
    """Execute dataset statistics analysis from a configuration file."""
    log_level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        config = load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Error loading config:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1)

    if workers is not None:
        config.workers = workers
    if batch_size is not None:
        config.batch_size = batch_size

    try:
        result = run_analysis(config)
    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        console.print(f"[red]Analysis failed:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1)

    if output:
        result.to_json(output)
        if not quiet:
            console.print(f"[green]Results written to[/green] {output}")

    if plots:
        target_dir = plot_dir or getattr(config, "plot_dir", None) or Path("plots")
        try:
            generated = result.plot(
                out_dir=target_dir,
                format=plot_format,
                captions=captions,
                style=plot_style,
            )
            if not quiet and generated:
                console.print(f"[green]Generated {len(generated)} plot(s) in[/green] [bold]{target_dir}/[/bold]")
                for name, path in generated.items():
                    console.print(f"  • [cyan]{name}[/cyan] → {path.name}")
        except Exception as exc:
            logger.warning("Could not generate plots: %s", exc)
            if not quiet:
                console.print(f"[yellow]Warning: Could not generate plots:[/yellow] {exc}")

    if not quiet and not output:
        console.print_json(json.dumps(result.to_dict()))

    if result.files_failed > 0:
        raise typer.Exit(code=2)


@app.command(name="mcp")
def mcp_command(
    transport: str = typer.Option(
        "stdio",
        "-t",
        "--transport",
        help="MCP transport protocol ('stdio', 'sse').",
    ),
) -> None:
    """Start the DataLens Model Context Protocol (MCP) server for LLMs and AI agents."""
    try:
        from .mcp.server import run_mcp_server

        run_mcp_server(transport=transport)
    except ImportError as exc:
        console.print(f"[red]Error:[/red] {exc}", highlight=False)
        console.print("Install MCP support with: [bold]pip install 'datalens[mcp]'[/bold]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
