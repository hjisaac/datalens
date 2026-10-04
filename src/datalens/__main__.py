"""Typer CLI entry point for datalens.

Usage:
    datalens config.yaml
    datalens config.yaml --output stats.json
    python -m datalens config.yaml -w 4
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .core.config import load_config
from .core.orchestrator import run_analysis

app = typer.Typer(
    name="datalens",
    help="DataLens — fast, streaming map-reduce statistics for tabular datasets (Parquet, CSV, TSV, JSONL).",
    add_completion=False,
)
console = Console()
logger = logging.getLogger("datalens")


@app.command()
def main(
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

    if not quiet and not output:
        console.print_json(json.dumps(result.to_dict()))

    if result.files_failed > 0:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
