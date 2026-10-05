"""Model Context Protocol (MCP) server for DataLens.

Exposes dataset inspection, streaming Map-Reduce statistics, and automated
profiling as tools, resources, and prompts for AI coding assistants and LLMs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as MCPServer
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required to run the DataLens MCP server. "
            "Install it with: pip install 'datalens[mcp]' or pip install mcp"
        ) from exc

from .tools import (
    SUPPORTED_EXTENSIONS,
    compute_statistics as _compute_statistics,
    generate_dataset_plots as _generate_dataset_plots,
    inspect_dataset as _inspect_dataset,
    profile_dataset as _profile_dataset,
)

logger = logging.getLogger("datalens.mcp")


def create_mcp_server() -> MCPServer:
    """Create and configure the DataLens MCP server instance."""
    server = MCPServer(
        name="datalens",
        version="0.1.0",
        instructions=(
            "DataLens MCP server: fast, streaming Map-Reduce statistics for tabular datasets "
            "(Parquet, CSV, TSV, JSONL). Use 'inspect_dataset' to check schemas and column types, "
            "'compute_statistics' to run custom streaming statistics, or 'profile_dataset' "
            "for one-click comprehensive dataset profiling."
        ),
    )

    @server.tool(
        name="inspect_dataset",
        description=(
            "Inspect a tabular dataset file or directory without loading everything into memory. "
            "Infers column types (numeric, boolean, string), null rates, sample values, "
            "and recommends optimal accumulator kinds (numeric, quantile, categorical, cardinality)."
        ),
    )
    def inspect_dataset(path: str, sample_size: int = 100) -> str:
        """Inspect schema, sample values, and recommend statistics configuration."""
        data = _inspect_dataset(path=path, sample_size=sample_size)
        return json.dumps(data, indent=2, default=str)

    @server.tool(
        name="compute_statistics",
        description=(
            "Execute streaming Map-Reduce statistics on a tabular dataset. "
            "Computes exact/streaming metrics in bounded memory: "
            "'numeric' (mean, std, min, max), 'quantile' (T-Digest percentiles p01-p99, IQR), "
            "'categorical' (top frequency tables), 'cardinality' (distinct counts), and row counts."
        ),
    )
    def compute_statistics(
        path: str,
        columns: dict[str, str],
        file_pattern: str | None = None,
        partition_depth: int | None = None,
        quantiles: list[float] | None = None,
        top_categories: int = 10,
        workers: int | None = None,
        batch_size: int = 65_536,
    ) -> str:
        """Compute parallel streaming dataset statistics."""
        data = _compute_statistics(
            path=path,
            columns=columns,
            file_pattern=file_pattern,
            partition_depth=partition_depth,
            quantiles=quantiles,
            top_categories=top_categories,
            workers=workers,
            batch_size=batch_size,
        )
        return json.dumps(data, indent=2, default=str)

    @server.tool(
        name="profile_dataset",
        description=(
            "One-click complete profiling of any tabular dataset (Parquet, CSV, TSV, JSONL). "
            "Automatically inspects schema, selects appropriate accumulators, and runs streaming "
            "Map-Reduce statistics across all partitions."
        ),
    )
    def profile_dataset(
        path: str,
        include_quantiles: bool = True,
        max_categories: int = 10,
        workers: int | None = None,
    ) -> str:
        """Automated end-to-end dataset profiling."""
        data = _profile_dataset(
            path=path,
            include_quantiles=include_quantiles,
            max_categories=max_categories,
            workers=workers,
        )
        return json.dumps(data, indent=2, default=str)

    @server.tool(
        name="generate_dataset_plots",
        description=(
            "Generate publication-ready statistical figures (quantile distributions, category frequencies, "
            "partition comparisons) for any dataset. Supports 'datalens', 'paper' (academic), and 'dark' styles, "
            "and optional figure captions for paper inclusion."
        ),
    )
    def generate_dataset_plots(
        path: str,
        out_dir: str = "plots",
        format: str = "png",
        captions: bool = True,
        style: str = "datalens",
    ) -> str:
        """Generate statistical figures on disk."""
        data = _generate_dataset_plots(
            path=path,
            out_dir=out_dir,
            format=format,
            captions=captions,
            style=style,
        )
        return json.dumps(data, indent=2, default=str)

    @server.resource(
        uri="datalens://workspace/datasets",
        name="Workspace Datasets",
        description="Lists supported tabular dataset files in the current working directory.",
        mime_type="application/json",
    )
    def list_workspace_datasets() -> str:
        """Scan current directory for supported tabular dataset files."""
        cwd = Path.cwd()
        found: list[dict[str, Any]] = []
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in cwd.glob(f"*{ext}"):
                if file_path.is_file():
                    found.append(
                        {
                            "name": file_path.name,
                            "path": str(file_path.resolve()),
                            "size_bytes": file_path.stat().st_size,
                            "extension": ext,
                        }
                    )
        return json.dumps(found, indent=2)

    @server.prompt(
        name="profile_and_analyze",
        description="Guided prompt to profile a dataset, analyze distributions, and report findings.",
    )
    def profile_and_analyze(path: str) -> str:
        """Generate analysis instructions for the assistant."""
        return (
            f"You are analyzing the dataset located at '{path}'.\n\n"
            "Please follow these analytical steps:\n"
            "1. Call 'inspect_dataset' to examine column names, inferred types, and null counts.\n"
            "2. Call 'profile_dataset' to compute streaming statistics (or 'compute_statistics' for targeted columns).\n"
            "3. Analyze distribution characteristics:\n"
            "   - Compare mean vs median (p50) to evaluate skewness.\n"
            "   - Check extreme tails (p99 vs max, p01 vs min) for outliers or data entry anomalies.\n"
            "   - Review categorical frequencies for dominant classes or high cardinality.\n"
            "4. Provide a structured analytical summary with observations, data quality flags, and recommendations."
        )

    return server


def run_mcp_server(transport: str = "stdio") -> None:
    """Run the DataLens MCP server using the specified transport (default: stdio)."""
    server = create_mcp_server()
    server.run(transport=transport)


def main() -> None:
    """Standalone CLI entry point for 'datalens-mcp'."""
    run_mcp_server(transport="stdio")


if __name__ == "__main__":
    main()
