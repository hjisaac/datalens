"""DataLens MCP (Model Context Protocol) package."""

from .server import create_mcp_server, run_mcp_server
from .tools import compute_statistics, generate_dataset_plots, inspect_dataset, profile_dataset

__all__ = [
    "compute_statistics",
    "create_mcp_server",
    "generate_dataset_plots",
    "inspect_dataset",
    "profile_dataset",
    "run_mcp_server",
]
