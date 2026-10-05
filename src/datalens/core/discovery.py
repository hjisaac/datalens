"""Discover map input splits by walking the folder tree or resolving provided sources."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from .types import DataTask, FileTask, Task


def discover_tasks(config: Any, root: Path | str | None = None) -> list[FileTask]:
    """Walk ``root`` (or ``config.root``) and return one :class:`FileTask` per matching file.

    The partition key for each task is the sequence of folder names from
    ``root`` down to the file's parent directory. For example::

        root/lcfm/PXD000561/run1.parquet  →  partition ("lcfm", "PXD000561")
        root/train_0.parquet              →  partition ()
    """
    target_root = root if root is not None else getattr(config, "root", None)
    if target_root is None:
        raise ValueError("No root directory specified in config or arguments.")
    root_path = Path(target_root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root_path}")

    tasks: list[FileTask] = []
    pattern = getattr(config, "file_pattern", "*.parquet")
    depth = getattr(config, "partition_depth", None)

    for dirpath, _, filenames in os.walk(root_path):
        for name in sorted(filenames):
            if not fnmatch.fnmatch(name, pattern):
                continue
            path = Path(dirpath) / name
            partition = _partition_key(root_path, path.parent, depth)
            tasks.append(FileTask(path=str(path), partition=partition))
    tasks.sort(key=lambda task: (task.partition, task.path))
    return tasks


def resolve_tasks(config: Any, source: Any = None) -> list[Task]:
    """Normalize any input source into a list of unified :class:`Task` splits."""
    if source is not None:
        if isinstance(source, Task):
            return [source]

        if isinstance(source, (str, Path)):
            src_path = Path(source).resolve()
            if src_path.is_dir():
                return discover_tasks(config, root=src_path)
            if src_path.is_file():
                return [FileTask(path=str(src_path))]
            raise FileNotFoundError(f"Source path not found: {source}")

        if isinstance(source, list):
            if not source:
                return []
            first = source[0]
            if isinstance(first, Task):
                return list(source)
            if isinstance(first, (str, Path)):
                return [FileTask(path=str(p)) for p in source]
            if isinstance(first, dict):
                return [DataTask(data=source)]
            raise TypeError(f"Unsupported task item type in list: {type(first)!r}")

        if isinstance(source, dict):
            columns = getattr(config, "columns", {})
            is_columnar = bool(columns and all(k in columns for k in source.keys()))
            if is_columnar:
                return [DataTask(data=source)]

            tasks: list[Task] = []
            for k, v in source.items():
                part = k if isinstance(k, tuple) else (str(k),)
                tasks.append(DataTask(data=v, partition=part))
            return tasks

        # PyArrow Table, DataFrame, or other in-memory object
        return [DataTask(data=source)]

    if getattr(config, "root", None) is not None:
        root_path = Path(config.root).resolve()
        if root_path.is_file():
            return [FileTask(path=str(root_path))]
        return discover_tasks(config)

    raise ValueError("No input source provided and config has no 'root' directory.")


def _partition_key(root: Path, parent: Path, depth: int | None) -> tuple[str, ...]:
    """Derive the shuffle partition key from a file's parent directory."""
    relative = parent.resolve().relative_to(root.resolve())
    if not relative.parts:
        return ()
    parts = relative.parts
    if depth is not None:
        if depth <= 0:
            return ()
        parts = parts[:depth]
    return tuple(parts)
