"""Driver: discover splits, run map, shuffle, and reduce."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from . import config as config_module
from .config import AnalysisConfig, boot_config
from .discovery import resolve_tasks
from .mapper import map_task_safe
from .reducer import reduce_group
from .shuffler import shuffle
from .types import AnalysisResult, PartialStats, Task

logger = logging.getLogger(__name__)


def run_analysis(
    config: AnalysisConfig | dict[str, Any] | Any,
    source: Any = None,
) -> AnalysisResult:
    """Execute the full map-shuffle-reduce statistics job.

    Args:
        config: An :class:`AnalysisConfig` or configuration dict.
        source: Optional data source. Can be a directory path, a list of file paths,
            in-memory records (list of dicts, columnar dict, PyArrow Table), or
            explicit :class:`Task` instances. If omitted, uses ``config.root``.

    Returns:
        :class:`AnalysisResult` with aggregated statistics by partition.
    """
    if not isinstance(config, AnalysisConfig):
        config = config_module.build_config(config)

    boot_config(config)

    tasks = resolve_tasks(config, source=source)
    if not tasks:
        logger.warning("No tasks found to analyze.")
        return AnalysisResult({}, 0, 0, config_module.config_summary(config))

    # In-memory tasks always run in-process to avoid IPC pickling overhead
    is_in_memory = any(getattr(task, "is_in_memory", False) for task in tasks)
    workers = 1 if is_in_memory else config_module.resolved_workers(config)

    logger.info("Mapping %d task(s) with %d worker(s) ...", len(tasks), workers)
    mapped, files_processed, files_failed = _map_all(tasks, workers)

    logger.info("Shuffling %d map output(s) ...", len(mapped))
    groups = shuffle(mapped)

    logger.info("Reducing %d partition(s) ...", len(groups))
    results = {
        key: reduce_group(key, partials)
        for key, partials in sorted(groups.items())
    }

    return AnalysisResult(
        groups=results,
        files_processed=files_processed,
        files_failed=files_failed,
        config_summary=config_module.config_summary(config),
    )


def _map_all(
    tasks: Sequence[Task],
    workers: int,
) -> tuple[list[tuple[tuple[str, ...], PartialStats]], int, int]:
    mapped: list[tuple[tuple[str, ...], PartialStats]] = []
    files_processed = 0
    files_failed = 0

    if workers == 1:
        # Same process as the caller, which already booted the config.
        for task in tasks:
            outcome = map_task_safe(task)
            if outcome is None:
                files_failed += 1
                continue
            mapped.append(outcome)
            files_processed += 1
        return mapped, files_processed, files_failed

    # Each worker boots its own copy of `config` once via the pool initializer,
    # rather than pickling it into every task submission.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=boot_config, initargs=(config_module.CONFIG,)
    ) as executor:
        futures = {executor.submit(_map_task_picklable, task): task for task in tasks}
        for future in as_completed(futures):
            outcome = future.result()
            if outcome is None:
                files_failed += 1
                continue
            mapped.append(outcome)
            files_processed += 1
    return mapped, files_processed, files_failed


def _map_task_picklable(task: Task):
    """Top-level picklable wrapper for process-pool map workers."""
    return map_task_safe(task)
