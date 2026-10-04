"""Resolve project and run identifiers from paths or parquet columns.

Column names are never hard-coded here — callers pass a :class:`FieldMap` that
maps standard logical names (``project``, ``run``, ``filepath``) to whatever
physical names the client parquet files use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from instanovo.dataset_stats.config import FieldMap

Source = Literal["auto", "column", "path"]
RunSource = Literal["auto", "column", "path", "filename"]

_DEFAULT_PROJECT_PATTERN = r"(PXD\d+|MSV\d+|JPST\d+)"
UNRESOLVED = "__UNRESOLVED__"
DEFAULT_PROJECT_PATTERN = _DEFAULT_PROJECT_PATTERN


@dataclass(frozen=True)
class IdentityConfig:
    """Rules for resolving project/run identity from paths or parquet columns.

    Path indices count **from the file upward** (right to left):

    * ``0`` — file stem (``run.parquet`` → ``run``)
    * ``1`` — immediate parent directory
    * ``2`` — grandparent directory, and so on

    Which parquet column to read is determined by :class:`FieldMap` together
    with ``project_source`` / ``run_source``:

    * ``project_source=column`` → logical ``project`` column
    * ``project_source=path`` → logical ``filepath`` column (parsed as a path)
    * ``run_source=column`` → logical ``run`` column
    * ``run_source=path`` → logical ``filepath`` column
    * ``run_source=filename`` → file stem (path index 0)
    """

    project_source: Source = "auto"
    run_source: RunSource = "auto"
    project_path_index: int | None = None
    run_path_index: int | None = None
    project_pattern: str = _DEFAULT_PROJECT_PATTERN
    run_pattern: str | None = None

    def resolves_project_per_row(self) -> bool:
        """Return whether project must be derived from row values."""
        return self.project_source in ("column", "path")

    def resolves_run_per_row(self) -> bool:
        """Return whether run should be read from row values."""
        return self.run_source in ("column", "path")

    def project_field(self, fields: FieldMap) -> str | None:
        """Physical column used for row-level project resolution."""
        if self.project_source == "column":
            return fields.project
        if self.project_source == "path":
            return fields.filepath
        return None

    def run_field(self, fields: FieldMap) -> str | None:
        """Physical column used for row-level run resolution."""
        if self.run_source == "column":
            return fields.run
        if self.run_source == "path":
            return fields.filepath
        return None

    def row_identity_columns(self, fields: FieldMap) -> set[str]:
        """Parquet columns that must be read to resolve row-level identity."""
        columns: set[str] = set()
        if self.resolves_project_per_row():
            column = self.project_field(fields)
            if column:
                columns.add(column)
        if self.resolves_run_per_row():
            column = self.run_field(fields)
            if column:
                columns.add(column)
        return columns


def path_components_from_right(path: str | Path) -> list[str]:
    """Return path components indexed from the file upward.

    For ``tier/PXD/run.parquet`` this yields ``["run", "PXD", "tier", ...]``.
    """
    p = Path(str(path))
    stem = p.name
    if stem.endswith(".parquet"):
        stem = stem[: -len(".parquet")]
    elif stem.endswith(".parq"):
        stem = stem[: -len(".parq")]
    return [stem, *reversed(p.parent.parts)]


def component_at_index(path: str | Path, index: int) -> str | None:
    """Return the path component at ``index`` (right-to-left, file stem at 0)."""
    if index < 0:
        return None
    components = path_components_from_right(path)
    return components[index] if index < len(components) else None


def extract_pattern(text: str, pattern: str) -> str | None:
    """Return the first regex match in ``text``, preferring capture group 1."""
    if not text:
        return None
    match = re.search(pattern, text)
    if not match:
        return None
    if match.lastindex:
        return match.group(1).upper()
    return match.group(0).upper()


def resolve_project_from_path(path: str | Path, identity: IdentityConfig) -> str | None:
    """Resolve a project ID from a filesystem or row-level path string."""
    text = str(path)
    if identity.project_path_index is not None:
        segment = component_at_index(text, identity.project_path_index)
        if segment:
            matched = extract_pattern(segment, identity.project_pattern)
            if matched:
                return matched
            return segment
    return extract_pattern(text, identity.project_pattern)


def resolve_run_from_path(path: str | Path, identity: IdentityConfig) -> str | None:
    """Resolve a run name from a filesystem or row-level path string."""
    text = str(path)
    pattern = identity.run_pattern or identity.project_pattern
    if identity.run_path_index is not None:
        segment = component_at_index(text, identity.run_path_index)
        if segment:
            if identity.run_pattern:
                return extract_pattern(segment, pattern) or segment
            return segment
    if identity.run_source == "filename" or identity.run_path_index == 0:
        return component_at_index(text, 0)
    if identity.run_pattern:
        return extract_pattern(text, pattern)
    return component_at_index(text, 0)


def resolve_project(value: object, identity: IdentityConfig, *, file_path: str) -> str:
    """Resolve project from a row value and/or file path according to ``identity``."""
    source = identity.project_source
    if source == "column":
        text = _stringify(value)
        return text.upper() if text else UNRESOLVED

    path_text = _stringify(value) if source == "path" else file_path
    resolved = resolve_project_from_path(path_text, identity)
    if resolved:
        return resolved

    if source == "auto":
        resolved = resolve_project_from_path(file_path, identity)
        if resolved:
            return resolved
        segment = component_at_index(file_path, 1)
        if segment:
            matched = extract_pattern(segment, identity.project_pattern)
            if matched:
                return matched

    return UNRESOLVED


def resolve_run(value: object, identity: IdentityConfig, *, file_path: str) -> str | None:
    """Resolve run from a row value and/or file path according to ``identity``."""
    source = identity.run_source
    if source == "column":
        return _stringify(value)

    path_text = _stringify(value) if source == "path" else file_path
    if source in ("path", "filename", "auto"):
        return resolve_run_from_path(path_text, identity)
    return None


def discovery_project(file_path: str, identity: IdentityConfig) -> str:
    """Resolve the file-level project used for task scheduling (may be a placeholder)."""
    if identity.resolves_project_per_row():
        return UNRESOLVED
    resolved = resolve_project_from_path(file_path, identity)
    if resolved:
        return resolved
    if identity.project_source == "auto":
        segment = component_at_index(file_path, 1)
        if segment:
            matched = extract_pattern(segment, identity.project_pattern)
            if matched:
                return matched
    return UNRESOLVED


def _stringify(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
