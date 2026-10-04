"""Load and normalise the external run-level metadata catalog (``search_data.csv``).

The catalog maps each ``(project, run)`` pair to curated run-level metadata
(organism, instrument, acquisition, fragmentation, …).  It is loaded once in
the runner before the process pool and passed as a plain dict so it pickles
cleanly across process boundaries.

The CSV schema expected::

    project, file path, workflow, acquisition, fragmentation, instrument, …, organism

where ``file path`` is a Windows-style absolute path whose basename — after
stripping the mzML / raw extension — is the run identifier that matches the
``experiment_name`` column in the parquet files.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

# (project_accession, run_basename) → CatalogEntry
RunKey = tuple[str, str]

# Strip recognised raw-file extensions (Windows paths, with optional .gz).
_RAW_SUFFIX = re.compile(r"\.(mzML|mzml|raw|RAW|mgf)(\.gz)?$")


@dataclass(frozen=True)
class CatalogEntry:
    """Curated metadata for one run (one raw file)."""

    organisms: tuple[str, ...]
    instrument: str | None
    acquisition: str | None = None
    fragmentations: tuple[str, ...] = ()
    detectors: tuple[str, ...] = ()
    enzyme: str | None = None


def _catalog_run_name(file_path: str) -> str:
    """Derive the run identifier from the catalog ``file path`` column.

    The value is a Windows absolute path; we take the basename and strip the
    raw-file extension to obtain the name that matches ``experiment_name`` in
    the parquet files.
    """
    basename = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    return _RAW_SUFFIX.sub("", basename)


def normalize_run_name(run: str) -> str:
    """Normalize a run identifier to the catalog key form.

    Strips optional ``.parquet`` and raw-file suffixes (``.mzML``, ``.raw``, …)
    so values from ``experiment_name``, file paths, and the catalog align.
    """
    text = run.strip()
    text = re.sub(r"\.parquet$", "", text, flags=re.IGNORECASE)
    return _RAW_SUFFIX.sub("", text)


def parquet_run_name(path: str) -> str:
    """Derive the run identifier from a parquet file path.

    Parquet files are named ``<run>[.mzML].parquet``; stripping both suffixes
    gives the run name that matches the catalog and ``experiment_name``.
    """
    basename = Path(path).name
    basename = re.sub(r"\.parquet$", "", basename, flags=re.IGNORECASE)
    return normalize_run_name(basename)


def _normalise_instrument(raw: object) -> str | None:
    """Normalise a raw instrument cell value to a consistent title-cased string.

    Applies the same casing convention as organisms so that ``Q Exactive HF``
    and ``q exactive hf`` collapse to the same key and are never counted as
    separate instruments.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip().title()


def _normalise_organism(raw: object) -> tuple[str, ...]:
    """Split and normalise a raw organism cell value into clean name tokens.

    Handles multi-organism cells (``A; B``), case inconsistencies
    (``Homo Sapiens`` / ``Homo sapiens``), and stray trailing whitespace —
    all observed in the real catalog file.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ()
    parts = [p.strip().title() for p in raw.split(";") if p.strip()]
    return tuple(dict.fromkeys(parts))  # deduplicate, preserve order


def _normalise_acquisition(raw: object) -> str | None:
    """Normalise a catalog acquisition cell (``DDA``, ``DIA``, …)."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip().upper()


def _normalise_fragmentations(raw: object) -> tuple[str, ...]:
    """Split multi-value fragmentation cells (``HCD;CID``, ``HCD|CID``, …)."""
    if not isinstance(raw, str) or not raw.strip():
        return ()
    parts = [p.strip().upper() for p in re.split(r"[;|]", raw) if p.strip()]
    return tuple(dict.fromkeys(parts))


def _normalise_detectors(raw: object) -> tuple[str, ...]:
    """Split multi-value detector cells (``Orbitrap|IonTrap``, …)."""
    if not isinstance(raw, str) or not raw.strip():
        return ()
    parts = [p.strip().title() for p in re.split(r"[;|]", raw) if p.strip()]
    return tuple(dict.fromkeys(parts))


def _normalise_enzyme(raw: object) -> str | None:
    """Normalise a catalog enzyme cell (``trypsin``, ``chymotrypsin``, …)."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip().title()


def load_catalog(path: str | Path) -> dict[RunKey, CatalogEntry]:
    """Load ``search_data.csv`` into a ``(project, run) → CatalogEntry`` mapping.

    Args:
        path: Path to the catalog CSV file.  The columns ``project``,
            ``file path``, ``instrument``, ``organism``, ``acquisition``,
            ``fragmentation``, ``detector``, and ``enzyme`` are used; all others
            are ignored.

    Returns:
        A plain dict that is picklable and safe to pass to worker processes.
    """
    catalog: dict[RunKey, CatalogEntry] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            project = (row.get("project") or "").strip()
            run = _catalog_run_name(row.get("file path") or "")
            if not project or not run:
                continue
            catalog[(project, run)] = CatalogEntry(
                organisms=_normalise_organism(row.get("organism")),
                instrument=_normalise_instrument(row.get("instrument")),
                acquisition=_normalise_acquisition(row.get("acquisition")),
                fragmentations=_normalise_fragmentations(row.get("fragmentation")),
                detectors=_normalise_detectors(row.get("detector")),
                enzyme=_normalise_enzyme(row.get("enzyme")),
            )
    return catalog
