"""Pure, side-effect-free parsers for the encodings found in MS2 parquet fields.

Each function takes a single raw value and returns a normalised token (or ``None``
when the value cannot be interpreted). Keeping these isolated and pure makes the
field-level encoding decisions easy to read, test, and override.
"""

from __future__ import annotations

import re

UNKNOWN = "UNKNOWN"

# `[UNIMOD:35]` style modification tags embedded in the canonical ``sequence`` field.
_UNIMOD_PATTERN = re.compile(r"\[UNIMOD:(\d+)\]")
# Leading mass-analyzer token of an mzML ``header`` filter string, e.g. ``FTMS`` / ``ITMS``.
_ANALYZER_PATTERN = re.compile(r"^\s*([A-Za-z]+)")


def project_from_usi(usi: str | None) -> str | None:
    """Extract the PRIDE project accession from a USI.

    A USI looks like ``mzspec:PXD000561:<run>:scan:<n>:<PEPTIDE>/<charge>``; the
    project is the second colon-delimited field.

    Args:
        usi: The Universal Spectrum Identifier string.

    Returns:
        The project accession (e.g. ``"PXD000561"``) or ``None`` if unparseable.
    """
    if not usi:
        return None
    parts = usi.split(":")
    return parts[1] if len(parts) > 1 and parts[1] else None


def run_from_usi(usi: str | None) -> str | None:
    """Extract the run/sample identifier (third USI field).

    Args:
        usi: The Universal Spectrum Identifier string.

    Returns:
        The run identifier or ``None`` if unparseable.
    """
    if not usi:
        return None
    parts = usi.split(":")
    return parts[2] if len(parts) > 2 and parts[2] else None


def organism_from_protein(protein: str | None) -> str:
    """Derive a UniProt organism mnemonic from a ``protein`` identifier.

    Values look like ``sp|Q05682|CALD1_HUMAN``; the organism is the token after the
    final underscore of the entry name. This is a proxy: contaminant proteins carry
    their own organism, and non-UniProt databases may not follow the convention.

    Args:
        protein: The protein identifier string (the first entry is used when several
            are concatenated with ``;``).

    Returns:
        The organism mnemonic (e.g. ``"HUMAN"``) or :data:`UNKNOWN`.
    """
    if not protein:
        return UNKNOWN
    first = protein.split(";", 1)[0]
    entry = first.split("|")[-1]
    if "_" not in entry:
        return UNKNOWN
    mnemonic = entry.rsplit("_", 1)[-1].strip().upper()
    return mnemonic or UNKNOWN


def analyzer_class_from_header(header: str | None) -> str:
    """Extract the mass-analyzer class from an mzML header filter string.

    The header begins with a token such as ``FTMS`` or ``ITMS``. This is the only
    instrument signal available in the data; the actual instrument *model* is not
    recorded here.

    Args:
        header: The mzML filter/header string.

    Returns:
        The upper-cased analyzer token (e.g. ``"FTMS"``) or :data:`UNKNOWN`.
    """
    if not header:
        return UNKNOWN
    match = _ANALYZER_PATTERN.match(header)
    return match.group(1).upper() if match else UNKNOWN


def modifications_from_sequence(sequence: str | None) -> list[str]:
    """Return the list of UNIMOD modification accessions found in a sequence.

    Modifications are encoded inline in the canonical ``sequence`` field as
    ``[UNIMOD:<id>]``. The same modification may legitimately appear multiple times
    in one sequence, so duplicates are preserved.

    Args:
        sequence: The UNIMOD-encoded peptide sequence.

    Returns:
        A list of accession strings such as ``["UNIMOD:35", "UNIMOD:4"]``.
    """
    if not sequence:
        return []
    return [f"UNIMOD:{acc}" for acc in _UNIMOD_PATTERN.findall(sequence)]
