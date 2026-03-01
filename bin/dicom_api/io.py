"""
io — DICOM file I/O utilities.

Reading, writing, discovery, and validation helpers for DICOM files.
Every function is public and individually importable::

    from bin.dicom_api.io import read_dcm, list_dcm, is_valid
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset

logger = logging.getLogger(__name__)

__all__ = [
    "read_dcm",
    "read_pixels",
    "read_header",
    "list_dcm",
    "is_valid",
    "has_pixels",
    "dcm_shape",
    "dcm_modality",
    "dcm_spacing",
    "verify_dcm",
    "read_batch",
    "copy_dcm",
    "anonymize",
    "write_dcm",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Reading
# ═══════════════════════════════════════════════════════════════════════════

def read_dcm(path: str | Path) -> Dataset:
    """Read a DICOM file and return the parsed ``pydicom.Dataset``.

    Parameters
    ----------
    path : str | Path
        File-system path to the ``.dicom`` file.

    Returns
    -------
    pydicom.Dataset
        Parsed DICOM dataset with pixel data.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    ValueError
        If the file contains no pixel data.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DICOM file not found: {path}")
    ds: Dataset = pydicom.dcmread(str(path))
    if not hasattr(ds, "pixel_array"):
        raise ValueError(f"DICOM file has no pixel data: {path}")
    return ds


def read_pixels(path: str | Path) -> np.ndarray:
    """Read a DICOM and return the raw pixel array.

    Parameters
    ----------
    path : str | Path
        File-system path to the ``.dicom`` file.

    Returns
    -------
    np.ndarray
        Raw pixel array straight from the DICOM (dtype as stored).
    """
    ds = read_dcm(path)
    return ds.pixel_array


def read_header(path: str | Path) -> Dataset:
    """Read only the DICOM header (no pixel data loaded).

    Significantly faster than :func:`read_dicom` when you only need
    metadata fields. Pixel data is *not* read into memory.

    Parameters
    ----------
    path : str | Path
        File-system path to the ``.dicom`` file.

    Returns
    -------
    pydicom.Dataset
        Parsed DICOM dataset **without** pixel data.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DICOM file not found: {path}")
    return pydicom.dcmread(str(path), stop_before_pixels=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Discovery / validation
# ═══════════════════════════════════════════════════════════════════════════

def list_dcm(
    directory: str | Path,
    pattern: str = "*.dicom",
    recursive: bool = False,
) -> list[Path]:
    """List all DICOM files in a directory.

    Parameters
    ----------
    directory : str | Path
        Root folder to scan.
    pattern : str
        Glob pattern to match filenames.  Default ``"*.dicom"``.
    recursive : bool
        If ``True``, search subdirectories recursively.  Default ``False``.

    Returns
    -------
    list[Path]
        Sorted list of matching file paths.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    glob_fn = directory.rglob if recursive else directory.glob
    return sorted(glob_fn(pattern))


def is_valid(path: str | Path) -> bool:
    """Check whether a file is a readable DICOM.

    Does **not** raise; returns ``False`` for any error.

    Parameters
    ----------
    path : str | Path
        File-system path.

    Returns
    -------
    bool
        ``True`` if the file can be parsed by pydicom.
    """
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True)
        return True
    except Exception:
        return False


def has_pixels(path: str | Path) -> bool:
    """Check whether a DICOM file contains pixel data.

    Parameters
    ----------
    path : str | Path
        File-system path to the ``.dicom`` file.

    Returns
    -------
    bool
        ``True`` if the DICOM contains a ``PixelData`` element.
    """
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=False)
        return hasattr(ds, "PixelData")
    except Exception:
        return False


def dcm_shape(path: str | Path) -> tuple[int, int]:
    """Return ``(Rows, Columns)`` from the DICOM header without reading pixels.

    Parameters
    ----------
    path : str | Path
        File-system path to the ``.dicom`` file.

    Returns
    -------
    tuple[int, int]
        ``(rows, columns)`` as stored in the DICOM header.

    Raises
    ------
    ValueError
        If ``Rows`` or ``Columns`` tags are missing.
    """
    ds = read_header(path)
    rows = getattr(ds, "Rows", None)
    cols = getattr(ds, "Columns", None)
    if rows is None or cols is None:
        raise ValueError(f"Rows/Columns tags missing in {path}")
    return (int(rows), int(cols))


# ═══════════════════════════════════════════════════════════════════════════
#  Writing / anonymization
# ═══════════════════════════════════════════════════════════════════════════

# Tags to remove for basic de-identification (a subset of DICOM PS3.15 E.1).
_PHI_TAGS: list[str] = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "OtherPatientIDs",
    "OtherPatientNames",
    "AccessionNumber",
]


def anonymize(
    ds: Dataset,
    tags: list[str] | None = None,
    keep_uids: bool = True,
) -> Dataset:
    """Remove protected health information (PHI) tags from a Dataset.

    This performs a **basic** tag-level de-identification suitable for
    research pipelines.  For full HIPAA compliance, use a validated
    de-identification tool.

    Parameters
    ----------
    ds : pydicom.Dataset
        Parsed DICOM dataset (modified **in-place** and returned).
    tags : list[str] | None
        List of DICOM tag names to remove.  ``None`` uses a built-in
        default list of common PHI tags.
    keep_uids : bool
        If ``True`` (default), ``StudyInstanceUID`` and
        ``SeriesInstanceUID`` are not removed.

    Returns
    -------
    pydicom.Dataset
        The same dataset with PHI tags deleted.
    """
    tags_to_remove = tags if tags is not None else _PHI_TAGS
    for tag_name in tags_to_remove:
        if hasattr(ds, tag_name):
            delattr(ds, tag_name)
    return ds


def dcm_modality(path: str | Path) -> str | None:
    """Read the ``Modality`` tag from the DICOM header without loading pixels.

    Parameters
    ----------
    path : str | Path
        File-system path to the ``.dicom`` file.

    Returns
    -------
    str | None
        Modality string (e.g. ``"CR"``, ``"DX"``) or ``None`` if absent.
    """
    ds = read_header(path)
    value = getattr(ds, "Modality", None)
    return str(value) if value is not None else None


def dcm_spacing(path: str | Path) -> tuple[float, float] | None:
    """Return pixel spacing ``(row_mm, col_mm)`` from the DICOM header.

    Parameters
    ----------
    path : str | Path
        File-system path to the ``.dicom`` file.

    Returns
    -------
    tuple[float, float] | None
        ``(row_spacing_mm, col_spacing_mm)`` or ``None`` if the tag is absent.
    """
    ds = read_header(path)
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is None or len(spacing) < 2:
        return None
    return (float(spacing[0]), float(spacing[1]))


def verify_dcm(path: str | Path) -> dict[str, bool | str | None]:
    """Perform a deep integrity check on a DICOM file.

    Attempts to read the full file including pixel data and checks for the
    most common failure modes.

    Parameters
    ----------
    path : str | Path
        File-system path.

    Returns
    -------
    dict
        Keys: ``valid`` (bool), ``has_pixels`` (bool), ``error`` (str or None).
    """
    path = Path(path)
    result: dict[str, bool | str | None] = {
        "valid": False,
        "has_pixels": False,
        "error": None,
    }
    try:
        ds = pydicom.dcmread(str(path))
        result["valid"] = True
        if hasattr(ds, "PixelData"):
            _ = ds.pixel_array  # force decode
            result["has_pixels"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def read_batch(
    paths: list[str | Path],
    on_error: str = "skip",
) -> dict[str, "Dataset"]:
    """Read multiple DICOM files and return a stem → Dataset mapping.

    Parameters
    ----------
    paths : list[str | Path]
        File paths to read.
    on_error : str
        ``"skip"`` (default) — log and skip failures.
        ``"raise"`` — propagate the first exception.

    Returns
    -------
    dict[str, pydicom.Dataset]
        ``{dicom_stem: dataset}`` for each successfully read file.
    """
    result: dict[str, "Dataset"] = {}
    for p in paths:
        p = Path(p)
        try:
            result[p.stem] = read_dcm(p)
        except Exception as exc:
            logger.warning("read_batch — %s: %s", p.name, exc)
            if on_error == "raise":
                raise
    return result


def copy_dcm(src: str | Path, dst: str | Path) -> Path:
    """Copy a DICOM file to a new location, creating parent directories.

    Parameters
    ----------
    src : str | Path
        Source ``.dicom`` file.
    dst : str | Path
        Destination path (file, not directory).

    Returns
    -------
    Path
        Resolved destination path.

    Raises
    ------
    FileNotFoundError
        If *src* does not exist.
    """
    import shutil

    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"Source DICOM not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.resolve()


# ═══════════════════════════════════════════════════════════════════════════
#  Writing / anonymization
# ═══════════════════════════════════════════════════════════════════════════

def write_dcm(ds: Dataset, path: str | Path) -> Path:
    """Write a pydicom Dataset back to disk.

    Parameters
    ----------
    ds : pydicom.Dataset
        Dataset to save.
    path : str | Path
        Destination file path.

    Returns
    -------
    Path
        Resolved path to the written DICOM file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path))
    return path.resolve()
