"""
metadata — DICOM metadata extraction, comparison, and persistence utilities.

Every function is public and individually importable::

    from bin.dicom_api.metadata import (
        extract_metadata, extract_full_header, compare_metadata,
        load_metadata, save_metadata, save_image, load_image,
    )
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydicom.dataset import Dataset

__all__ = [
    "get_meta",
    "full_header",
    "acq_params",
    "pixel_stats",
    "window_info",
    "compare_meta",
    "merge_meta",
    "pick_meta",
    "flatten_meta",
    "batch_meta",
    "summarize",
    "save_meta",
    "load_meta",
    "meta_csv",
    "save_img",
    "load_img",
]

# DICOM fields to extract (core clinical subset).
_FIELDS: list[str] = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Modality",
    "ViewPosition",
    "Rows",
    "Columns",
    "PixelSpacing",
    "PhotometricInterpretation",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Extraction
# ═══════════════════════════════════════════════════════════════════════════

def get_meta(ds: Dataset) -> dict[str, Any]:
    """Extract selected clinical metadata from a DICOM dataset.

    Parameters
    ----------
    ds : pydicom.Dataset
        Parsed DICOM dataset.

    Returns
    -------
    dict[str, Any]
        Dictionary of extracted field values.  Missing fields are ``None``.
    """
    meta: dict[str, Any] = {}
    for field in _FIELDS:
        value = getattr(ds, field, None)
        if value is not None:
            value = _make_json_safe(value)
        meta[field] = value
    return meta


def full_header(ds: Dataset) -> dict[str, Any]:
    """Extract **every** non-pixel DICOM tag into a flat dictionary.

    Useful for exploratory analysis or debugging when you need to see all
    available header information.

    Parameters
    ----------
    ds : pydicom.Dataset
        Parsed DICOM dataset.

    Returns
    -------
    dict[str, Any]
        All tag names → values (JSON-safe types).
    """
    header: dict[str, Any] = {}
    for elem in ds:
        if elem.tag == (0x7FE0, 0x0010):  # skip PixelData
            continue
        name = elem.keyword if elem.keyword else str(elem.tag)
        try:
            header[name] = _make_json_safe(elem.value)
        except Exception:
            header[name] = str(elem.value)
    return header


def pixel_stats(ds: Dataset) -> dict[str, float | str | list[int]]:
    """Compute basic statistics on the raw pixel data.

    Parameters
    ----------
    ds : pydicom.Dataset
        Parsed DICOM dataset (must contain ``pixel_array``).

    Returns
    -------
    dict[str, float | str | list[int]]
        Keys: ``min``, ``max``, ``mean``, ``std``, ``median``,
        ``dtype``, ``shape``.
    """
    arr = ds.pixel_array.astype(np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "dtype": str(ds.pixel_array.dtype),
        "shape": list(ds.pixel_array.shape),
    }


def acq_params(ds: Dataset) -> dict[str, Any]:
    """Extract X-ray acquisition technique parameters.

    Covers standard radiography tags: tube voltage, current, exposure,
    and acquisition configuration.

    Parameters
    ----------
    ds : pydicom.Dataset
        Parsed DICOM dataset.

    Returns
    -------
    dict[str, Any]
        Keys: ``KVP``, ``ExposureTime``, ``XRayTubeCurrent``, ``Exposure``,
        ``ExposureInuAs``, ``FocalSpots``, ``FilterType``,
        ``RadiationSetting``, ``AcquisitionDeviceProcessingDescription``.
        Missing tags are ``None``.
    """
    _ACQ_FIELDS = [
        "KVP",
        "ExposureTime",
        "XRayTubeCurrent",
        "Exposure",
        "ExposureInuAs",
        "FocalSpots",
        "FilterType",
        "RadiationSetting",
        "AcquisitionDeviceProcessingDescription",
        "DistanceSourceToDetector",
        "DistanceSourceToPatient",
        "BodyPartExamined",
        "PatientPosition",
        "ViewPosition",
    ]
    params: dict[str, Any] = {}
    for field in _ACQ_FIELDS:
        value = getattr(ds, field, None)
        params[field] = _make_json_safe(value) if value is not None else None
    return params


def window_info(ds: Dataset) -> dict[str, float | str | None]:
    """Extract VOI windowing settings directly from a parsed Dataset.

    Parameters
    ----------
    ds : pydicom.Dataset
        Parsed DICOM dataset.

    Returns
    -------
    dict
        ``{"center": float | None, "width": float | None,
           "explanation": str | None}``
    """
    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)
    explanation = getattr(ds, "WindowCenterWidthExplanation", None)

    # Tags can hold multi-value sequences — use the first.
    def _first(val: Any) -> Any:
        if val is None:
            return None
        if hasattr(val, "__iter__") and not isinstance(val, str):
            items = list(val)
            return items[0] if items else None
        return val

    return {
        "center": float(_first(center)) if _first(center) is not None else None,
        "width": float(_first(width)) if _first(width) is not None else None,
        "explanation": str(_first(explanation)) if _first(explanation) is not None else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Comparison / merging
# ═══════════════════════════════════════════════════════════════════════════

def compare_meta(
    meta_a: dict[str, Any],
    meta_b: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compare two metadata dictionaries and report differences.

    Parameters
    ----------
    meta_a : dict[str, Any]
        First metadata dict.
    meta_b : dict[str, Any]
        Second metadata dict.

    Returns
    -------
    dict[str, dict[str, Any]]
        Only keys that differ.  Each value is
        ``{"a": value_in_a, "b": value_in_b}``.
    """
    all_keys = set(meta_a) | set(meta_b)
    diffs: dict[str, dict[str, Any]] = {}
    for key in sorted(all_keys):
        val_a = meta_a.get(key)
        val_b = meta_b.get(key)
        if val_a != val_b:
            diffs[key] = {"a": val_a, "b": val_b}
    return diffs


def merge_meta(
    *dicts: dict[str, Any],
    conflict: str = "last",
) -> dict[str, Any]:
    """Merge multiple metadata dictionaries into one.

    Parameters
    ----------
    *dicts : dict[str, Any]
        Metadata dictionaries to merge (positional args).
    conflict : str
        How to handle key conflicts.
        ``"last"`` (default) — last value wins.
        ``"first"`` — first non-None value wins.

    Returns
    -------
    dict[str, Any]
        Merged dictionary.
    """
    merged: dict[str, Any] = {}
    for d in dicts:
        for k, v in d.items():
            if conflict == "first" and k in merged and merged[k] is not None:
                continue
            merged[k] = v
    return merged


def pick_meta(
    meta: dict[str, Any],
    keys: list[str],
    missing: Any = None,
) -> dict[str, Any]:
    """Return a new metadata dict containing only the specified keys.

    Parameters
    ----------
    meta : dict[str, Any]
        Source metadata dictionary.
    keys : list[str]
        Keys to include in the output.
    missing : Any
        Value to use for keys not present in *meta*.  Default ``None``.

    Returns
    -------
    dict[str, Any]
        Filtered dictionary.
    """
    return {k: meta.get(k, missing) for k in keys}


def flatten_meta(
    meta: dict[str, Any],
    sep: str = ".",
    _prefix: str = "",
) -> dict[str, Any]:
    """Flatten a nested metadata dictionary into a single-level dict.

    Nested lists of dicts are **not** recursed — only nested dicts.

    Parameters
    ----------
    meta : dict[str, Any]
        Potentially nested metadata dictionary.
    sep : str
        Separator between key levels.  Default ``"."``.

    Returns
    -------
    dict[str, Any]
        Flat dictionary with dot-separated keys.

    Example
    -------
    >>> flatten_meta({"a": {"b": 1}, "c": 2})
    {"a.b": 1, "c": 2}
    """
    out: dict[str, Any] = {}
    for k, v in meta.items():
        full_key = f"{_prefix}{sep}{k}" if _prefix else k
        if isinstance(v, dict):
            out.update(flatten_meta(v, sep=sep, _prefix=full_key))
        else:
            out[full_key] = v
    return out


def batch_meta(
    paths: "list[str | Path]",
    on_error: str = "skip",
) -> list[dict[str, Any]]:
    """Extract clinical metadata from a list of DICOM file paths.

    Parameters
    ----------
    paths : list[str | Path]
        DICOM file paths to process.
    on_error : str
        ``"skip"`` (default) — log and skip unreadable files.
        ``"raise"`` — propagate the first exception.

    Returns
    -------
    list[dict[str, Any]]
        One metadata dict per successfully read file.  Each dict contains
        an extra ``"_path"`` key with the source file path.
    """
    import logging
    from pathlib import Path as _Path
    import pydicom as _pydicom

    _log = logging.getLogger(__name__)
    results: list[dict[str, Any]] = []
    for p in paths:
        p = _Path(p)
        try:
            ds = _pydicom.dcmread(str(p), stop_before_pixels=True)
            meta = get_meta(ds)
            meta["_path"] = str(p)
            results.append(meta)
        except Exception as exc:
            _log.warning("batch_meta — %s: %s", p.name, exc)
            if on_error == "raise":
                raise
    return results


def summarize(
    records: list[dict[str, Any]],
    numeric_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate statistics across a collection of metadata records.

    Parameters
    ----------
    records : list[dict[str, Any]]
        List of metadata dicts (e.g. from :func:`batch_extract_metadata`).
    numeric_keys : list[str] | None
        Keys for which to compute numeric stats (min/max/mean/std).
        ``None`` auto-detects numeric fields.

    Returns
    -------
    dict[str, Any]
        ``total``, ``field_coverage`` (% of records with each key non-None),
        ``unique_values`` (for string fields), ``numeric_stats`` (for
        numeric fields).
    """
    if not records:
        return {"total": 0}

    all_keys = list(dict.fromkeys(k for r in records for k in r))
    total = len(records)

    coverage: dict[str, float] = {}
    unique_vals: dict[str, list[Any]] = {}
    num_stats: dict[str, dict[str, float]] = {}

    for key in all_keys:
        values = [r[key] for r in records if key in r and r[key] is not None]
        coverage[key] = round(len(values) / total * 100, 1)

        if not values:
            continue

        # Determine whether to treat as numeric.
        is_numeric = all(isinstance(v, (int, float)) for v in values)
        if numeric_keys is not None:
            is_numeric = key in numeric_keys and is_numeric

        if is_numeric:
            fv = [float(v) for v in values]
            num_stats[key] = {
                "min": min(fv),
                "max": max(fv),
                "mean": statistics.mean(fv),
                "stdev": statistics.stdev(fv) if len(fv) > 1 else 0.0,
            }
        else:
            str_vals = [str(v) for v in values]
            unique_vals[key] = sorted(set(str_vals))

    return {
        "total": total,
        "field_coverage_pct": coverage,
        "unique_values": unique_vals,
        "numeric_stats": num_stats,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Persistence — JSON
# ═══════════════════════════════════════════════════════════════════════════

def save_meta(meta: dict[str, Any], path: str | Path) -> Path:
    """Persist metadata dictionary as a JSON file.

    Parameters
    ----------
    meta : dict[str, Any]
        Metadata dictionary.
    path : str | Path
        Destination file path.

    Returns
    -------
    Path
        Resolved path to the written JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return path.resolve()


def load_meta(path: str | Path) -> dict[str, Any]:
    """Load a metadata JSON file back into a dictionary.

    Parameters
    ----------
    path : str | Path
        Path to a ``.json`` metadata file.

    Returns
    -------
    dict[str, Any]
        Parsed metadata dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def meta_csv(
    metadata_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Export all JSON metadata files in a directory to a single CSV.

    Each JSON file becomes one row.  Keys become columns.

    Parameters
    ----------
    metadata_dir : str | Path
        Directory containing ``.json`` metadata files.
    output_path : str | Path
        Destination CSV file path.

    Returns
    -------
    Path
        Resolved path to the written CSV file.
    """
    metadata_dir = Path(metadata_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(metadata_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {metadata_dir}")

    rows: list[dict[str, Any]] = []
    for jf in json_files:
        data = load_meta(jf)
        data["_filename"] = jf.stem
        rows.append(data)

    # Collect all keys across every file for the CSV header.
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path.resolve()


# ═══════════════════════════════════════════════════════════════════════════
#  Persistence — Images
# ═══════════════════════════════════════════════════════════════════════════

def save_img(img: np.ndarray, path: str | Path) -> Path:
    """Save a NumPy image array to disk as a PNG file.

    Parameters
    ----------
    img : np.ndarray
        Image array (``uint8``, grayscale or BGR).
    path : str | Path
        Destination file path.

    Returns
    -------
    Path
        Resolved path to the written PNG file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path.resolve()


def load_img(path: str | Path, grayscale: bool = True) -> np.ndarray:
    """Load a saved image back into a NumPy array.

    Parameters
    ----------
    path : str | Path
        Path to the image file (PNG, JPEG, etc.).
    grayscale : bool
        If ``True`` (default), load as single-channel grayscale.

    Returns
    -------
    np.ndarray
        Image array (``uint8``).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_json_safe(value: Any) -> Any:
    """Convert pydicom-specific types to plain Python types."""
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        return [_make_json_safe(v) for v in value]
    try:
        if isinstance(value, (int, float, str, bool)):
            return value
        return float(value)
    except (TypeError, ValueError):
        return str(value)
