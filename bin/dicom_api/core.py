"""
core — Pipeline orchestrator for the DICOM preprocessing API.

Provides single-file, batch, and streaming pipeline runners as well as
utilities for progress tracking and error reporting.

Every function is public and individually importable::

    from bin.dicom_api.core import run_pipeline, batch_preprocess, preview
"""

from __future__ import annotations

import csv
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from bin.dicom_api.io import read_dcm, list_dcm
from bin.dicom_api.preprocess import (
    process_px, rescale, invert, pct_clip, normalize, resize,
    to_uint8, to_float32, pad_square,
)
from bin.dicom_api.metadata import (
    get_meta, save_meta, save_img,
    full_header, pixel_stats, acq_params, window_info,
)

logger = logging.getLogger(__name__)

# Default output root (relative to the working directory).
_OUTPUT_ROOT = Path("processed_data")

__all__ = [
    "explain",
    "fix",
    "preprocess",
    "retry_run",
    "batch_run",
    "stream_run",
    "preview",
    "pipe_summary",
    "fix_orphans",
    "summary_csv",
    "validate",
    "clean_out",
]


# ═══════════════════════════════════════════════════════════════════════════
#  DICOM Explain — full diagnostic summary
# ═══════════════════════════════════════════════════════════════════════════

def explain(
    path: str | Path,
    print_report: bool = True,
    details: bool = False,
    image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return a **complete diagnostic summary** of a DICOM file.

    Gathers file info, patient/study metadata, pixel statistics,
    windowing settings, acquisition parameters, and identifies every
    preprocessing issue that needs attention before the image is ready
    for model training.

    Parameters
    ----------
    path : str | Path
        Path to the ``.dicom`` file.
    print_report : bool, optional
        If ``True`` (default), a human-readable report is printed to
        stdout in addition to returning the dict.
    details : bool, optional
        If ``False`` (default), prints a compact one-screen summary
        (issues + readiness + key image properties).
        If ``True``, prints the full report including pixel statistics,
        windowing, acquisition parameters, and all recommendations.
    image : np.ndarray | None, optional
        A **fixed** image array (e.g. from ``fix(path)["image"]``).
        When provided, pixel-level checks (range, bit depth, shape)
        are performed on this array instead of the raw DICOM pixels,
        so already-fixed issues no longer appear.

    Returns
    -------
    dict[str, Any]
        Structured summary with sections:
        ``file``, ``patient``, ``image``, ``pixel_stats``,
        ``windowing``, ``acquisition``, ``issues``, ``training_readiness``.

    Examples
    --------
    >>> from bin.dicom_api import explain, fix
    >>> report = explain("raw_data/sample.dicom")            # compact summary
    >>> result = fix("raw_data/sample.dicom")                # fix issues
    >>> report = explain("raw_data/sample.dicom", image=result["image"])  # verify fix
    """
    import pydicom as _pydicom

    path = Path(path)
    report: dict[str, Any] = {}

    # ── 1. File-level info ──────────────────────────────────────────────
    file_info: dict[str, Any] = {
        "filename": path.name,
        "stem": path.stem,
        "extension": path.suffix,
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path.is_file() else None,
        "exists": path.is_file(),
    }
    report["file"] = file_info

    if not path.is_file():
        report["error"] = f"File not found: {path}"
        if print_report:
            print(f"ERROR: File not found — {path}")
        return report

    # ── 2. Try reading the DICOM ────────────────────────────────────────
    try:
        ds = _pydicom.dcmread(str(path))
    except Exception as exc:
        report["error"] = f"Cannot read DICOM: {exc}"
        if print_report:
            print(f"ERROR: Cannot read DICOM — {exc}")
        return report

    has_px = hasattr(ds, "PixelData")
    file_info["valid_dicom"] = True
    file_info["has_pixel_data"] = has_px

    # ── 3. Patient / study metadata ─────────────────────────────────────
    meta = get_meta(ds)
    patient_info: dict[str, Any] = {
        "patient_id": meta.get("PatientID"),
        "study_uid": meta.get("StudyInstanceUID"),
        "series_uid": meta.get("SeriesInstanceUID"),
        "modality": meta.get("Modality"),
        "view_position": meta.get("ViewPosition"),
    }
    report["patient"] = patient_info

    # ── 4. Image properties ─────────────────────────────────────────────
    rows = getattr(ds, "Rows", None)
    cols = getattr(ds, "Columns", None)
    bits_allocated = getattr(ds, "BitsAllocated", None)
    bits_stored = getattr(ds, "BitsStored", None)
    high_bit = getattr(ds, "HighBit", None)
    pixel_rep = getattr(ds, "PixelRepresentation", None)
    samples = getattr(ds, "SamplesPerPixel", None)
    photometric = getattr(ds, "PhotometricInterpretation", None)
    spacing = getattr(ds, "PixelSpacing", None)
    rescale_slope = float(getattr(ds, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    transfer_syntax = str(getattr(ds.file_meta, "TransferSyntaxUID", "N/A")) if hasattr(ds, "file_meta") else "N/A"

    image_info: dict[str, Any] = {
        "rows": int(rows) if rows is not None else None,
        "columns": int(cols) if cols is not None else None,
        "is_square": (int(rows) == int(cols)) if rows is not None and cols is not None else None,
        "aspect_ratio": round(int(cols) / int(rows), 3) if rows and cols else None,
        "bits_allocated": bits_allocated,
        "bits_stored": bits_stored,
        "high_bit": high_bit,
        "pixel_representation": "signed" if pixel_rep == 1 else "unsigned" if pixel_rep == 0 else None,
        "samples_per_pixel": samples,
        "photometric_interpretation": str(photometric) if photometric else None,
        "pixel_spacing_mm": [float(spacing[0]), float(spacing[1])] if spacing and len(spacing) >= 2 else None,
        "rescale_slope": rescale_slope,
        "rescale_intercept": rescale_intercept,
        "transfer_syntax": transfer_syntax,
    }
    report["image"] = image_info

    # ── 5. Pixel statistics ─────────────────────────────────────────────
    # When a fixed image is supplied, analyse that instead of raw pixels.
    using_fixed = image is not None
    px_min_val: float = 0.0
    px_max_val: float = 0.0

    if using_fixed:
        fixed_arr = image.astype(np.float64)
        pstats_dict: dict[str, float | str | list[int]] = {
            "min": float(fixed_arr.min()),
            "max": float(fixed_arr.max()),
            "mean": float(fixed_arr.mean()),
            "std": float(fixed_arr.std()),
            "median": float(np.median(fixed_arr)),
            "dtype": str(image.dtype),
            "shape": list(image.shape),
        }
        report["pixel_stats"] = pstats_dict
        px_min_val = float(fixed_arr.min())
        px_max_val = float(fixed_arr.max())
        # Override image dimensions to match the fixed array
        image_info["rows"] = image.shape[0]
        image_info["columns"] = image.shape[1] if image.ndim >= 2 else image.shape[0]
        image_info["is_square"] = image.shape[0] == (image.shape[1] if image.ndim >= 2 else image.shape[0])
        report["source"] = "fixed_image"
    elif has_px:
        try:
            pstats = pixel_stats(ds)
            report["pixel_stats"] = pstats
            px_min_val = float(pstats["min"])  # type: ignore[arg-type]
            px_max_val = float(pstats["max"])  # type: ignore[arg-type]
        except Exception as exc:
            report["pixel_stats"] = {"error": str(exc)}
    else:
        report["pixel_stats"] = {"error": "No pixel data available"}

    # ── 6. Windowing ────────────────────────────────────────────────────
    winfo = window_info(ds)
    report["windowing"] = winfo

    # ── 7. Acquisition parameters ──────────────────────────────────────
    acq = acq_params(ds)
    report["acquisition"] = acq

    # ── 8. Identify issues & preprocessing needs ────────────────────────
    issues: list[dict[str, str]] = []
    recommendations: list[str] = []

    # --- No pixel data ---
    if not has_px and not using_fixed:
        issues.append({
            "severity": "CRITICAL",
            "issue": "No pixel data",
            "detail": "This DICOM has no PixelData element — cannot be used for training.",
        })

    # --- Rescale needed (only for raw DICOM, not for fixed image) ---
    if not using_fixed and (rescale_slope != 1.0 or rescale_intercept != 0.0):
        issues.append({
            "severity": "INFO",
            "issue": "Rescale required",
            "detail": f"RescaleSlope={rescale_slope}, RescaleIntercept={rescale_intercept}. "
                      "Raw pixel values must be transformed: pixel × slope + intercept.",
        })
        recommendations.append("Apply rescale() to convert raw stored values to meaningful units.")

    # --- MONOCHROME1 inversion (only for raw DICOM) ---
    if not using_fixed and photometric and str(photometric) == "MONOCHROME1":
        issues.append({
            "severity": "WARNING",
            "issue": "MONOCHROME1 — needs inversion",
            "detail": "Air is bright, bone is dark — opposite of standard convention. "
                      "Most models expect MONOCHROME2 (air=dark, bone=bright).",
        })
        recommendations.append("Apply invert() to flip to MONOCHROME2 convention.")

    # --- Non-square image ---
    check_rows = image_info["rows"]
    check_cols = image_info["columns"]
    if check_rows and check_cols and int(check_rows) != int(check_cols):
        issues.append({
            "severity": "INFO",
            "issue": "Non-square image",
            "detail": f"Image is {check_rows}×{check_cols}. Most models expect square input (e.g. 512×512).",
        })
        recommendations.append("Use pad_square() then resize(), or just resize() (may distort).")

    # --- Large image ---
    if check_rows and check_cols and (int(check_rows) > 3000 or int(check_cols) > 3000):
        issues.append({
            "severity": "INFO",
            "issue": "Very large image",
            "detail": f"Image is {check_rows}×{check_cols} pixels — will need significant downscaling.",
        })
        recommendations.append("Use resize() to bring down to 512 or 1024 for training.")

    # --- Pixel outliers / wide range ---
    if (has_px or using_fixed) and "error" not in report["pixel_stats"]:
        px_range = px_max_val - px_min_val
        if px_range > 10000:
            issues.append({
                "severity": "WARNING",
                "issue": "Very wide pixel range",
                "detail": f"Pixel range is {px_min_val:.0f}–{px_max_val:.0f} (range={px_range:.0f}). "
                          "Extreme outliers can compress useful contrast after normalization.",
            })
            recommendations.append("Apply pct_clip() to remove outlier intensities before normalize().")

        if px_min_val < -1000:
            issues.append({
                "severity": "INFO",
                "issue": "Negative pixel values (likely CT/HU)",
                "detail": f"Min pixel = {px_min_val:.0f}. Values below -1000 typically indicate "
                          "air in CT Hounsfield units.",
            })

    # --- Missing windowing ---
    if winfo["center"] is None or winfo["width"] is None:
        issues.append({
            "severity": "INFO",
            "issue": "No DICOM windowing tags",
            "detail": "WindowCenter / WindowWidth not set — cannot apply predefined windowing.",
        })
        recommendations.append("Use pct_clip() + normalize() instead of windowing().")
    else:
        recommendations.append(
            f"Can apply windowing(center={winfo['center']}, width={winfo['width']}) "
            "for standard radiology display."
        )

    # --- Missing pixel spacing ---
    if not spacing:
        issues.append({
            "severity": "INFO",
            "issue": "No pixel spacing",
            "detail": "PixelSpacing tag is absent — physical measurements unreliable.",
        })

    # --- Color image ---
    if using_fixed:
        is_color = image is not None and image.ndim == 3 and image.shape[-1] > 1
    else:
        is_color = samples is not None and int(samples) > 1
    if is_color:
        issues.append({
            "severity": "WARNING",
            "issue": "Color image (multi-sample)",
            "detail": f"SamplesPerPixel={samples}. Expected grayscale (1) for chest X-ray.",
        })
        recommendations.append("Convert to grayscale before training (e.g. cv2.cvtColor).")

    # --- High bit depth (skip for fixed uint8 images) ---
    if using_fixed:
        is_high_bits = image is not None and image.dtype != np.uint8
    else:
        is_high_bits = bits_stored is not None and int(bits_stored) > 12
    if is_high_bits:
        issues.append({
            "severity": "INFO",
            "issue": "High bit depth",
            "detail": f"BitsStored={bits_stored if not using_fixed else image.dtype}. " # type: ignore[union-attr]
                      "More than 12-bit — extra precision may "
                      "not be useful for CNN training.",
        })
        recommendations.append("Normalize to float32 [0,1] then convert to uint8 for storage.")

    report["issues"] = issues
    report["recommendations"] = recommendations

    # ── 9. Training readiness score ─────────────────────────────────────
    critical_count = sum(1 for i in issues if i["severity"] == "CRITICAL")
    warning_count = sum(1 for i in issues if i["severity"] == "WARNING")
    info_count = sum(1 for i in issues if i["severity"] == "INFO")

    if critical_count > 0:
        readiness = "NOT READY"
        ready_detail = "Critical issues must be resolved before this file can be used."
    elif warning_count > 0:
        readiness = "NEEDS PREPROCESSING"
        ready_detail = "Warnings need attention — apply recommended preprocessing steps."
    elif info_count > 0:
        readiness = "READY (minor notes)"
        ready_detail = "Usable with standard preprocessing pipeline (process_px)."
    else:
        readiness = "READY"
        ready_detail = "No issues found — ready for preprocessing and training."

    # Standard preprocessing pipeline summary
    pipeline_steps = [
        "1. read_dcm(path)          — Load DICOM file",
        "2. rescale(img, ds)        — Apply RescaleSlope/Intercept",
        "3. invert(img, ds)         — Fix MONOCHROME1 if needed",
        "4. pct_clip(img)           — Remove outlier intensities",
        "5. normalize(img)          — Scale to [0, 1]",
        "6. resize(img, 512)        — Resize to model input size",
        "7. to_uint8(img)           — Convert to uint8 for saving",
    ]
    shortcut = "Or simply: img = preprocess(path)  — runs all steps automatically."

    report["training_readiness"] = {
        "status": readiness,
        "detail": ready_detail,
        "critical_issues": critical_count,
        "warnings": warning_count,
        "info_notes": info_count,
        "pipeline_steps": pipeline_steps,
        "shortcut": shortcut,
    }

    # ── 10. Print human-readable report ─────────────────────────────────
    if print_report:
        _print_explain_report(report, details=details)

    return report


# ═══════════════════════════════════════════════════════════════════════════
#  Fix — auto-fix all issues found by explain()
# ═══════════════════════════════════════════════════════════════════════════

# Map issue names to internal fix keys.
_FIXABLE_ISSUES: dict[str, str] = {
    "Rescale required":                 "rescale",
    "MONOCHROME1 — needs inversion":    "invert",
    "Non-square image":                 "pad_square",
    "Very large image":                 "resize",
    "Very wide pixel range":            "pct_clip",
    "No DICOM windowing tags":          "normalize",
    "Color image (multi-sample)":       "grayscale",
    "High bit depth":                   "to_uint8",
}


def fix(
    path: str | Path,
    size: int = 512,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    save: bool = False,
    output_root: str | Path = _OUTPUT_ROOT,
    print_report: bool = True,
) -> dict[str, Any]:
    """Automatically fix all (or selected) issues found by :func:`explain`.

    Reads the DICOM, runs ``explain()`` to discover issues, then applies
    the appropriate preprocessing fix for every fixable issue.

    Parameters
    ----------
    path : str | Path
        Path to the ``.dicom`` file.
    size : int, optional
        Target image size for resize.  Default **512**.
    only : list[str] | None, optional
        If provided, fix **only** these issues (by fix-key name).
        Available fix keys: ``"rescale"``, ``"invert"``, ``"pad_square"``,
        ``"resize"``, ``"pct_clip"``, ``"normalize"``, ``"grayscale"``,
        ``"to_uint8"``.
        Default ``None`` → fix everything.
    skip : list[str] | None, optional
        Fix keys to **skip** even if the issue is found.
        Default ``None`` → skip nothing.
    save : bool, optional
        If ``True``, save the fixed PNG + metadata JSON to *output_root*.
        Default ``False``.
    output_root : str | Path, optional
        Output directory when *save* is ``True``.  Default ``processed_data/``.
    print_report : bool, optional
        If ``True`` (default), print a summary of applied fixes.

    Returns
    -------
    dict[str, Any]
        Keys:
        ``image`` — fixed ``np.ndarray`` (uint8, shape ``(size, size)``).
        ``applied`` — list of fix-key names that were applied.
        ``skipped`` — list of fix-key names that were skipped.
        ``unfixable`` — list of issue names that cannot be auto-fixed.
        ``issues_before`` — original issues from ``explain()``.

    Examples
    --------
    >>> from bin.dicom_api import fix
    >>> result = fix("raw_data/sample.dicom")
    >>> result["image"].shape
    (512, 512)
    >>>
    >>> # Fix only specific issues
    >>> result = fix("raw_data/sample.dicom", only=["rescale", "invert"])
    >>>
    >>> # Fix everything except resize
    >>> result = fix("raw_data/sample.dicom", skip=["resize"])
    """
    import cv2 as _cv2

    path = Path(path)
    only_set = set(only) if only else None
    skip_set = set(skip) if skip else set()

    # 1. Run explain to discover issues
    report = explain(path, print_report=False)

    if "error" in report:
        if print_report:
            print(f"ERROR: {report['error']}")
        return {
            "image": None,
            "applied": [],
            "skipped": [],
            "unfixable": [report["error"]],
            "issues_before": [],
        }

    issues = report.get("issues", [])

    # 2. Determine which fixes are needed
    needed_fixes: list[str] = []       # fix keys to apply
    skipped_fixes: list[str] = []      # fix keys user skipped
    unfixable: list[str] = []          # issues with no auto-fix

    for iss in issues:
        issue_name = iss["issue"]
        fix_key = _FIXABLE_ISSUES.get(issue_name)
        if fix_key is None:
            unfixable.append(issue_name)
            continue
        # If user specified 'only', skip anything not in it
        if only_set is not None and fix_key not in only_set:
            skipped_fixes.append(fix_key)
            continue
        # If user chose to skip this fix
        if fix_key in skip_set:
            skipped_fixes.append(fix_key)
            continue
        needed_fixes.append(fix_key)

    # 3. Read DICOM and apply fixes in the correct order
    ds = read_dcm(path)
    img = ds.pixel_array.astype(np.float32)

    applied: list[str] = []

    # Order matters — apply in logical preprocessing sequence:
    # grayscale → rescale → invert → pct_clip → normalize → pad_square → resize → to_uint8

    if "grayscale" in needed_fixes:
        if img.ndim == 3 and img.shape[-1] == 3:
            img = _cv2.cvtColor(img.astype(np.uint8), _cv2.COLOR_BGR2GRAY).astype(np.float32)
        elif img.ndim == 3:
            img = img[:, :, 0]  # take first channel
        applied.append("grayscale")

    if "rescale" in needed_fixes:
        img = rescale(img, ds)
        applied.append("rescale")

    if "invert" in needed_fixes:
        img = invert(img, ds)
        applied.append("invert")

    if "pct_clip" in needed_fixes:
        img = pct_clip(img)
        applied.append("pct_clip")

    if "normalize" in needed_fixes:
        img = normalize(img)
        applied.append("normalize")
    else:
        # Always normalize even if not flagged as issue
        img = normalize(img)

    if "pad_square" in needed_fixes:
        # Convert to uint8 temporarily for pad_square, then back
        img_u8 = to_uint8(img)
        img_u8 = pad_square(img_u8)
        img = img_u8.astype(np.float32) / 255.0
        applied.append("pad_square")

    if "resize" in needed_fixes:
        img = resize(img, size)
        applied.append("resize")
    else:
        # Always resize to target size
        img = resize(img, size)

    if "to_uint8" in needed_fixes:
        img = to_uint8(img)
        applied.append("to_uint8")
    else:
        img = to_uint8(img)

    # 4. Optionally save
    if save:
        output_root = Path(output_root)
        stem = path.stem
        save_img(img, output_root / "image" / f"{stem}.png")
        meta = get_meta(ds)
        save_meta(meta, output_root / "metadata" / f"{stem}.json")

    # 5. Post-fix analysis
    after_report = explain(path, print_report=False, image=img)
    after_issues = after_report.get("issues", [])
    after_readiness = after_report.get("training_readiness", {})

    # 6. Print summary
    if print_report:
        _print_fix_report(
            path.name, issues, applied, skipped_fixes, unfixable, img,
            after_issues, after_readiness,
        )

    return {
        "image": img,
        "applied": applied,
        "skipped": skipped_fixes,
        "unfixable": unfixable,
        "issues_before": issues,
        "issues_after": after_issues,
        "training_readiness": after_readiness,
    }


def _print_fix_report(
    filename: str,
    issues: list[dict[str, str]],
    applied: list[str],
    skipped: list[str],
    unfixable: list[str],
    img: np.ndarray,
    after_issues: list[dict[str, str]],
    after_readiness: dict[str, Any],
) -> None:
    """Print a human-readable summary of the fix() results."""
    sep = "─" * 70

    print(f"\n{'═' * 70}")
    print(f"  DICOM FIX REPORT — {filename}")
    print(f"{'═' * 70}")

    # ── BEFORE ────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  BEFORE FIX — issues found: {len(issues)}")
    print(sep)
    for iss in issues:
        sev = iss["severity"]
        marker = "✗" if sev == "CRITICAL" else "⚠" if sev == "WARNING" else "•"
        fix_key = _FIXABLE_ISSUES.get(iss["issue"])
        tag = f"  → fixable ({fix_key})" if fix_key else "  → not auto-fixable"
        print(f"  {marker} [{sev}] {iss['issue']}{tag}")

    # ── Applied ───────────────────────────────────────────────
    if applied:
        print(f"\n{sep}")
        print(f"  FIXES APPLIED ({len(applied)})")
        print(sep)
        for i, fx in enumerate(applied, 1):
            print(f"  ✓ {i}. {fx}")
    else:
        print(f"\n  ✓ No fixes were needed or applied.")

    if skipped:
        print(f"\n{sep}")
        print(f"  SKIPPED ({len(skipped)})")
        print(sep)
        for fx in skipped:
            print(f"  ⊘ {fx}")

    if unfixable:
        print(f"\n{sep}")
        print(f"  NOT AUTO-FIXABLE ({len(unfixable)})")
        print(sep)
        for u in unfixable:
            print(f"  ✗ {u}")

    # ── AFTER ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  AFTER FIX — remaining issues: {len(after_issues)}")
    print(sep)
    if not after_issues:
        print("  ✓ All issues resolved!")
    else:
        for iss in after_issues:
            sev = iss["severity"]
            marker = "✗" if sev == "CRITICAL" else "⚠" if sev == "WARNING" else "•"
            print(f"  {marker} [{sev}] {iss['issue']}")

    # ── Output ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  OUTPUT IMAGE")
    print(sep)
    print(f"  Shape    : {img.shape}")
    print(f"  dtype    : {img.dtype}")
    print(f"  Range    : [{img.min()}, {img.max()}]")

    # ── Readiness ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  TRAINING READINESS")
    print(sep)
    print(f"  Status   : {after_readiness.get('status', '?')}")
    print(f"  Detail   : {after_readiness.get('detail', '')}")

    print(f"\n{'═' * 70}\n")


def _print_explain_report(report: dict[str, Any], details: bool = False) -> None:
    """Pretty-print the explain() report to stdout.

    Parameters
    ----------
    details : bool
        ``False`` → compact one-screen summary.
        ``True``  → full report with all sections.
    """
    sep = "─" * 70
    f = report["file"]

    print(f"\n{'═' * 70}")
    print(f"  DICOM DIAGNOSTIC REPORT — {f['filename']}")
    if report.get("source") == "fixed_image":
        print("  (analyzing fixed image — not the raw DICOM pixels)")
    print(f"{'═' * 70}")

    # ── Always shown ────────────────────────────────────────────────────

    # Core file + image snapshot (one block)
    p = report.get("patient", {})
    im = report.get("image", {})
    print(f"\n{sep}")
    print("  OVERVIEW")
    print(sep)
    print(f"  File         : {f['filename']}  ({f.get('size_mb', '?')} MB)")
    print(f"  Valid DICOM  : {f.get('valid_dicom', False)}   "
          f"Has Pixels : {f.get('has_pixel_data', False)}")
    print(f"  Modality     : {p.get('modality', 'N/A')}   "
          f"View : {p.get('view_position', 'N/A')}")
    print(f"  Dimensions   : {im.get('rows')} × {im.get('columns')}  "
          f"{'(square)' if im.get('is_square') else '(non-square)'}")
    print(f"  Photometric  : {im.get('photometric_interpretation', 'N/A')}   "
          f"Bits stored : {im.get('bits_stored', 'N/A')}")
    rescale_needed = im.get('rescale_slope', 1.0) != 1.0 or im.get('rescale_intercept', 0.0) != 0.0
    print(f"  Rescale      : slope={im.get('rescale_slope')}  intercept={im.get('rescale_intercept')}  "
          f"{'⚠ needed' if rescale_needed else '✓ none'}")
    w = report.get("windowing", {})
    print(f"  Windowing    : center={w.get('center', 'N/A')}  width={w.get('width', 'N/A')}")

    # Issues — always show
    issues = report.get("issues", [])
    print(f"\n{sep}")
    print(f"  ISSUES FOUND ({len(issues)})")
    print(sep)
    if not issues:
        print("  ✓ No issues detected.")
    else:
        for iss in issues:
            severity = iss["severity"]
            marker = "✗" if severity == "CRITICAL" else "⚠" if severity == "WARNING" else "•"
            if details:
                print(f"  {marker} [{severity}] {iss['issue']}")
                print(f"    {iss['detail']}")
            else:
                print(f"  {marker} [{severity}] {iss['issue']}")

    # Recommendations — always show
    recs = report.get("recommendations", [])
    if recs:
        print(f"\n{sep}")
        print("  RECOMMENDATIONS")
        print(sep)
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r}")

    # Training readiness — always show
    tr = report.get("training_readiness", {})
    print(f"\n{sep}")
    print("  TRAINING READINESS")
    print(sep)
    print(f"  Status       : {tr.get('status', '?')}")
    print(f"  Detail       : {tr.get('detail', '')}")
    if not details:
        print(f"  Shortcut     : {tr.get('shortcut', '')}")
        print(f"\n  Tip: use explain(path, details=True) for full report")

    # ── Only shown when details=True ─────────────────────────────────────
    if details:
        # Patient / study
        print(f"\n{sep}")
        print("  PATIENT / STUDY")
        print(sep)
        print(f"  Patient ID   : {p.get('patient_id', 'N/A')}")
        print(f"  Study UID    : {str(p.get('study_uid', 'N/A'))[:55]}")
        print(f"  Series UID   : {str(p.get('series_uid', 'N/A'))[:55]}")

        # Full image properties
        print(f"\n{sep}")
        print("  IMAGE PROPERTIES (full)")
        print(sep)
        print(f"  Dimensions   : {im.get('rows')} × {im.get('columns')}")
        print(f"  Aspect Ratio : {im.get('aspect_ratio')}")
        print(f"  Bits         : {im.get('bits_allocated')}-bit allocated, {im.get('bits_stored')}-bit stored")
        print(f"  High Bit     : {im.get('high_bit')}")
        print(f"  Pixel Rep    : {im.get('pixel_representation')}")
        print(f"  Samples/Pixel: {im.get('samples_per_pixel')}")
        print(f"  Photometric  : {im.get('photometric_interpretation')}")
        print(f"  Pixel Spacing: {im.get('pixel_spacing_mm')} mm")
        print(f"  Rescale      : slope={im.get('rescale_slope')}, intercept={im.get('rescale_intercept')}")
        print(f"  Transfer UID : {im.get('transfer_syntax')}")

        # Pixel stats
        if "pixel_stats" in report and "error" not in report["pixel_stats"]:
            ps = report["pixel_stats"]
            print(f"\n{sep}")
            print("  PIXEL STATISTICS (raw stored values)")
            print(sep)
            print(f"  dtype        : {ps.get('dtype')}")
            print(f"  shape        : {ps.get('shape')}")
            print(f"  min          : {ps.get('min', 0):.2f}")
            print(f"  max          : {ps.get('max', 0):.2f}")
            print(f"  mean         : {ps.get('mean', 0):.2f}")
            print(f"  std          : {ps.get('std', 0):.2f}")
            print(f"  median       : {ps.get('median', 0):.2f}")

        # Windowing detail
        print(f"\n{sep}")
        print("  WINDOWING (VOI LUT)")
        print(sep)
        print(f"  Center       : {w.get('center', 'not set')}")
        print(f"  Width        : {w.get('width', 'not set')}")
        print(f"  Explanation  : {w.get('explanation', 'N/A')}")

        # Acquisition parameters
        acq = report.get("acquisition", {})
        non_none = {k: v for k, v in acq.items() if v is not None}
        if non_none:
            print(f"\n{sep}")
            print("  ACQUISITION PARAMETERS")
            print(sep)
            for k, v in non_none.items():
                print(f"  {k:35s}: {v}")

        # Full pipeline steps
        print(f"\n{sep}")
        print("  PREPROCESSING PIPELINE STEPS")
        print(sep)
        for step in tr.get("pipeline_steps", []):
            print(f"  {step}")
        print(f"\n  {tr.get('shortcut', '')}")

    print(f"\n{'═' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Single-file pipeline
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(
    path: str | Path,
    size: int = 512,
    output_root: str | Path = _OUTPUT_ROOT,
) -> np.ndarray:
    """Execute the full DICOM preprocessing pipeline on a single file.

    Steps performed:
        1. Read the DICOM file.
        2. Apply CAD-grade pixel preprocessing.
        3. Extract clinical metadata.
        4. Save the processed PNG image.
        5. Save the metadata JSON.
        6. Return the processed image array.

    Parameters
    ----------
    path : str | Path
        Path to the input ``.dicom`` file.
    size : int, optional
        Target spatial resolution (square). Default **512**.
    output_root : str | Path, optional
        Root directory for outputs. Default ``processed_data/``.

    Returns
    -------
    np.ndarray
        Processed image of shape ``(size, size)`` and dtype ``uint8``.

    Example
    -------
    >>> from bin.dicom_api import preprocess
    >>> img = preprocess("raw_data/0001.dicom")
    >>> img.shape
    (512, 512)
    """
    path = Path(path)
    output_root = Path(output_root)

    # Derive a clean stem for the output filenames.
    stem: str = path.stem  # e.g. "0001" from "0001.dicom"

    # 1. Read DICOM
    ds = read_dcm(path)

    # 2. Preprocess pixel data
    img: np.ndarray = process_px(ds, size=size)

    # 3. Extract metadata
    meta: dict = get_meta(ds)

    # 4. Save image
    image_path = output_root / "image" / f"{stem}.png"
    save_img(img, image_path)

    # 5. Save metadata
    meta_path = output_root / "metadata" / f"{stem}.json"
    save_meta(meta, meta_path)

    return img


# ═══════════════════════════════════════════════════════════════════════════
#  Batch pipeline
# ═══════════════════════════════════════════════════════════════════════════

def batch_run(
    input_dir: str | Path,
    output_root: str | Path = _OUTPUT_ROOT,
    size: int = 512,
    pattern: str = "*.dicom",
    recursive: bool = False,
    on_error: str = "skip",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Preprocess every DICOM in a directory.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing DICOM files.
    output_root : str | Path, optional
        Root directory for outputs.  Default ``processed_data/``.
    size : int, optional
        Target image size.  Default **512**.
    pattern : str, optional
        Glob pattern for DICOM files.  Default ``"*.dicom"``.
    recursive : bool, optional
        Search subdirectories.  Default ``False``.
    on_error : str, optional
        ``"skip"`` (default) to log and continue, ``"raise"`` to abort.
    progress_callback : callable | None
        Optional ``fn(current, total, filename)`` called after each file.

    Returns
    -------
    dict[str, Any]
        Summary dict with keys:
        ``total``, ``succeeded``, ``failed``, ``errors`` (list of dicts),
        ``elapsed_seconds``.

    Example
    -------
    >>> from bin.dicom_api import batch_preprocess
    >>> report = batch_preprocess("raw_data/", size=256)
    >>> print(report["succeeded"], "of", report["total"], "done")
    """
    start = time.perf_counter()
    files = list_dcm(input_dir, pattern=pattern, recursive=recursive)
    total = len(files)
    succeeded = 0
    errors: list[dict[str, str]] = []

    for idx, fpath in enumerate(files, 1):
        try:
            preprocess(fpath, size=size, output_root=output_root)
            succeeded += 1
        except Exception as exc:
            msg = f"{fpath.name}: {exc}"
            logger.warning("batch_run — %s", msg)
            errors.append({"file": str(fpath), "error": str(exc)})
            if on_error == "raise":
                raise
        if progress_callback is not None:
            progress_callback(idx, total, fpath.name)

    elapsed = time.perf_counter() - start
    return {
        "total": total,
        "succeeded": succeeded,
        "failed": total - succeeded,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Streaming / lazy pipeline
# ═══════════════════════════════════════════════════════════════════════════

def stream_run(
    input_dir: str | Path,
    size: int = 512,
    pattern: str = "*.dicom",
    recursive: bool = False,
    save: bool = True,
    output_root: str | Path = _OUTPUT_ROOT,
) -> Iterator[tuple[str, np.ndarray, dict[str, Any]]]:
    """Lazily yield ``(stem, image, metadata)`` for each DICOM in a directory.

    Unlike :func:`batch_preprocess`, this is a **generator** — it processes
    one file at a time, making it memory-friendly for large datasets and
    easy to integrate into training loops.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing DICOM files.
    size : int, optional
        Target image size.  Default **512**.
    pattern : str, optional
        Glob pattern.  Default ``"*.dicom"``.
    recursive : bool, optional
        Search subdirectories.  Default ``False``.
    save : bool, optional
        If ``True`` (default), save PNG + JSON alongside yielding.
    output_root : str | Path, optional
        Root directory for outputs.  Default ``processed_data/``.

    Yields
    ------
    tuple[str, np.ndarray, dict[str, Any]]
        ``(dicom_stem, processed_image, metadata_dict)`` for each file.

    Example
    -------
    >>> for stem, img, meta in stream_preprocess("raw_data/"):
    ...     print(stem, img.shape, meta.get("Modality"))
    """
    output_root = Path(output_root)
    files = list_dcm(input_dir, pattern=pattern, recursive=recursive)

    for fpath in files:
        try:
            ds = read_dcm(fpath)
            img = process_px(ds, size=size)
            meta = get_meta(ds)
            stem = fpath.stem

            if save:
                save_img(img, output_root / "image" / f"{stem}.png")
                save_meta(meta, output_root / "metadata" / f"{stem}.json")

            yield stem, img, meta
        except Exception as exc:
            logger.warning("stream_run — %s: %s", fpath.name, exc)


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════

def preview(
    path: str | Path,
    size: int = 256,
) -> np.ndarray:
    """Quick preview: preprocess without saving to disk.

    Useful during interactive exploration or in notebooks where you want
    the processed array but don't need persistence.

    Parameters
    ----------
    path : str | Path
        Path to the DICOM file.
    size : int, optional
        Preview size.  Default **256** (smaller for speed).

    Returns
    -------
    np.ndarray
        Processed image (``uint8``).
    """
    ds = read_dcm(path)
    return process_px(ds, size=size)


def pipe_summary(
    output_root: str | Path = _OUTPUT_ROOT,
) -> dict[str, Any]:
    """Return a summary of already-processed files in the output directory.

    Parameters
    ----------
    output_root : str | Path, optional
        Root directory for outputs.  Default ``processed_data/``.

    Returns
    -------
    dict[str, Any]
        ``image_count``, ``metadata_count``, ``image_stems``,
        ``metadata_stems``, ``orphan_images`` (images without JSON),
        ``orphan_metadata`` (JSON without images).
    """
    output_root = Path(output_root)
    img_dir = output_root / "image"
    meta_dir = output_root / "metadata"

    img_stems = {p.stem for p in img_dir.glob("*.png")} if img_dir.is_dir() else set()
    meta_stems = {p.stem for p in meta_dir.glob("*.json")} if meta_dir.is_dir() else set()

    return {
        "image_count": len(img_stems),
        "metadata_count": len(meta_stems),
        "image_stems": sorted(img_stems),
        "metadata_stems": sorted(meta_stems),
        "orphan_images": sorted(img_stems - meta_stems),
        "orphan_metadata": sorted(meta_stems - img_stems),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Recovery / repair
# ═══════════════════════════════════════════════════════════════════════════

def fix_orphans(
    output_root: str | Path,
    input_dir: str | Path,
    size: int = 512,
    pattern: str = "*.dicom",
    recursive: bool = False,
) -> dict[str, Any]:
    """Re-run the pipeline only for files whose output is incomplete.

    An *orphan* is any DICOM stem that is missing either its PNG image or
    its JSON metadata.  This is useful after a partial or interrupted batch
    run.

    Parameters
    ----------
    output_root : str | Path
        Root directory for outputs (same as used in the original run).
    input_dir : str | Path
        Source directory containing the original DICOM files.
    size : int, optional
        Target image size.  Default **512**.
    pattern : str, optional
        Glob pattern for DICOM discovery.  Default ``"*.dicom"``.
    recursive : bool, optional
        Search ``input_dir`` recursively.  Default ``False``.

    Returns
    -------
    dict[str, Any]
        ``reprocessed`` count, ``skipped`` count, ``errors`` list,
        ``elapsed_seconds``.
    """
    from bin.dicom_api.io import list_dcm as _list_dcm

    start = time.perf_counter()
    summary = pipe_summary(output_root)
    orphan_stems: set[str] = set(summary["orphan_images"]) | set(summary["orphan_metadata"])

    # Also include stems that appear in neither output folder.
    all_files = _list_dcm(input_dir, pattern=pattern, recursive=recursive)
    existing_stems = set(summary["image_stems"]) | set(summary["metadata_stems"])
    missing_stems = {f.stem for f in all_files} - existing_stems
    targets = {f for f in all_files if f.stem in (orphan_stems | missing_stems)}

    reprocessed = 0
    errors: list[dict[str, str]] = []
    for fpath in sorted(targets):
        try:
            preprocess(fpath, size=size, output_root=output_root)
            reprocessed += 1
        except Exception as exc:
            logger.warning("fix_orphans — %s: %s", fpath.name, exc)
            errors.append({"file": str(fpath), "error": str(exc)})

    elapsed = time.perf_counter() - start
    return {
        "reprocessed": reprocessed,
        "skipped": len(all_files) - reprocessed - len(errors),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 2),
    }


def retry_run(
    path: str | Path,
    size: int = 512,
    output_root: str | Path = _OUTPUT_ROOT,
    retries: int = 3,
    delay: float = 1.0,
) -> np.ndarray:
    """Run the pipeline with automatic retries on transient errors.

    Network file-systems and remote storage can yield intermittent I/O
    errors.  This wrapper retries the full pipeline up to *retries* times
    before re-raising.

    Parameters
    ----------
    path : str | Path
        Path to the DICOM file.
    size : int, optional
        Target image size.  Default **512**.
    output_root : str | Path, optional
        Output root directory.  Default ``processed_data/``.
    retries : int, optional
        Maximum number of additional attempts after the first failure.
        Default **3**.
    delay : float, optional
        Seconds to wait between attempts.  Default **1.0**.

    Returns
    -------
    np.ndarray
        Processed image on success.

    Raises
    ------
    Exception
        The last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            return preprocess(path, size=size, output_root=output_root)
        except Exception as exc:
            last_exc = exc
            if attempt <= retries:
                logger.warning(
                    "retry_run — attempt %d/%d failed for %s: %s",
                    attempt, retries + 1, Path(path).name, exc,
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════════

def summary_csv(
    output_root: str | Path = _OUTPUT_ROOT,
    csv_path: str | Path | None = None,
) -> Path:
    """Export the pipeline summary to a CSV file for audit or reporting.

    Each row records one stem with boolean flags for ``has_image`` and
    ``has_metadata``.

    Parameters
    ----------
    output_root : str | Path, optional
        Root directory for outputs.  Default ``processed_data/``.
    csv_path : str | Path | None, optional
        Destination CSV path.  Defaults to
        ``<output_root>/pipeline_summary.csv``.

    Returns
    -------
    Path
        Resolved path to the written CSV file.
    """
    output_root = Path(output_root)
    csv_path = Path(csv_path) if csv_path else output_root / "pipeline_summary.csv"

    summary = pipe_summary(output_root)
    img_stems = set(summary["image_stems"])
    meta_stems = set(summary["metadata_stems"])
    all_stems = sorted(img_stems | meta_stems)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stem", "has_image", "has_metadata"])
        writer.writeheader()
        for stem in all_stems:
            writer.writerow({
                "stem": stem,
                "has_image": stem in img_stems,
                "has_metadata": stem in meta_stems,
            })
    return csv_path.resolve()


def validate(
    output_root: str | Path = _OUTPUT_ROOT,
) -> dict[str, Any]:
    """Validate processed output files for basic integrity.

    For each image, checks that it is a non-empty, decodable PNG.
    For each metadata file, checks that it is valid JSON.

    Parameters
    ----------
    output_root : str | Path, optional
        Root directory for outputs.  Default ``processed_data/``.

    Returns
    -------
    dict[str, Any]
        ``valid_images``, ``corrupt_images``, ``valid_metadata``,
        ``corrupt_metadata``, ``corrupt_image_files``,
        ``corrupt_metadata_files``.
    """
    import json as _json
    import cv2 as _cv2

    output_root = Path(output_root)
    img_dir = output_root / "image"
    meta_dir = output_root / "metadata"

    valid_img = corrupt_img = 0
    corrupt_img_files: list[str] = []
    valid_meta = corrupt_meta = 0
    corrupt_meta_files: list[str] = []

    for png in sorted(img_dir.glob("*.png")) if img_dir.is_dir() else []:
        arr = _cv2.imread(str(png), _cv2.IMREAD_GRAYSCALE)
        if arr is None or arr.size == 0:
            corrupt_img += 1
            corrupt_img_files.append(png.name)
        else:
            valid_img += 1

    for jf in sorted(meta_dir.glob("*.json")) if meta_dir.is_dir() else []:
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                _json.load(fh)
            valid_meta += 1
        except Exception:
            corrupt_meta += 1
            corrupt_meta_files.append(jf.name)

    return {
        "valid_images": valid_img,
        "corrupt_images": corrupt_img,
        "corrupt_image_files": corrupt_img_files,
        "valid_metadata": valid_meta,
        "corrupt_metadata": corrupt_meta,
        "corrupt_metadata_files": corrupt_meta_files,
    }


def clean_out(
    output_root: str | Path = _OUTPUT_ROOT,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Delete all processed images and metadata in the output directory.

    Parameters
    ----------
    output_root : str | Path, optional
        Root directory for outputs.  Default ``processed_data/``.
    dry_run : bool, optional
        If ``True`` (default), only *list* what would be deleted without
        actually removing anything.  Set to ``False`` to perform deletion.

    Returns
    -------
    dict[str, Any]
        ``deleted_images`` count, ``deleted_metadata`` count,
        ``dry_run`` flag.
    """
    output_root = Path(output_root)
    img_dir = output_root / "image"
    meta_dir = output_root / "metadata"

    pngs = sorted(img_dir.glob("*.png")) if img_dir.is_dir() else []
    jsons = sorted(meta_dir.glob("*.json")) if meta_dir.is_dir() else []

    if not dry_run:
        for f in pngs:
            f.unlink(missing_ok=True)
        for f in jsons:
            f.unlink(missing_ok=True)
        logger.info("clean_outputs — deleted %d images, %d metadata files", len(pngs), len(jsons))

    return {
        "deleted_images": len(pngs),
        "deleted_metadata": len(jsons),
        "dry_run": dry_run,
    }
