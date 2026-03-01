"""
dicom_api — Reusable DICOM preprocessing API for chest X-ray CAD systems.

Quick-start
-----------
::

    from bin.dicom_api import preprocess
    img = preprocess("raw_data/sample.dicom")

All functions have short, meaningful names::

    from bin.dicom_api import (
        # I/O
        read_dcm, read_pixels, read_header, list_dcm, is_valid,
        has_pixels, dcm_shape, dcm_modality, dcm_spacing,
        verify_dcm, read_batch, copy_dcm, anonymize, write_dcm,
        # Pipeline & Diagnostics
        explain, fix, preprocess, retry_run, batch_run, stream_run, preview,
        pipe_summary, fix_orphans, summary_csv, validate, clean_out,
        # Preprocessing
        process_px, rescale, invert, pct_clip, windowing,
        normalize, resize, to_uint8, to_float32,
        clahe, gamma, sharpen, gauss_blur, med_blur, bilat_blur,
        pad_square, autocrop, center_crop,
        # Metadata
        get_meta, full_header, acq_params, pixel_stats, window_info,
        compare_meta, merge_meta, pick_meta, flatten_meta,
        batch_meta, summarize, save_meta, load_meta, meta_csv,
        save_img, load_img,
        # Training prep
        split_data, load_dataset, load_paired,
        augment, rand_flip, rand_rotate, rand_brightness,
        rand_contrast, rand_crop, rand_zoom, rand_noise,
        standardize, img_stats, to_channels, to_rgb,
        to_tensor, to_batch, from_batch,
    )
"""

# ── Pipeline ────────────────────────────────────────────────────────────
from bin.dicom_api.core import (  # noqa: F401
    explain,
    fix,
    preprocess,
    retry_run,
    batch_run,
    stream_run,
    preview,
    pipe_summary,
    fix_orphans,
    summary_csv,
    validate,
    clean_out,
)

# ── I/O ─────────────────────────────────────────────────────────────────
from bin.dicom_api.io import (  # noqa: F401
    read_dcm,
    read_pixels,
    read_header,
    list_dcm,
    is_valid,
    has_pixels,
    dcm_shape,
    dcm_modality,
    dcm_spacing,
    verify_dcm,
    read_batch,
    copy_dcm,
    anonymize,
    write_dcm,
)

# ── Preprocessing ───────────────────────────────────────────────────────
from bin.dicom_api.preprocess import (  # noqa: F401
    process_px,
    # Core DICOM-aware transforms
    rescale,
    invert,
    pct_clip,
    windowing,
    normalize,
    to_float32,
    to_uint8,
    # Enhancement
    clahe,
    gamma,
    sharpen,
    # Noise reduction
    gauss_blur,
    med_blur,
    bilat_blur,
    # Geometry
    resize,
    pad_square,
    autocrop,
    center_crop,
)

# ── Metadata ────────────────────────────────────────────────────────────
from bin.dicom_api.metadata import (  # noqa: F401
    get_meta,
    full_header,
    acq_params,
    pixel_stats,
    window_info,
    compare_meta,
    merge_meta,
    pick_meta,
    flatten_meta,
    batch_meta,
    summarize,
    save_meta,
    load_meta,
    meta_csv,
    save_img,
    load_img,
)

# ── Training / pre-model prep ──────────────────────────────────────────
from bin.dicom_api.training import (  # noqa: F401
    split_data,
    load_dataset,
    load_paired,
    augment,
    rand_flip,
    rand_rotate,
    rand_brightness,
    rand_contrast,
    rand_crop,
    rand_zoom,
    rand_noise,
    standardize,
    img_stats,
    to_channels,
    to_rgb,
    to_tensor,
    to_batch,
    from_batch,
)

__all__ = [
    # Pipeline & Diagnostics
    "explain", "fix",
    "preprocess", "retry_run", "batch_run", "stream_run", "preview",
    "pipe_summary", "fix_orphans", "summary_csv", "validate", "clean_out",
    # I/O
    "read_dcm", "read_pixels", "read_header", "list_dcm", "is_valid",
    "has_pixels", "dcm_shape", "dcm_modality", "dcm_spacing",
    "verify_dcm", "read_batch", "copy_dcm", "anonymize", "write_dcm",
    # Preprocessing
    "process_px", "rescale", "invert", "pct_clip", "windowing",
    "normalize", "to_float32", "to_uint8",
    "clahe", "gamma", "sharpen", "gauss_blur", "med_blur", "bilat_blur",
    "resize", "pad_square", "autocrop", "center_crop",
    # Metadata
    "get_meta", "full_header", "acq_params", "pixel_stats", "window_info",
    "compare_meta", "merge_meta", "pick_meta", "flatten_meta",
    "batch_meta", "summarize", "save_meta", "load_meta", "meta_csv",
    "save_img", "load_img",
    # Training / pre-model prep
    "split_data", "load_dataset", "load_paired",
    "augment", "rand_flip", "rand_rotate",
    "rand_brightness", "rand_contrast", "rand_crop",
    "rand_zoom", "rand_noise",
    "standardize", "img_stats",
    "to_channels", "to_rgb", "to_tensor", "to_batch", "from_batch",
]
