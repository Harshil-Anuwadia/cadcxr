"""
preprocess — CAD-grade pixel preprocessing for chest X-ray DICOMs.

Every function is public and can be imported individually::

    from bin.dicom_api.preprocess import apply_clahe, normalize, resize

The convenience function :func:`preprocess_pixels` chains the standard
pipeline steps, but each step is also available as a standalone,
composable building-block so you can build custom pipelines.

Pipeline steps (default ``preprocess_pixels``):
    1. Cast ``pixel_array`` → ``float32``
    2. Apply RescaleSlope / RescaleIntercept
    3. Invert MONOCHROME1 images
    4. Percentile windowing (1st – 99th)
    5. Normalize to [0, 1]
    6. Resize to ``size × size``
    7. Convert to ``uint8`` for saving
"""

from __future__ import annotations

import numpy as np
import cv2
from pydicom.dataset import Dataset

__all__ = [
    "process_px",
    # Core transforms
    "rescale",
    "invert",
    "pct_clip",
    "windowing",
    "normalize",
    "resize",
    "to_uint8",
    "to_float32",
    # Enhancement filters
    "clahe",
    "gamma",
    "sharpen",
    # Noise reduction
    "gauss_blur",
    "med_blur",
    "bilat_blur",
    # Geometry helpers
    "pad_square",
    "autocrop",
    "center_crop",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def process_px(ds: Dataset, size: int = 512) -> np.ndarray:
    """Apply full CAD-grade preprocessing to DICOM pixel data.

    Parameters
    ----------
    ds : pydicom.Dataset
        Parsed DICOM dataset (must contain ``pixel_array``).
    size : int, optional
        Target spatial resolution (square). Default **512**.

    Returns
    -------
    np.ndarray
        Preprocessed image of shape ``(size, size)`` and dtype ``uint8``
        with values in ``[0, 255]``.
    """
    img = ds.pixel_array.astype(np.float32)

    img = rescale(img, ds)
    img = invert(img, ds)
    img = pct_clip(img)
    img = normalize(img)
    img = resize(img, size)
    img = to_uint8(img)

    return img


# ═══════════════════════════════════════════════════════════════════════════
#  Core transforms  (DICOM-aware)
# ═══════════════════════════════════════════════════════════════════════════

def rescale(img: np.ndarray, ds: Dataset) -> np.ndarray:
    """Apply ``RescaleSlope`` and ``RescaleIntercept`` from DICOM header.

    Parameters
    ----------
    img : np.ndarray
        Pixel array (``float32``).
    ds : pydicom.Dataset
        Parsed DICOM dataset.

    Returns
    -------
    np.ndarray
        Rescaled pixel array.
    """
    slope: float = float(getattr(ds, "RescaleSlope", 1.0))
    intercept: float = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        img = img * slope + intercept
    return img


def invert(img: np.ndarray, ds: Dataset) -> np.ndarray:
    """Invert pixel values when ``PhotometricInterpretation`` is MONOCHROME1.

    MONOCHROME1 encodes air as *bright* and bone as *dark*, which is the
    opposite of the convention used by most CAD models.

    Parameters
    ----------
    img : np.ndarray
        Pixel array.
    ds : pydicom.Dataset
        Parsed DICOM dataset.

    Returns
    -------
    np.ndarray
        Inverted array (if MONOCHROME1), otherwise unchanged.
    """
    photometric: str = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        img = img.max() - img
    return img


def pct_clip(
    img: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    """Clip pixel values to the ``[low_pct, high_pct]`` percentile range.

    Removes extreme outlier intensities that would otherwise compress the
    useful dynamic range after normalization.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    low_pct : float
        Lower percentile bound (default **1.0**).
    high_pct : float
        Upper percentile bound (default **99.0**).

    Returns
    -------
    np.ndarray
        Clipped image.
    """
    p_low, p_high = np.percentile(img, [low_pct, high_pct])
    return np.clip(img, p_low, p_high)


def windowing(
    img: np.ndarray,
    center: float,
    width: float,
) -> np.ndarray:
    """Apply standard radiology window / level transform.

    Maps the intensity range ``[center - width/2, center + width/2]``
    to ``[0, 1]``.  Values outside the window are clamped.

    Parameters
    ----------
    img : np.ndarray
        Input image (``float32``, rescaled Hounsfield / raw values).
    center : float
        Window center (level).
    width : float
        Window width.

    Returns
    -------
    np.ndarray
        Windowed image in ``[0, 1]``.
    """
    lower = center - width / 2.0
    upper = center + width / 2.0
    img = np.clip(img, lower, upper)
    return (img - lower) / (upper - lower + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════
#  Intensity transforms  (array-only, no Dataset needed)
# ═══════════════════════════════════════════════════════════════════════════

def normalize(img: np.ndarray) -> np.ndarray:
    """Min-max normalize pixel values to ``[0, 1]``.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Normalized image (``float32``).
    """
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        return (img - img_min) / (img_max - img_min)
    return np.zeros_like(img)


def to_float32(img: np.ndarray) -> np.ndarray:
    """Cast any image array to ``float32``.

    Parameters
    ----------
    img : np.ndarray
        Input image of any numeric dtype.

    Returns
    -------
    np.ndarray
        Image as ``float32``.
    """
    return img.astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a ``[0, 1]`` float image to ``uint8`` in ``[0, 255]``.

    Parameters
    ----------
    img : np.ndarray
        Image with values in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Image as ``uint8``.
    """
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply power-law (gamma) correction.

    Values < 1.0 brighten the image; values > 1.0 darken it.

    Parameters
    ----------
    img : np.ndarray
        Image in ``[0, 1]`` (``float32``).
    gamma : float
        Gamma exponent.  Default **1.0** (no change).

    Returns
    -------
    np.ndarray
        Gamma-corrected image in ``[0, 1]``.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    return np.power(np.clip(img, 0.0, 1.0), gamma).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Enhancement filters
# ═══════════════════════════════════════════════════════════════════════════

def clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE).

    Commonly used in medical imaging to improve local contrast without
    amplifying noise.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image (``uint8``).
    clip_limit : float
        Threshold for contrast limiting.  Default **2.0**.
    tile_grid_size : tuple[int, int]
        Size of the grid for histogram equalization.  Default **(8, 8)**.

    Returns
    -------
    np.ndarray
        Enhanced image (``uint8``).
    """
    if img.dtype != np.uint8:
        raise TypeError("clahe expects a uint8 image. "
                         "Call to_uint8() first if your image is float.")
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(img)


def sharpen(
    img: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    amount: float = 1.0,
) -> np.ndarray:
    """Sharpen an image using unsharp masking.

    ``sharpened = original + amount * (original − blurred)``

    Parameters
    ----------
    img : np.ndarray
        Input image (``float32`` or ``uint8``).
    kernel_size : int
        Gaussian kernel size (must be odd).  Default **5**.
    sigma : float
        Gaussian sigma.  Default **1.0**.
    amount : float
        Sharpening strength.  Default **1.0**.

    Returns
    -------
    np.ndarray
        Sharpened image (same dtype as input).
    """
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(
        img,
        1.0 + amount,
        blurred,
        -amount,
        0,
    )
    if img.dtype == np.uint8:
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Noise reduction filters
# ═══════════════════════════════════════════════════════════════════════════

def gauss_blur(
    img: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> np.ndarray:
    """Apply Gaussian smoothing.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    kernel_size : int
        Kernel size (must be odd).  Default **5**.
    sigma : float
        Gaussian sigma.  **0.0** lets OpenCV compute from kernel size.

    Returns
    -------
    np.ndarray
        Smoothed image (same dtype).
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def med_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filtering (salt-and-pepper noise removal).

    Parameters
    ----------
    img : np.ndarray
        Input image (``uint8``).
    kernel_size : int
        Kernel size (must be odd).  Default **5**.

    Returns
    -------
    np.ndarray
        Filtered image.
    """
    return cv2.medianBlur(img, kernel_size)


def bilat_blur(
    img: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply bilateral filtering (edge-preserving noise reduction).

    Parameters
    ----------
    img : np.ndarray
        Input image (``uint8``).
    d : int
        Diameter of each pixel neighbourhood.  Default **9**.
    sigma_color : float
        Filter sigma in the color space.  Default **75.0**.
    sigma_space : float
        Filter sigma in the coordinate space.  Default **75.0**.

    Returns
    -------
    np.ndarray
        Filtered image.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to ``(size, size)`` with automatic interpolation selection.

    Uses ``INTER_AREA`` when down-sampling and ``INTER_LINEAR`` when
    up-sampling, following OpenCV best-practice guidelines.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    size : int
        Target spatial resolution (square).

    Returns
    -------
    np.ndarray
        Resized image of shape ``(size, size)``.
    """
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def pad_square(img: np.ndarray, fill: int = 0) -> np.ndarray:
    """Pad a rectangular image to a square with a constant fill value.

    Padding is added symmetrically on the shorter axis.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape ``(H, W)`` or ``(H, W, C)``.
    fill : int
        Padding pixel value.  Default **0** (black).

    Returns
    -------
    np.ndarray
        Square image of shape ``(max(H,W), max(H,W), ...)``.
    """
    h, w = img.shape[:2]
    if h == w:
        return img

    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left

    return cv2.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=fill,
    )


def autocrop(
    img: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """Crop dark / black borders from a normalized image.

    Finds the bounding box of all pixels above *threshold* intensity and
    crops to that region.  Useful for removing letter-boxing or collimation
    borders in chest X-rays.

    Parameters
    ----------
    img : np.ndarray
        Input image (``float32`` in ``[0, 1]`` or ``uint8``).
    threshold : float
        Intensity threshold used to distinguish content from border.
        Default **0.05** (for ``[0, 1]``-range images).

    Returns
    -------
    np.ndarray
        Cropped image.  Returns the original if no content is found.
    """
    if img.dtype == np.uint8:
        thresh_abs = int(threshold * 255)
    else:
        thresh_abs = threshold

    mask = img > thresh_abs
    if mask.ndim == 3:
        mask = mask.any(axis=2)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return img

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return img[rmin : rmax + 1, cmin : cmax + 1]


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Extract a square center crop of ``size × size`` pixels.

    If the image is smaller than *size* in either dimension, it is returned
    unchanged.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    size : int
        Desired crop size (square).

    Returns
    -------
    np.ndarray
        Center-cropped image.
    """
    h, w = img.shape[:2]
    if h < size or w < size:
        return img

    y_start = (h - size) // 2
    x_start = (w - size) // 2
    return img[y_start : y_start + size, x_start : x_start + size]
