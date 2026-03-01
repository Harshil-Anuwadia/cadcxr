"""
training — Pre-training data preparation utilities for chest X-ray CAD models.

Data augmentation, train/val/test splitting, dataset loading,
standardization, and format conversion helpers::

    from bin.dicom_api.training import (
        split_data, load_dataset, augment, rand_flip, rand_rotate,
        rand_brightness, rand_contrast, rand_crop, standardize,
        to_channels, to_rgb, to_tensor, to_batch, img_stats,
    )
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    # Dataset splitting
    "split_data",
    # Dataset loading
    "load_dataset",
    "load_paired",
    # Augmentation
    "augment",
    "rand_flip",
    "rand_rotate",
    "rand_brightness",
    "rand_contrast",
    "rand_crop",
    "rand_zoom",
    "rand_noise",
    # Standardization & format
    "standardize",
    "img_stats",
    "to_channels",
    "to_rgb",
    "to_tensor",
    "to_batch",
    "from_batch",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset splitting
# ═══════════════════════════════════════════════════════════════════════════

def split_data(
    paths: list[str | Path],
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
    shuffle: bool = True,
) -> dict[str, list[Path]]:
    """Split file paths into train / validation / test sets.

    Parameters
    ----------
    paths : list[str | Path]
        List of file paths (DICOM, PNG, etc.) to split.
    train : float
        Fraction for the training set.  Default **0.7**.
    val : float
        Fraction for the validation set.  Default **0.15**.
    test : float
        Fraction for the test set.  Default **0.15**.
    seed : int
        Random seed for reproducibility.  Default **42**.
    shuffle : bool
        Shuffle before splitting.  Default ``True``.

    Returns
    -------
    dict[str, list[Path]]
        ``{"train": [...], "val": [...], "test": [...]}``.

    Raises
    ------
    ValueError
        If fractions don't sum to ~1.0 or inputs are empty.

    Example
    -------
    >>> from bin.dicom_api import split_data, list_dcm
    >>> files = list_dcm("processed_data/image/")
    >>> splits = split_data(files)
    >>> len(splits["train"]), len(splits["val"]), len(splits["test"])
    """
    if abs(train + val + test - 1.0) > 0.01:
        raise ValueError(
            f"train + val + test must equal 1.0, got {train + val + test:.3f}"
        )
    if not paths:
        raise ValueError("paths list is empty")

    file_paths: list[Path] = [Path(p) for p in paths]
    if shuffle:
        rng = random.Random(seed)
        file_paths = file_paths.copy()
        rng.shuffle(file_paths)

    n = len(file_paths)
    n_train = int(n * train)
    n_val = int(n * val)

    return {
        "train": file_paths[:n_train],
        "val": file_paths[n_train : n_train + n_val],
        "test": file_paths[n_train + n_val :],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset loading
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(
    image_dir: str | Path,
    size: int | None = None,
    pattern: str = "*.png",
    grayscale: bool = True,
    max_count: int | None = None,
) -> np.ndarray:
    """Load all images from a directory into a single NumPy array.

    Parameters
    ----------
    image_dir : str | Path
        Directory containing image files.
    size : int | None
        If given, resize every image to ``(size, size)``.
    pattern : str
        Glob pattern for image files.  Default ``"*.png"``.
    grayscale : bool
        Load as single-channel grayscale.  Default ``True``.
    max_count : int | None
        Limit number of images loaded (useful for quick tests).

    Returns
    -------
    np.ndarray
        Array of shape ``(N, H, W)`` for grayscale or ``(N, H, W, 3)``
        for colour, dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If ``image_dir`` doesn't exist or contains no matching files.

    Example
    -------
    >>> X = load_dataset("processed_data/image/", size=224)
    >>> X.shape  # (N, 224, 224)
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    files = sorted(image_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files in {image_dir}")

    if max_count is not None:
        files = files[:max_count]

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    images: list[np.ndarray] = []

    for f in files:
        img = cv2.imread(str(f), flag)
        if img is None:
            logger.warning("load_dataset — skipping unreadable: %s", f.name)
            continue
        if size is not None:
            img = cv2.resize(img, (size, size))
        images.append(img)

    return np.stack(images)


def load_paired(
    image_dir: str | Path,
    meta_dir: str | Path,
    size: int | None = None,
    pattern: str = "*.png",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load images and their matching JSON metadata side by side.

    Matches files by stem name (e.g. ``abc.png`` ↔ ``abc.json``).

    Parameters
    ----------
    image_dir : str | Path
        Directory containing image files.
    meta_dir : str | Path
        Directory containing ``.json`` metadata files.
    size : int | None
        Resize images to ``(size, size)`` if given.
    pattern : str
        Glob pattern for images.  Default ``"*.png"``.

    Returns
    -------
    tuple[np.ndarray, list[dict]]
        ``(images_array, metadata_list)`` — paired by stem name.
    """
    import json

    image_dir, meta_dir = Path(image_dir), Path(meta_dir)
    img_files = sorted(image_dir.glob(pattern))

    images: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []

    for f in img_files:
        json_path = meta_dir / f"{f.stem}.json"
        if not json_path.is_file():
            logger.warning("load_paired — no metadata for %s, skipping", f.name)
            continue

        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if size is not None:
            img = cv2.resize(img, (size, size))

        with open(json_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        images.append(img)
        metas.append(meta)

    return np.stack(images), metas


# ═══════════════════════════════════════════════════════════════════════════
#  Data augmentation
# ═══════════════════════════════════════════════════════════════════════════

def augment(
    img: np.ndarray,
    flip: bool = True,
    rotate: float = 10.0,
    brightness: float = 0.1,
    contrast: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a random combination of augmentations in one call.

    Convenience wrapper that chains individual augmentation functions.

    Parameters
    ----------
    img : np.ndarray
        Input image (``uint8`` or ``float32``).
    flip : bool
        Enable random horizontal flip.  Default ``True``.
    rotate : float
        Max rotation in degrees (±).  Default **10.0**.  Set 0 to disable.
    brightness : float
        Max brightness shift (±).  Default **0.1**.  Set 0 to disable.
    contrast : float
        Max contrast factor deviation (±).  Default **0.1**.  Set 0 to disable.
    seed : int | None
        Random seed for reproducibility.  ``None`` = non-deterministic.

    Returns
    -------
    np.ndarray
        Augmented image (same dtype as input).
    """
    rng = np.random.RandomState(seed)

    if flip and rng.rand() > 0.5:
        img = rand_flip(img, seed=int(rng.randint(1 << 31)))
    if rotate > 0:
        img = rand_rotate(img, max_deg=rotate, seed=int(rng.randint(1 << 31)))
    if brightness > 0:
        img = rand_brightness(img, factor=brightness, seed=int(rng.randint(1 << 31)))
    if contrast > 0:
        img = rand_contrast(img, factor=contrast, seed=int(rng.randint(1 << 31)))

    return img


def rand_flip(img: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Randomly flip image horizontally (50 % chance).

    Parameters
    ----------
    img : np.ndarray
        Input image.
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Flipped or original image.
    """
    rng = np.random.RandomState(seed)
    if rng.rand() > 0.5:
        return np.fliplr(img).copy()
    return img


def rand_rotate(
    img: np.ndarray,
    max_deg: float = 15.0,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly rotate image within ``[-max_deg, +max_deg]``.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    max_deg : float
        Maximum rotation angle in degrees.  Default **15.0**.
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Rotated image (same shape, black fill for borders).
    """
    rng = np.random.RandomState(seed)
    angle = rng.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=0)


def rand_brightness(
    img: np.ndarray,
    factor: float = 0.15,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly shift image brightness.

    Parameters
    ----------
    img : np.ndarray
        Input image (``uint8`` or ``float32``).
    factor : float
        Maximum brightness shift (± fraction of range).  Default **0.15**.
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Brightness-adjusted image (same dtype).
    """
    rng = np.random.RandomState(seed)
    shift = rng.uniform(-factor, factor)

    if img.dtype == np.uint8:
        delta = int(shift * 255)
        return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    return np.clip(img + shift, 0.0, 1.0).astype(np.float32)


def rand_contrast(
    img: np.ndarray,
    factor: float = 0.15,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly adjust image contrast.

    Multiplies pixel values by a random factor in ``[1 - factor, 1 + factor]``.

    Parameters
    ----------
    img : np.ndarray
        Input image (``uint8`` or ``float32``).
    factor : float
        Maximum contrast deviation (±).  Default **0.15**.
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Contrast-adjusted image (same dtype).
    """
    rng = np.random.RandomState(seed)
    alpha = rng.uniform(1.0 - factor, 1.0 + factor)

    if img.dtype == np.uint8:
        mean = img.mean()
        result = (alpha * (img.astype(np.float32) - mean) + mean)
        return np.clip(result, 0, 255).astype(np.uint8)

    mean = img.mean()
    result = alpha * (img - mean) + mean
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def rand_crop(
    img: np.ndarray,
    crop_size: int,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly crop a ``crop_size × crop_size`` patch from the image.

    Parameters
    ----------
    img : np.ndarray
        Input image (must be larger than ``crop_size`` in both dimensions).
    crop_size : int
        Output square size.
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Cropped patch.
    """
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Image ({h}×{w}) is smaller than crop_size ({crop_size})"
        )
    y = rng.randint(0, h - crop_size + 1)
    x = rng.randint(0, w - crop_size + 1)
    return img[y : y + crop_size, x : x + crop_size]


def rand_zoom(
    img: np.ndarray,
    zoom_range: tuple[float, float] = (0.9, 1.1),
    seed: int | None = None,
) -> np.ndarray:
    """Randomly zoom into the image and resize back to original dimensions.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    zoom_range : tuple[float, float]
        Min and max zoom factor.  Default **(0.9, 1.1)**.
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Zoomed image (same shape as input).
    """
    rng = np.random.RandomState(seed)
    factor = rng.uniform(*zoom_range)
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)

    resized = cv2.resize(img, (new_w, new_h))

    # Crop or pad back to original size
    if factor >= 1.0:
        # Crop centre
        y_start = (new_h - h) // 2
        x_start = (new_w - w) // 2
        return resized[y_start : y_start + h, x_start : x_start + w]
    else:
        # Pad with black
        out = np.zeros_like(img)
        y_start = (h - new_h) // 2
        x_start = (w - new_w) // 2
        out[y_start : y_start + new_h, x_start : x_start + new_w] = resized
        return out


def rand_noise(
    img: np.ndarray,
    std: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """Add random Gaussian noise to an image.

    Parameters
    ----------
    img : np.ndarray
        Input image (``float32`` in ``[0, 1]`` or ``uint8``).
    std : float
        Noise standard deviation.  Default **0.02** (for ``[0,1]`` range).
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Noisy image (same dtype as input).
    """
    rng = np.random.RandomState(seed)

    if img.dtype == np.uint8:
        noise = rng.normal(0, std * 255, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    noise = rng.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Standardization & format conversion
# ═══════════════════════════════════════════════════════════════════════════

def standardize(
    img: np.ndarray,
    mean: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    """Zero-mean / unit-variance standardization.

    If *mean* and *std* are ``None``, they are computed from the image
    itself (per-image standardization).  Pass dataset-level values for
    consistent standardization across all images.

    Parameters
    ----------
    img : np.ndarray
        Input image (any dtype — will be cast to ``float32``).
    mean : float | None
        Global mean.  ``None`` = compute from image.
    std : float | None
        Global std.  ``None`` = compute from image.

    Returns
    -------
    np.ndarray
        Standardized ``float32`` image.

    Example
    -------
    >>> stats = img_stats(X_train)
    >>> X_train = standardize(X_train, mean=stats["mean"], std=stats["std"])
    """
    img = img.astype(np.float32)
    m = mean if mean is not None else img.mean()
    s = std if std is not None else img.std()
    if s < 1e-8:
        return img - m
    return (img - m) / s


def img_stats(images: np.ndarray) -> dict[str, float]:
    """Compute dataset-level mean and std for standardization.

    Parameters
    ----------
    images : np.ndarray
        Array of images — shape ``(N, H, W)`` or ``(N, H, W, C)``.

    Returns
    -------
    dict[str, float]
        ``{"mean": float, "std": float, "min": float, "max": float}``.

    Example
    -------
    >>> X = load_dataset("processed_data/image/", size=224)
    >>> stats = img_stats(X)
    >>> print(stats)
    {'mean': 127.3, 'std': 58.1, 'min': 0.0, 'max': 255.0}
    """
    arr = images.astype(np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def to_channels(img: np.ndarray) -> np.ndarray:
    """Add a channel dimension: ``(H, W)`` → ``(H, W, 1)``.

    If already 3-D, returns unchanged.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image of shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        Image of shape ``(H, W, 1)``.
    """
    if img.ndim == 2:
        return img[:, :, np.newaxis]
    return img


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert grayscale to 3-channel RGB: ``(H, W)`` → ``(H, W, 3)``.

    If already 3-channel, returns unchanged.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image.

    Returns
    -------
    np.ndarray
        RGB image of shape ``(H, W, 3)``.
    """
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    return img


def to_tensor(img: np.ndarray) -> np.ndarray:
    """Convert image to channel-first ``float32`` tensor layout.

    ``(H, W)`` → ``(1, H, W)``
    ``(H, W, C)`` → ``(C, H, W)``

    Also normalizes ``uint8`` [0, 255] → ``float32`` [0, 1].

    Parameters
    ----------
    img : np.ndarray
        Single image.

    Returns
    -------
    np.ndarray
        Channel-first ``float32`` tensor.

    Example
    -------
    >>> t = to_tensor(img)        # (1, 224, 224) float32
    >>> t = to_tensor(to_rgb(img))  # (3, 224, 224) float32
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    if img.ndim == 2:
        return img[np.newaxis, :, :]        # (1, H, W)
    return np.transpose(img, (2, 0, 1))     # (C, H, W)


def to_batch(images: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Stack images into a batch tensor: ``(N, C, H, W)`` float32.

    Applies :func:`to_tensor` to each image individually, then stacks.

    Parameters
    ----------
    images : np.ndarray | list[np.ndarray]
        Array of shape ``(N, H, W)`` / ``(N, H, W, C)`` or a list of
        individual images.

    Returns
    -------
    np.ndarray
        Batch tensor of shape ``(N, C, H, W)``, dtype ``float32``.

    Example
    -------
    >>> X = load_dataset("processed_data/image/", size=224)
    >>> batch = to_batch(X)
    >>> batch.shape  # (N, 1, 224, 224)
    """
    if isinstance(images, np.ndarray) and images.ndim >= 3:
        return np.stack([to_tensor(images[i]) for i in range(len(images))])
    return np.stack([to_tensor(img) for img in images])


def from_batch(batch: np.ndarray) -> list[np.ndarray]:
    """Unpack a batch tensor back to a list of HWC ``uint8`` images.

    Reverses :func:`to_batch` — converts ``(N, C, H, W)`` float32 →
    list of ``(H, W)`` or ``(H, W, C)`` uint8 images.

    Parameters
    ----------
    batch : np.ndarray
        Batch tensor of shape ``(N, C, H, W)``.

    Returns
    -------
    list[np.ndarray]
        List of individual images.
    """
    images: list[np.ndarray] = []
    for i in range(batch.shape[0]):
        t = batch[i]  # (C, H, W)
        if t.shape[0] == 1:
            img = t[0]  # (H, W)
        else:
            img = np.transpose(t, (1, 2, 0))  # (H, W, C)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        images.append(img)
    return images
