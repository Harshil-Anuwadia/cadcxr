# CAD-CXR

Computer-Aided Detection for Chest X-Rays — a reusable DICOM preprocessing API for medical imaging pipelines.

## Features

- **DICOM I/O** — read, write, validate, copy, anonymize
- **Preprocessing** — rescale, invert, windowing, CLAHE, normalize, resize, pad, crop
- **Metadata** — extract, compare, merge, export CSV/JSON
- **Diagnostics** — `explain()` full diagnostic report, `fix()` auto-fix all issues
- **Training prep** — split, augment, standardize, tensor conversion

## Quick Start

```python
from bin.dicom_api import *

# Diagnose a DICOM
explain("raw_data/sample.dicom")

# Auto-fix all issues
result = fix("raw_data/sample.dicom")
img = result["image"]  # clean uint8 numpy array (512×512)

# Verify fixes are applied
explain("raw_data/sample.dicom", image=result["image"])

# Run full preprocessing pipeline
img = preprocess("raw_data/sample.dicom")
```

## Project Structure

```
bin/
  dicom_api/
    __init__.py     # all exports
    io.py           # DICOM I/O
    preprocess.py   # pixel preprocessing
    metadata.py     # metadata utilities
    core.py         # pipeline, explain(), fix()
    training.py     # training data preparation
Untitled.ipynb      # exploration notebook
```

## Available Fix Keys

Used with `fix(path, only=[...])` or `fix(path, skip=[...])`:

| Key | What it fixes |
|---|---|
| `rescale` | RescaleSlope / RescaleIntercept |
| `invert` | MONOCHROME1 → MONOCHROME2 |
| `pad_square` | Non-square → square |
| `resize` | Downsample large images |
| `pct_clip` | Remove outlier pixel values |
| `normalize` | Scale to \[0, 1\] |
| `grayscale` | Multi-channel → single channel |
| `to_uint8` | Convert to uint8 |

## Dependencies

- `pydicom`
- `numpy`
- `opencv-python`
