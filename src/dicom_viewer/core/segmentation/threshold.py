"""Intensity-threshold segmentation."""
from __future__ import annotations

import numpy as np

from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.volume import Volume


def threshold(volume: Volume, low: float, high: float) -> Segmentation:
    """Select voxels with low <= value <= high."""
    if high < low:
        raise ValueError(f"threshold high ({high}) < low ({low})")
    mask = (volume.array >= low) & (volume.array <= high)
    return Segmentation(
        mask=np.ascontiguousarray(mask),
        method="threshold",
        params={"low": low, "high": high},
    )
