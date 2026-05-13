"""Region-growing segmentation via SimpleITK.ConnectedThreshold."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.volume import Volume


def region_grow(
    volume: Volume, seed: tuple[int, int, int], tolerance: float
) -> Segmentation:
    """Flood-fill from `seed` (in z,y,x voxel coords) within ±tolerance of seed value."""
    z, y, x = seed
    sz, sy, sx = volume.shape
    if not (0 <= z < sz and 0 <= y < sy and 0 <= x < sx):
        raise ValueError(f"seed {seed} outside volume shape {volume.shape}")

    # SimpleITK uses (x, y, z) index order.
    image = sitk.GetImageFromArray(volume.array)
    seed_value = float(volume.array[z, y, x])

    grown = sitk.ConnectedThreshold(
        image,
        seedList=[(int(x), int(y), int(z))],
        lower=float(seed_value - tolerance),
        upper=float(seed_value + tolerance),
        replaceValue=1,
    )
    mask = sitk.GetArrayFromImage(grown).astype(bool)
    return Segmentation(
        mask=np.ascontiguousarray(mask),
        method="region_grow",
        params={"seed": (int(z), int(y), int(x)), "tolerance": tolerance},
    )
