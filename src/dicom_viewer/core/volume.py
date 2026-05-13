"""3D voxel volume with spacing and modality metadata.

The array layout is always (z, y, x). Methods that take an Orientation slice
through the array; the returned 2D arrays use display conventions:
- AXIAL slice (constant z): rows=y, cols=x.
- CORONAL slice (constant y): rows=z (flipped so superior is up), cols=x.
- SAGITTAL slice (constant x): rows=z (flipped), cols=y.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from dicom_viewer.core.region import Region


class Orientation(str, Enum):
    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"


@dataclass(frozen=True)
class Volume:
    array: np.ndarray  # shape (z, y, x); dtype int16 for CT, float32 for MR
    spacing_mm: tuple[float, float, float]  # (z, y, x)
    modality: str  # "CT", "MR", ...

    @property
    def shape(self) -> tuple[int, int, int]:
        z, y, x = self.array.shape
        return (z, y, x)

    def bbox(self) -> Region:
        z, y, x = self.shape
        return Region(z=(0, z), y=(0, y), x=(0, x))

    def slice(self, orientation: Orientation, index: int) -> np.ndarray:
        z, y, x = self.shape
        if orientation is Orientation.AXIAL:
            if not 0 <= index < z:
                raise IndexError(f"axial index {index} out of [0,{z})")
            return self.array[index, :, :]
        if orientation is Orientation.CORONAL:
            if not 0 <= index < y:
                raise IndexError(f"coronal index {index} out of [0,{y})")
            # Flip z so the rendered image shows superior at top.
            return self.array[:, index, :][::-1, :]
        if orientation is Orientation.SAGITTAL:
            if not 0 <= index < x:
                raise IndexError(f"sagittal index {index} out of [0,{x})")
            return self.array[:, :, index][::-1, :]
        raise ValueError(f"unknown orientation {orientation!r}")

    def windowed(
        self, orientation: Orientation, index: int, center: float, width: float
    ) -> np.ndarray:
        if width <= 0:
            raise ValueError("window width must be > 0")
        s = self.slice(orientation, index).astype(np.float32)
        lo = center - width / 2.0
        hi = center + width / 2.0
        scaled = np.clip((s - lo) / (hi - lo), 0.0, 1.0)
        return (scaled * 255.0 + 0.5).astype(np.uint8)

    def crop(self, region: Region) -> "Volume":
        bounds = self.bbox()
        r = region.clamp_to(bounds)
        if r.is_empty:
            raise ValueError("cropping with an empty region")
        sub = self.array[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]]
        return Volume(array=sub, spacing_mm=self.spacing_mm, modality=self.modality)

    def intensity_range(self) -> tuple[float, float]:
        return (float(self.array.min()), float(self.array.max()))

    def intensity_percentiles(self, low_pct: float, high_pct: float) -> tuple[float, float]:
        lo = float(np.percentile(self.array, low_pct))
        hi = float(np.percentile(self.array, high_pct))
        return lo, hi
