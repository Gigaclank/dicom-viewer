"""Axis-aligned 3D bounding box in voxel space.

Coordinates are in (z, y, x). All ranges are half-open [start, stop)
matching numpy slice semantics. shape is (stop - start) per axis.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    z: tuple[int, int]
    y: tuple[int, int]
    x: tuple[int, int]

    def __post_init__(self) -> None:
        for axis_name, (lo, hi) in (("z", self.z), ("y", self.y), ("x", self.x)):
            if lo > hi:
                raise ValueError(f"Region.{axis_name} has lo>hi: {(lo, hi)}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            self.z[1] - self.z[0],
            self.y[1] - self.y[0],
            self.x[1] - self.x[0],
        )

    @property
    def is_empty(self) -> bool:
        return any(s <= 0 for s in self.shape)

    def intersect(self, other: "Region") -> "Region":
        def _clamp(lo: int, hi: int) -> tuple[int, int]:
            return (lo, max(lo, hi))

        z0 = max(self.z[0], other.z[0])
        y0 = max(self.y[0], other.y[0])
        x0 = max(self.x[0], other.x[0])
        return Region(
            z=_clamp(z0, min(self.z[1], other.z[1])),
            y=_clamp(y0, min(self.y[1], other.y[1])),
            x=_clamp(x0, min(self.x[1], other.x[1])),
        )

    def clamp_to(self, bounds: "Region") -> "Region":
        return self.intersect(bounds)

    def size_mm(self, spacing_mm: tuple[float, float, float]) -> tuple[float, float, float]:
        sz, sy, sx = self.shape
        dz, dy, dx = spacing_mm
        return (sz * dz, sy * dy, sx * dx)
