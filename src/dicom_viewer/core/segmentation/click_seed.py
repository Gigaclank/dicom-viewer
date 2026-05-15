"""Click-seed brush — grow a mask from a single seed voxel and merge it into
the running segmentation (Add / Remove modes).

The brush is the interactive complement to the slider-driven Region grow
tool. It exists because tumour intensity rarely has a clean HU/intensity
cutoff: the user clicks inside the tumour to grow a region, then clicks
adjacent organs in Remove mode to subtract leaks. Cropping the grow to the
active Region keeps a single click from flood-filling half the scan.

This module hosts the math for every brush "kind" the UI exposes:
- ``grow_from_seed``: SITK ConnectedThreshold — connected, intensity-band.
- ``threshold_from_seed``: same intensity band, no connectivity (catches
  internal voids in lesions that flood-fill would miss).
- ``sphere_from_seed``: voxels within a world-space radius of the click.
- ``box_from_seed``: axis-aligned box around the click.
- ``paint_disc_2d``: mutate one slice with a 2D disc — the per-event call
  that 2D paint brushes use during a drag.
- ``confidence_grow_from_seed``: SITK ConfidenceConnected — statistically
  driven region grow that handles soft-tissue tumours with diffuse edges
  better than a fixed-tolerance threshold.

All region-aware brushes accept an optional ``Region`` and constrain output
to that bbox so a click can't accidentally fill the entire scan.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from dicom_viewer.core.region import Region
from dicom_viewer.core.volume import Orientation, Volume


def grow_from_seed(
    volume: Volume,
    seed: tuple[int, int, int],
    tolerance: float,
    region: Region | None = None,
) -> np.ndarray:
    """Flood-fill from `seed` (z, y, x in voxel coords) within ±tolerance of
    the seed voxel's intensity. Returns a bool array matching ``volume.shape``.

    ``region``, if given, restricts the grow to its bounding box. The seed
    must lie inside that box; otherwise the result is empty (a click that
    landed outside the active region shouldn't surprise the user with a
    blank result that wipes their accumulated mask).
    """
    sz, sy, sx = volume.shape
    z, y, x = seed
    if not (0 <= z < sz and 0 <= y < sy and 0 <= x < sx):
        raise ValueError(f"seed {seed} outside volume shape {volume.shape}")

    seed_value = float(volume.array[z, y, x])
    lo = seed_value - float(tolerance)
    hi = seed_value + float(tolerance)

    if region is not None and not region.is_empty:
        r = region.clamp_to(volume.bbox())
        seed_inside = (
            r.z[0] <= z < r.z[1]
            and r.y[0] <= y < r.y[1]
            and r.x[0] <= x < r.x[1]
        )
        if not seed_inside:
            return np.zeros(volume.shape, dtype=bool)
        sub = volume.array[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]]
        local_seed = (z - r.z[0], y - r.y[0], x - r.x[0])
    else:
        sub = volume.array
        local_seed = (z, y, x)
        r = None

    image = sitk.GetImageFromArray(sub)
    grown = sitk.ConnectedThreshold(
        image,
        # SimpleITK indexes (x, y, z) — opposite of our (z, y, x) convention.
        seedList=[(int(local_seed[2]), int(local_seed[1]), int(local_seed[0]))],
        lower=lo,
        upper=hi,
        replaceValue=1,
    )
    local_mask = sitk.GetArrayFromImage(grown).astype(bool)

    mask = np.zeros(volume.shape, dtype=bool)
    if r is None:
        mask[:] = local_mask
    else:
        mask[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]] = local_mask
    return mask


def apply_brush_stroke(
    base_mask: np.ndarray | None,
    addition: np.ndarray,
    mode: str,
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """Merge a fresh region-grow result into the accumulated brush mask.

    ``mode`` is "add" (OR) or "remove" (AND-NOT). A None base means the
    accumulator hasn't started yet — Add seeds the mask; Remove no-ops.
    """
    if mode not in ("add", "remove"):
        raise ValueError(f"unknown brush mode: {mode!r}")
    if base_mask is None:
        if mode == "add":
            return addition.astype(bool, copy=True)
        return np.zeros(volume_shape, dtype=bool)
    if base_mask.shape != addition.shape:
        raise ValueError(
            f"shape mismatch: base {base_mask.shape} vs addition {addition.shape}"
        )
    if mode == "add":
        return np.logical_or(base_mask, addition)
    return np.logical_and(base_mask, np.logical_not(addition))


# --- additional brush kinds -------------------------------------------------


def _resolve_region(volume: Volume, region: Region | None) -> Region:
    """Return a region clamped to the volume bbox. Falls back to the full
    bbox when no region (or an empty one) is supplied."""
    if region is None or region.is_empty:
        return volume.bbox()
    return region.clamp_to(volume.bbox())


def threshold_from_seed(
    volume: Volume,
    seed: tuple[int, int, int],
    tolerance: float,
    region: Region | None = None,
) -> np.ndarray:
    """Mark every voxel within ±tolerance of the seed value, no connectivity
    check. Useful when a tumour has internal voids that flood-fill would
    skip — threshold catches every voxel in the intensity band, leaving the
    user to clean up extra structures with Remove clicks.
    """
    sz, sy, sx = volume.shape
    z, y, x = seed
    if not (0 <= z < sz and 0 <= y < sy and 0 <= x < sx):
        raise ValueError(f"seed {seed} outside volume shape {volume.shape}")
    seed_value = float(volume.array[z, y, x])
    lo, hi = seed_value - float(tolerance), seed_value + float(tolerance)
    r = _resolve_region(volume, region)
    mask = np.zeros(volume.shape, dtype=bool)
    sub = volume.array[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]]
    mask[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]] = (sub >= lo) & (sub <= hi)
    return mask


def sphere_from_seed(
    volume: Volume,
    seed: tuple[int, int, int],
    radius_mm: float,
    region: Region | None = None,
) -> np.ndarray:
    """Mark every voxel within ``radius_mm`` of the seed (Euclidean distance
    in world coordinates, so anisotropic spacing is respected). Constrained
    to ``region`` if supplied."""
    sz, sy, sx = volume.shape
    z, y, x = seed
    if not (0 <= z < sz and 0 <= y < sy and 0 <= x < sx):
        raise ValueError(f"seed {seed} outside volume shape {volume.shape}")
    if radius_mm <= 0:
        return np.zeros(volume.shape, dtype=bool)
    r = _resolve_region(volume, region)
    spacing_z, spacing_y, spacing_x = volume.spacing_mm
    # Build a local grid covering the cropped region only — saves a lot of
    # memory on big scans where the radius is small.
    zz = (np.arange(r.z[0], r.z[1]) - z) * spacing_z
    yy = (np.arange(r.y[0], r.y[1]) - y) * spacing_y
    xx = (np.arange(r.x[0], r.x[1]) - x) * spacing_x
    dz, dy, dx = np.meshgrid(zz, yy, xx, indexing="ij")
    dist2 = dz * dz + dy * dy + dx * dx
    inside = dist2 <= (radius_mm * radius_mm)
    mask = np.zeros(volume.shape, dtype=bool)
    mask[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]] = inside
    return mask


def box_from_seed(
    volume: Volume,
    seed: tuple[int, int, int],
    half_extent_mm: float,
    region: Region | None = None,
) -> np.ndarray:
    """Mark an axis-aligned box centered on the seed, half-extent specified
    in mm so the box stays geometrically square even on anisotropic scans.
    Useful for blocking out a known anatomical region with one click."""
    sz, sy, sx = volume.shape
    z, y, x = seed
    if not (0 <= z < sz and 0 <= y < sy and 0 <= x < sx):
        raise ValueError(f"seed {seed} outside volume shape {volume.shape}")
    if half_extent_mm <= 0:
        return np.zeros(volume.shape, dtype=bool)
    r = _resolve_region(volume, region)
    spacing_z, spacing_y, spacing_x = volume.spacing_mm
    hz = max(1, int(round(half_extent_mm / spacing_z)))
    hy = max(1, int(round(half_extent_mm / spacing_y)))
    hx = max(1, int(round(half_extent_mm / spacing_x)))
    # Intersect the seed-centred box with the active region.
    z0 = max(r.z[0], z - hz)
    z1 = min(r.z[1], z + hz + 1)
    y0 = max(r.y[0], y - hy)
    y1 = min(r.y[1], y + hy + 1)
    x0 = max(r.x[0], x - hx)
    x1 = min(r.x[1], x + hx + 1)
    mask = np.zeros(volume.shape, dtype=bool)
    if z0 < z1 and y0 < y1 and x0 < x1:
        mask[z0:z1, y0:y1, x0:x1] = True
    return mask


def paint_disc_2d(
    mask: np.ndarray,
    orientation: Orientation,
    slice_index: int,
    center_yx: tuple[int, int],
    radius_px: int,
    *,
    set_value: bool = True,
) -> np.ndarray:
    """Mutate ``mask`` in place: paint a 2D disc on the slice picked by
    (``orientation``, ``slice_index``). ``center_yx`` is in the slice's
    pixel coords (row, col). ``radius_px`` is the disc radius in voxels —
    paint brushes don't try to be world-space because the user is drawing
    with the cursor and expects pixel-radius feedback. Returns the same
    ``mask`` array for ergonomics; the array is mutated regardless.

    ``set_value`` chooses paint (True) vs erase (False) — Remove mode uses
    erase to subtract from the running mask without going through
    apply_brush_stroke.
    """
    if mask.dtype != np.bool_:
        raise ValueError(f"mask must be bool, got {mask.dtype}")
    sz, sy, sx = mask.shape
    row, col = center_yx
    if radius_px <= 0:
        return mask
    # Resolve the slice plane back to (z, y, x) indices for the disc grid.
    # Each orientation gives us a 2D plane: we need (axis0_range, axis1_range)
    # for the row/col axes and a fixed slice index for the third.
    if orientation is Orientation.AXIAL:
        if not 0 <= slice_index < sz:
            return mask
        # rows = y, cols = x
        plane = mask[slice_index, :, :]
        h, w = sy, sx
    elif orientation is Orientation.CORONAL:
        if not 0 <= slice_index < sy:
            return mask
        # Volume.slice flips z for display; rows here are the FLIPPED z.
        plane = mask[::-1, slice_index, :]
        h, w = sz, sx
    elif orientation is Orientation.SAGITTAL:
        if not 0 <= slice_index < sx:
            return mask
        plane = mask[::-1, :, slice_index]
        h, w = sz, sy
    else:
        raise ValueError(f"unknown orientation: {orientation!r}")

    # Bounding window inside the plane to avoid allocating the full grid.
    r0, r1 = max(0, row - radius_px), min(h, row + radius_px + 1)
    c0, c1 = max(0, col - radius_px), min(w, col + radius_px + 1)
    if r0 >= r1 or c0 >= c1:
        return mask
    rr = np.arange(r0, r1) - row
    cc = np.arange(c0, c1) - col
    dy, dx = np.meshgrid(rr, cc, indexing="ij")
    disc = (dy * dy + dx * dx) <= (radius_px * radius_px)
    target = plane[r0:r1, c0:c1]
    if set_value:
        np.logical_or(target, disc, out=target)
    else:
        np.logical_and(target, ~disc, out=target)
    return mask


def confidence_grow_from_seed(
    volume: Volume,
    seed: tuple[int, int, int],
    multiplier: float,
    iterations: int = 4,
    initial_neighborhood_radius: int = 1,
    region: Region | None = None,
) -> np.ndarray:
    """SITK ConfidenceConnected — statistically driven region grow. Computes
    the mean and stddev of the seed's local neighborhood, then includes
    voxels within ``multiplier × stddev`` of the mean, iterating to refine.
    Handles soft-tissue tumours with diffuse boundaries better than fixed-
    tolerance ConnectedThreshold."""
    sz, sy, sx = volume.shape
    z, y, x = seed
    if not (0 <= z < sz and 0 <= y < sy and 0 <= x < sx):
        raise ValueError(f"seed {seed} outside volume shape {volume.shape}")
    r = _resolve_region(volume, region)
    inside = r.z[0] <= z < r.z[1] and r.y[0] <= y < r.y[1] and r.x[0] <= x < r.x[1]
    if not inside:
        return np.zeros(volume.shape, dtype=bool)
    sub = volume.array[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]]
    local_seed = (x - r.x[0], y - r.y[0], z - r.z[0])  # SITK is (x, y, z)
    image = sitk.GetImageFromArray(sub)
    grown = sitk.ConfidenceConnected(
        image,
        seedList=[local_seed],
        numberOfIterations=int(iterations),
        multiplier=float(multiplier),
        initialNeighborhoodRadius=int(initial_neighborhood_radius),
        replaceValue=1,
    )
    local_mask = sitk.GetArrayFromImage(grown).astype(bool)
    mask = np.zeros(volume.shape, dtype=bool)
    mask[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]] = local_mask
    return mask
