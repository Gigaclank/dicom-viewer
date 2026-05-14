"""Marching-cubes mesh export pipeline.

generate_mesh: Volume × Segmentation × Region -> Mesh (VTK polydata + metadata)
export_stl:    Mesh -> on-disk binary STL.

Two execution modes:
  * Full (preview_mode=False, default): the full mask goes through marching
    cubes -> smoothing -> decimation -> manifold fix. Slow on large scans
    (a 512x512x500 head CT can produce 3M+ triangles in seconds).
  * Preview (preview_mode=True): the mask is first max-pool downsampled to a
    target voxel budget. Triangle count drops roughly with the downsample
    factor squared; the resulting mesh is geometrically close enough for an
    interactive preview but the export should still be run in full mode.

generate_mesh accepts an optional `progress` callback that receives
(stage_name, fraction in 0..1) at each pipeline stage — UI consumers wire
this to a progress bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import vtk
from vtk.util import numpy_support  # type: ignore[import-untyped]

from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.volume import Volume

# Target voxel budget for preview mode. ~8M voxels -> roughly 200^3, which
# typically marches into a few hundred thousand triangles in well under a
# second on a laptop. Tunable if it feels too coarse / too slow.
PREVIEW_VOXEL_BUDGET = 8_000_000

ProgressCallback = Callable[[str, float], None]


class EmptyMeshError(Exception):
    """The chosen segmentation/region intersection produced zero voxels or zero triangles."""


def resolve_export_segmentation(
    volume: Volume,
    segmentation: "Segmentation | None",
    iso_value: float,
) -> "tuple[Segmentation, str]":
    """Pick the mask to mesh for STL export or preview.

    If a user-applied segmentation is present, use it. Otherwise fabricate a
    one-shot iso-surface mask by thresholding above `iso_value` (typically
    the active windowing center). Returns (segmentation, label) where
    `label` is `'user-segmentation'` or `'iso@<value>'` for UI display.

    Raises EmptyMeshError if neither path produces any voxels.
    """
    # Local import to avoid module-load cycle with segmentation.threshold ->
    # segmentation.base -> mesh_export (this module).
    from dicom_viewer.core.segmentation.threshold import threshold

    if segmentation is not None:
        return segmentation, "user-segmentation"

    _, hi = volume.intensity_range()
    iso_seg = threshold(volume, iso_value, max(hi, iso_value))
    if iso_seg.is_empty:
        raise EmptyMeshError(
            f"iso-surface at intensity {iso_value:.0f} selected zero voxels — "
            f"adjust windowing to a brightness threshold that includes the target."
        )
    return iso_seg, f"iso@{iso_value:.0f}"


@dataclass(frozen=True)
class ExportOptions:
    smoothing_iterations: int = 15
    pass_band: float = 0.1
    decimation_target_reduction: float = 0.5
    ensure_manifold: bool = True


@dataclass(frozen=True)
class Mesh:
    polydata: "vtk.vtkPolyData"
    triangle_count: int
    bounds_mm: tuple[tuple[float, float, float], tuple[float, float, float]]


def generate_mesh(
    volume: Volume,
    segmentation: Segmentation,
    region: Region,
    options: ExportOptions,
    *,
    preview_mode: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> Mesh:
    def report(stage: str, frac: float) -> None:
        if progress is not None:
            try:
                progress(stage, frac)
            except Exception:
                # The progress channel must never fail the pipeline.
                pass

    report("Cropping", 0.0)
    bounds = volume.bbox()
    r = region.clamp_to(bounds)
    if r.is_empty:
        raise EmptyMeshError("region is empty after clamping to volume")

    cropped_mask = segmentation.mask[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]]
    if not cropped_mask.any():
        raise EmptyMeshError("no voxels selected within region")

    voxel_spacing = volume.spacing_mm
    # World origin where the cropped block starts — this stays constant under
    # downsampling so the resulting mesh is at the correct anatomical position.
    world_origin = (
        r.z[0] * volume.spacing_mm[0],
        r.y[0] * volume.spacing_mm[1],
        r.x[0] * volume.spacing_mm[2],
    )

    if preview_mode and cropped_mask.size > PREVIEW_VOXEL_BUDGET:
        factor = int(
            np.ceil((cropped_mask.size / PREVIEW_VOXEL_BUDGET) ** (1.0 / 3.0))
        )
        report(f"Downsampling for preview ({factor}x)", 0.05)
        cropped_mask = _max_pool_downsample(cropped_mask, factor)
        voxel_spacing = (
            voxel_spacing[0] * factor,
            voxel_spacing[1] * factor,
            voxel_spacing[2] * factor,
        )
        if not cropped_mask.any():
            raise EmptyMeshError("preview downsample emptied the mask — try a finer region")

    image = _mask_to_vtk_image(cropped_mask, voxel_spacing, world_origin)

    report("Marching cubes", 0.20)
    marching = vtk.vtkDiscreteMarchingCubes()
    marching.SetInputData(image)
    marching.SetValue(0, 1)
    marching.Update()

    pipeline: vtk.vtkAlgorithm = marching

    if options.smoothing_iterations > 0:
        report("Smoothing", 0.50)
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(pipeline.GetOutputPort())
        smoother.SetNumberOfIterations(options.smoothing_iterations)
        smoother.SetPassBand(options.pass_band)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        pipeline = smoother

    if 0.0 < options.decimation_target_reduction < 1.0:
        report("Decimating", 0.70)
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputConnection(pipeline.GetOutputPort())
        decimator.SetTargetReduction(options.decimation_target_reduction)
        decimator.Update()
        pipeline = decimator

    if options.ensure_manifold:
        report("Fixing topology", 0.85)
        filler = vtk.vtkFillHolesFilter()
        filler.SetInputConnection(pipeline.GetOutputPort())
        filler.SetHoleSize(1e6)
        filler.Update()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(filler.GetOutputPort())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.Update()
        pipeline = normals

    report("Finalizing", 0.95)
    # Deep-copy the output so the returned Mesh is independent of the local
    # pipeline filters. Without this, the polydata is owned by the last filter's
    # output port; once that filter is garbage-collected (when this function
    # returns, or when the worker thread that called it exits) the data can end
    # up in an inconsistent state, causing blank renders or crashes on the
    # second use.
    poly = vtk.vtkPolyData()
    poly.DeepCopy(pipeline.GetOutput())
    n_tri = int(poly.GetNumberOfPolys())
    if n_tri == 0:
        raise EmptyMeshError("mesh has zero triangles after processing")

    vtk_bounds = poly.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    lo = (float(vtk_bounds[4]), float(vtk_bounds[2]), float(vtk_bounds[0]))  # (z,y,x)
    hi = (float(vtk_bounds[5]), float(vtk_bounds[3]), float(vtk_bounds[1]))

    report("Done", 1.0)
    return Mesh(polydata=poly, triangle_count=n_tri, bounds_mm=(lo, hi))


def export_stl(mesh: Mesh, path: Path) -> None:
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(mesh.polydata)
    if writer.Write() != 1:
        raise OSError(f"vtkSTLWriter failed to write {path}")


def _max_pool_downsample(mask: np.ndarray, factor: int) -> np.ndarray:
    """Block-wise max downsample for a boolean mask.

    Any voxel that's True in a `factor`^3 block survives — preserves the
    surface envelope of the segmentation. Trims edges to a multiple of
    `factor`; missing rim voxels are an acceptable trade for a much smaller
    mesh in preview mode.
    """
    if factor <= 1:
        return mask
    z, y, x = mask.shape
    nz, ny, nx = z // factor, y // factor, x // factor
    if nz == 0 or ny == 0 or nx == 0:
        # Mask is smaller than the factor — just return as-is.
        return mask
    trimmed = mask[: nz * factor, : ny * factor, : nx * factor]
    blocked = trimmed.reshape(nz, factor, ny, factor, nx, factor)
    return blocked.max(axis=(1, 3, 5))


def _mask_to_vtk_image(
    mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    world_origin_mm: tuple[float, float, float],
) -> "vtk.vtkImageData":
    """Convert a (z,y,x) boolean numpy mask into a vtkImageData (x,y,z order).

    voxel_spacing_mm is the size of one voxel (post-downsampling if applicable);
    world_origin_mm is the world-space position of the first voxel and must NOT
    be scaled by the downsample factor.
    """
    arr_uint = mask.astype(np.uint8)
    z, y, x = arr_uint.shape
    image = vtk.vtkImageData()
    image.SetDimensions(x, y, z)
    image.SetSpacing(voxel_spacing_mm[2], voxel_spacing_mm[1], voxel_spacing_mm[0])
    image.SetOrigin(world_origin_mm[2], world_origin_mm[1], world_origin_mm[0])
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=arr_uint.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    image.GetPointData().SetScalars(vtk_array)
    return image
