"""Marching-cubes mesh export pipeline.

generate_mesh: Volume × Segmentation × Region -> Mesh (VTK polydata + metadata)
export_stl:    Mesh -> on-disk binary STL.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support  # type: ignore[import-untyped]

from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.volume import Volume


class EmptyMeshError(Exception):
    """The chosen segmentation/region intersection produced zero voxels or zero triangles."""


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
) -> Mesh:
    bounds = volume.bbox()
    r = region.clamp_to(bounds)
    if r.is_empty:
        raise EmptyMeshError("region is empty after clamping to volume")

    cropped_mask = segmentation.mask[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]]
    if not cropped_mask.any():
        raise EmptyMeshError("no voxels selected within region")

    image = _mask_to_vtk_image(
        cropped_mask,
        volume.spacing_mm,
        origin_voxel=(r.z[0], r.y[0], r.x[0]),
        spacing=volume.spacing_mm,
    )

    marching = vtk.vtkDiscreteMarchingCubes()
    marching.SetInputData(image)
    marching.SetValue(0, 1)
    marching.Update()

    pipeline: vtk.vtkAlgorithm = marching

    if options.smoothing_iterations > 0:
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
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputConnection(pipeline.GetOutputPort())
        decimator.SetTargetReduction(options.decimation_target_reduction)
        decimator.Update()
        pipeline = decimator

    if options.ensure_manifold:
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

    poly: vtk.vtkPolyData = pipeline.GetOutput()
    n_tri = int(poly.GetNumberOfPolys())
    if n_tri == 0:
        raise EmptyMeshError("mesh has zero triangles after processing")

    vtk_bounds = poly.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    lo = (float(vtk_bounds[4]), float(vtk_bounds[2]), float(vtk_bounds[0]))  # (z,y,x)
    hi = (float(vtk_bounds[5]), float(vtk_bounds[3]), float(vtk_bounds[1]))
    return Mesh(polydata=poly, triangle_count=n_tri, bounds_mm=(lo, hi))


def export_stl(mesh: Mesh, path: Path) -> None:
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(mesh.polydata)
    if writer.Write() != 1:
        raise OSError(f"vtkSTLWriter failed to write {path}")


def _mask_to_vtk_image(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    origin_voxel: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> "vtk.vtkImageData":
    """Convert a (z,y,x) boolean numpy mask into a vtkImageData (x,y,z order)."""
    arr_uint = mask.astype(np.uint8)
    # VTK expects flat array in x-fastest order, but vtkImageData with the right dims
    # plus numpy_support handles the (z,y,x) -> (x,y,z) interpretation via SetDimensions.
    z, y, x = arr_uint.shape
    image = vtk.vtkImageData()
    image.SetDimensions(x, y, z)
    image.SetSpacing(spacing[2], spacing[1], spacing[0])
    image.SetOrigin(
        origin_voxel[2] * spacing[2],
        origin_voxel[1] * spacing[1],
        origin_voxel[0] * spacing[0],
    )
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=arr_uint.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    image.GetPointData().SetScalars(vtk_array)
    return image
