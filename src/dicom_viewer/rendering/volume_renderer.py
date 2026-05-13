"""3D volume rendering with optional segmentation overlay and region box."""
from __future__ import annotations

import os

import numpy as np
import vtk
from vtk.util import numpy_support  # type: ignore[import-untyped]

from dicom_viewer.core.region import Region
from dicom_viewer.core.volume import Volume


class VolumeRenderer:
    def __init__(self) -> None:
        self._volume: Volume | None = None
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.05, 0.05, 0.08)
        self._volume_actor: vtk.vtkVolume | None = None
        self._overlay_actor: vtk.vtkActor | None = None
        self._region_actor: vtk.vtkActor | None = None
        self._render_window: vtk.vtkRenderWindow | None = None

        if os.environ.get("DICOM_VIEWER_OFFSCREEN") == "1":
            rw = vtk.vtkRenderWindow()
            rw.SetOffScreenRendering(1)
            rw.AddRenderer(self._renderer)
            rw.SetSize(64, 64)
            self._render_window = rw

    def attach_render_window(self, render_window: vtk.vtkRenderWindow) -> None:
        render_window.AddRenderer(self._renderer)
        self._render_window = render_window

    def set_volume(self, volume: Volume) -> None:
        self._volume = volume
        if self._volume_actor is not None:
            self._renderer.RemoveVolume(self._volume_actor)
        image = _volume_to_vtk_image(volume)
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(image)
        prop = vtk.vtkVolumeProperty()
        opacity = vtk.vtkPiecewiseFunction()
        color = vtk.vtkColorTransferFunction()
        if volume.modality == "CT":
            # bone-emphasizing transfer function
            opacity.AddPoint(-1000, 0.0)
            opacity.AddPoint(150, 0.0)
            opacity.AddPoint(300, 0.5)
            opacity.AddPoint(1500, 0.9)
            color.AddRGBPoint(150, 0.4, 0.2, 0.1)
            color.AddRGBPoint(300, 0.9, 0.8, 0.7)
            color.AddRGBPoint(1500, 1.0, 1.0, 1.0)
        else:
            lo, hi = volume.intensity_range()
            opacity.AddPoint(lo, 0.0)
            opacity.AddPoint(lo + (hi - lo) * 0.3, 0.05)
            opacity.AddPoint(hi, 0.8)
            color.AddRGBPoint(lo, 0.1, 0.1, 0.2)
            color.AddRGBPoint(hi, 1.0, 1.0, 1.0)
        prop.SetColor(color)
        prop.SetScalarOpacity(opacity)
        prop.ShadeOn()
        actor = vtk.vtkVolume()
        actor.SetMapper(mapper)
        actor.SetProperty(prop)
        self._renderer.AddVolume(actor)
        self._volume_actor = actor
        self._renderer.ResetCamera()

    def set_region(self, region: Region) -> None:
        if self._region_actor is not None:
            self._renderer.RemoveActor(self._region_actor)
            self._region_actor = None
        if self._volume is None or region.is_empty:
            return
        sz, sy, sx = self._volume.spacing_mm
        cube = vtk.vtkCubeSource()
        cube.SetBounds(
            region.x[0] * sx,
            region.x[1] * sx,
            region.y[0] * sy,
            region.y[1] * sy,
            region.z[0] * sz,
            region.z[1] * sz,
        )
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(1.0, 0.8, 0.2)
        actor.GetProperty().SetLineWidth(2.0)
        self._renderer.AddActor(actor)
        self._region_actor = actor

    def set_overlay_mask(self, mask: np.ndarray | None) -> None:
        if self._overlay_actor is not None:
            self._renderer.RemoveActor(self._overlay_actor)
            self._overlay_actor = None
        if mask is None or self._volume is None or not mask.any():
            return
        image = _mask_to_vtk_image(mask, self._volume.spacing_mm)
        mc = vtk.vtkDiscreteMarchingCubes()
        mc.SetInputData(image)
        mc.SetValue(0, 1)
        mc.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(mc.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.95, 0.3, 0.3)
        actor.GetProperty().SetOpacity(0.6)
        self._renderer.AddActor(actor)
        self._overlay_actor = actor

    def render(self) -> None:
        if self._render_window is not None:
            self._render_window.Render()


def _volume_to_vtk_image(volume: Volume) -> vtk.vtkImageData:
    arr = volume.array
    z, y, x = arr.shape
    image = vtk.vtkImageData()
    image.SetDimensions(x, y, z)
    sz, sy, sx = volume.spacing_mm
    image.SetSpacing(sx, sy, sz)
    image.SetOrigin(0, 0, 0)
    vtk_type = vtk.VTK_SHORT if arr.dtype == np.int16 else vtk.VTK_FLOAT
    flat = numpy_support.numpy_to_vtk(arr.ravel(order="C"), deep=True, array_type=vtk_type)
    image.GetPointData().SetScalars(flat)
    return image


def _mask_to_vtk_image(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> vtk.vtkImageData:
    arr = mask.astype(np.uint8)
    z, y, x = arr.shape
    image = vtk.vtkImageData()
    image.SetDimensions(x, y, z)
    sz, sy, sx = spacing_mm
    image.SetSpacing(sx, sy, sz)
    image.SetOrigin(0, 0, 0)
    vtk_array = numpy_support.numpy_to_vtk(
        arr.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    image.GetPointData().SetScalars(vtk_array)
    return image
