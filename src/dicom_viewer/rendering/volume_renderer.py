"""3D volume rendering with optional segmentation overlay and region box."""
from __future__ import annotations

import os
import sys

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
        # Cache the most recently set mask + region so that updates to either
        # one re-render the overlay surface with the correct crop.
        self._latest_mask: np.ndarray | None = None
        self._region: Region | None = None
        # Snapshot of the camera state that we treat as the "home" position
        # for reset_view. Captured right after set_volume positions the camera
        # to fit the scene. None until a volume has been loaded.
        self._home_position: tuple[float, float, float] | None = None
        self._home_focal: tuple[float, float, float] | None = None
        self._home_view_up: tuple[float, float, float] | None = None
        self._home_parallel_scale: float | None = None

        if os.environ.get("DICOM_VIEWER_OFFSCREEN") == "1" and sys.platform != "win32":
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
        # Set a known default orientation BEFORE letting ResetCamera fit the
        # scene. ResetCamera preserves view direction and view-up, so without
        # this any rotation the user did on the previous volume would carry
        # over to the new one (and then get captured as that volume's "home").
        cam = self._renderer.GetActiveCamera()
        cam.SetPosition(0.0, -1.0, 0.0)  # looking anteriorly (-Y)
        cam.SetFocalPoint(0.0, 0.0, 0.0)
        cam.SetViewUp(0.0, 0.0, 1.0)     # Z is up
        self._renderer.ResetCamera()
        self._capture_home_camera()

    def _capture_home_camera(self) -> None:
        """Remember the current camera state as the reset target."""
        cam = self._renderer.GetActiveCamera()
        self._home_position = tuple(cam.GetPosition())
        self._home_focal = tuple(cam.GetFocalPoint())
        self._home_view_up = tuple(cam.GetViewUp())
        self._home_parallel_scale = float(cam.GetParallelScale())

    def set_region(self, region: Region) -> None:
        self._region = region
        # Re-apply the overlay mask so the surface re-crops to the new region.
        if self._latest_mask is not None:
            self.set_overlay_mask(self._latest_mask)
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
        """When a mask is set, hide the volume render and show ONLY the masked
        surface (cropped to the active region if one is set). This makes the 3D
        pane a live preview of what the STL export will look like. When the
        mask is cleared, the volume render comes back."""
        self._latest_mask = mask
        if self._overlay_actor is not None:
            self._renderer.RemoveActor(self._overlay_actor)
            self._overlay_actor = None

        if mask is None or self._volume is None or not mask.any():
            if self._volume_actor is not None:
                self._volume_actor.SetVisibility(True)
            return

        # Hide the underlying volume render so the surface shows clearly.
        if self._volume_actor is not None:
            self._volume_actor.SetVisibility(False)

        # Crop the mask to the active region if one is set — matches what the
        # STL pipeline does, so the preview reflects the actual export.
        cropped = self._crop_mask_to_region(mask)
        if not cropped.any():
            return

        image = _mask_to_vtk_image(cropped, self._volume.spacing_mm)
        mc = vtk.vtkDiscreteMarchingCubes()
        mc.SetInputData(image)
        mc.SetValue(0, 1)
        mc.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(mc.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.85, 0.85, 0.9)
        self._renderer.AddActor(actor)
        self._overlay_actor = actor

    def _crop_mask_to_region(self, mask: np.ndarray) -> np.ndarray:
        if self._volume is None:
            return mask
        bounds = self._volume.bbox()
        r = (self._region or bounds).clamp_to(bounds)
        if r.is_empty:
            return np.zeros_like(mask)
        cropped = np.zeros_like(mask)
        cropped[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]] = mask[
            r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]
        ]
        return cropped

    def render(self) -> None:
        if self._render_window is not None:
            self._render_window.Render()

    def reset_view(self) -> None:
        """Restore the camera to the position captured when the volume loaded.

        Plain ResetCamera() only re-distances the camera to fit the scene; it
        preserves the current view direction and view-up vector, so a user who
        has rotated the camera does NOT actually see the original orientation
        come back. Restoring the captured home state fixes that.
        """
        if (
            self._home_position is None
            or self._home_focal is None
            or self._home_view_up is None
        ):
            # No volume loaded yet — fall back to ResetCamera() so we at least
            # frame whatever's in the scene.
            self._renderer.ResetCamera()
        else:
            cam = self._renderer.GetActiveCamera()
            cam.SetPosition(*self._home_position)
            cam.SetFocalPoint(*self._home_focal)
            cam.SetViewUp(*self._home_view_up)
            if self._home_parallel_scale is not None:
                cam.SetParallelScale(self._home_parallel_scale)
        self._renderer.ResetCameraClippingRange()
        self.render()


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
