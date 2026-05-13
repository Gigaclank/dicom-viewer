"""Renders one MPR slice using VTK.

In production, an interactor (QVTKRenderWindowInteractor) is attached from
the UI layer via `attach_interactor`. For tests, the renderer falls back to a
detached offscreen window when DICOM_VIEWER_OFFSCREEN=1 is set.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import vtk
from vtk.util import numpy_support  # type: ignore[import-untyped]

from dicom_viewer.core.volume import Orientation, Volume


class SliceRenderer:
    def __init__(self, orientation: Orientation) -> None:
        self.orientation = orientation
        self._volume: Volume | None = None
        self._index: int = 0
        self._center: float = 40
        self._width: float = 400
        self._overlay_mask: np.ndarray | None = None

        self._image_actor = vtk.vtkImageActor()
        self._overlay_actor = vtk.vtkImageActor()
        self._renderer = vtk.vtkRenderer()
        self._renderer.AddActor(self._image_actor)
        self._renderer.AddActor(self._overlay_actor)
        self._renderer.SetBackground(0.0, 0.0, 0.0)
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
        # Use the image-restricted interactor style so the user can pan/zoom
        # but NOT rotate the slice (rotation makes no sense for an MPR view).
        interactor = render_window.GetInteractor()
        if interactor is not None:
            interactor.SetInteractorStyle(vtk.vtkInteractorStyleImage())

    # --- inputs ---
    def set_volume(self, volume: Volume) -> None:
        self._volume = volume
        max_index = self._max_index()
        self._index = min(self._index, max_index)
        self._refresh_image()
        # Each new volume gets a fresh, centred default view — otherwise any
        # zoom/pan the user left from the previous volume would carry over.
        self.reset_view()

    def set_slice_index(self, index: int) -> None:
        if self._volume is None:
            self._index = max(index, 0)
            return
        self._index = max(0, min(index, self._max_index()))
        self._refresh_image()
        # The overlay is per-slice too — refresh so the mask follows the slice.
        self._refresh_overlay()

    def set_windowing(self, center: float, width: float) -> None:
        self._center = center
        self._width = max(width, 1.0)
        self._refresh_image()

    def set_overlay_mask(self, mask: Optional[np.ndarray]) -> None:
        self._overlay_mask = mask
        self._refresh_overlay()

    # --- output ---
    @property
    def current_index(self) -> int:
        return self._index

    def render(self) -> None:
        if self._render_window is not None:
            self._render_window.Render()

    def reset_view(self) -> None:
        """Restore default zoom/pan/orientation for this pane."""
        camera = self._renderer.GetActiveCamera()
        camera.SetViewUp(0.0, 1.0, 0.0)
        self._renderer.ResetCamera()
        self._renderer.ResetCameraClippingRange()
        self.render()

    # --- internals ---
    def _max_index(self) -> int:
        if self._volume is None:
            return 0
        sz, sy, sx = self._volume.shape
        if self.orientation is Orientation.AXIAL:
            return sz - 1
        if self.orientation is Orientation.CORONAL:
            return sy - 1
        return sx - 1

    def _refresh_image(self) -> None:
        if self._volume is None:
            return
        slice2d = self._volume.windowed(self.orientation, self._index, self._center, self._width)
        self._image_actor.SetInputData(_to_rgb_image(slice2d))

    def _refresh_overlay(self) -> None:
        if self._volume is None or self._overlay_mask is None:
            self._overlay_actor.SetInputData(_empty_image())
            return
        # Defensive: if the cached mask doesn't match the current volume's
        # shape (e.g. mid-transition while switching DICOMs), drop it rather
        # than slicing it out of bounds. The next set_overlay_mask call will
        # install a correctly-shaped mask.
        if self._overlay_mask.shape != self._volume.shape:
            self._overlay_actor.SetInputData(_empty_image())
            return
        # Slice the mask the same way the volume is sliced.
        mask_volume = Volume(
            array=self._overlay_mask.astype(np.uint8),
            spacing_mm=self._volume.spacing_mm,
            modality=self._volume.modality,
        )
        slice2d = mask_volume.slice(self.orientation, self._index)
        self._overlay_actor.SetInputData(_to_rgba_overlay(slice2d))


def _to_rgb_image(slice2d: np.ndarray) -> vtk.vtkImageData:
    h, w = slice2d.shape
    rgb = np.stack([slice2d, slice2d, slice2d], axis=-1).astype(np.uint8)
    image = vtk.vtkImageData()
    image.SetDimensions(w, h, 1)
    vtk_array = numpy_support.numpy_to_vtk(
        rgb.reshape(-1, 3), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_array.SetNumberOfComponents(3)
    image.GetPointData().SetScalars(vtk_array)
    return image


def _to_rgba_overlay(slice2d: np.ndarray) -> vtk.vtkImageData:
    h, w = slice2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_bool = slice2d > 0
    rgba[mask_bool, 0] = 255   # red
    rgba[mask_bool, 3] = 96    # semi-transparent
    image = vtk.vtkImageData()
    image.SetDimensions(w, h, 1)
    vtk_array = numpy_support.numpy_to_vtk(
        rgba.reshape(-1, 4), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_array.SetNumberOfComponents(4)
    image.GetPointData().SetScalars(vtk_array)
    return image


def _empty_image() -> vtk.vtkImageData:
    image = vtk.vtkImageData()
    image.SetDimensions(1, 1, 1)
    arr = numpy_support.numpy_to_vtk(
        np.zeros((1, 4), dtype=np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    arr.SetNumberOfComponents(4)
    image.GetPointData().SetScalars(arr)
    return image
