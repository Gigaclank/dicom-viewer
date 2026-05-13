"""Preview the generated mesh before STL export."""
from __future__ import annotations

import os

import vtk

from dicom_viewer.core.mesh_export import Mesh


class MeshPreview:
    def __init__(self) -> None:
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.1, 0.1, 0.12)
        self._actor: vtk.vtkActor | None = None
        self._render_window: vtk.vtkRenderWindow | None = None
        # ResetCamera should only run when something new arrives that the
        # user hasn't oriented yet (first mesh of a session, after a study
        # swap, or on an explicit Reset). Settings-driven re-renders should
        # leave the user's current rotation/zoom alone.
        self._needs_fit: bool = True

    def attach_render_window(self, render_window: vtk.vtkRenderWindow) -> None:
        render_window.AddRenderer(self._renderer)
        self._render_window = render_window

    def request_fit(self) -> None:
        """Request that the next set_mesh() re-frame the camera. Call when
        the underlying study changes — the old camera pose may no longer
        make sense for a totally different geometry."""
        self._needs_fit = True

    def set_mesh(self, mesh: Mesh | None) -> None:
        if self._actor is not None:
            self._renderer.RemoveActor(self._actor)
            self._actor = None
        if mesh is None:
            return
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh.polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.85, 0.85, 0.9)
        self._renderer.AddActor(actor)
        self._actor = actor
        if self._needs_fit:
            self._renderer.ResetCamera()
            self._needs_fit = False
        # Always recompute the clipping range so the new actor is visible.
        self._renderer.ResetCameraClippingRange()

    def render(self) -> None:
        if self._render_window is not None:
            self._render_window.Render()

    def reset_view(self) -> None:
        """Re-fit the camera to whatever's currently in the scene."""
        self._renderer.ResetCamera()
        self._renderer.ResetCameraClippingRange()
        self._needs_fit = False
        self.render()
