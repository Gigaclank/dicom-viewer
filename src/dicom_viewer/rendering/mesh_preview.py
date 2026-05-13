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

        if os.environ.get("DICOM_VIEWER_OFFSCREEN") == "1":
            rw = vtk.vtkRenderWindow()
            rw.SetOffScreenRendering(1)
            rw.AddRenderer(self._renderer)
            rw.SetSize(64, 64)
            self._render_window = rw

    def attach_render_window(self, render_window: vtk.vtkRenderWindow) -> None:
        render_window.AddRenderer(self._renderer)
        self._render_window = render_window

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
        self._renderer.ResetCamera()

    def render(self) -> None:
        if self._render_window is not None:
            self._render_window.Render()

    def reset_view(self) -> None:
        """Restore default zoom/orientation for the mesh preview camera."""
        self._renderer.ResetCamera()
        self._renderer.ResetCameraClippingRange()
        self.render()
