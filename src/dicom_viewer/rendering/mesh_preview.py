"""Preview the generated mesh before STL export."""
from __future__ import annotations

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
        # Camera snapshot for reset_view. Plain ResetCamera() only re-distances
        # to fit the scene; it preserves view direction and view-up, so a user
        # who has rotated wouldn't see their orientation come back.
        self._home_position: tuple[float, float, float] | None = None
        self._home_focal: tuple[float, float, float] | None = None
        self._home_view_up: tuple[float, float, float] | None = None
        self._home_parallel_scale: float | None = None

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
            # Set a known orientation BEFORE ResetCamera so rotation centers
            # on the mesh and the home state is reproducible across reloads.
            xmin, xmax, ymin, ymax, zmin, zmax = mesh.polydata.GetBounds()
            cx = (xmin + xmax) * 0.5
            cy = (ymin + ymax) * 0.5
            cz = (zmin + zmax) * 0.5
            cam = self._renderer.GetActiveCamera()
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(cx, cy - 1.0, cz)
            cam.SetViewUp(0.0, 0.0, 1.0)
            self._renderer.ResetCamera()
            self._capture_home_camera()
            self._needs_fit = False
        # Always recompute the clipping range so the new actor is visible.
        self._renderer.ResetCameraClippingRange()

    def _capture_home_camera(self) -> None:
        cam = self._renderer.GetActiveCamera()
        self._home_position = tuple(cam.GetPosition())
        self._home_focal = tuple(cam.GetFocalPoint())
        self._home_view_up = tuple(cam.GetViewUp())
        self._home_parallel_scale = float(cam.GetParallelScale())

    def render(self) -> None:
        if self._render_window is not None:
            self._render_window.Render()

    def reset_view(self) -> None:
        """Restore the camera to the position captured when the mesh first fit.

        ResetCamera() alone preserves the current view direction, so rotation
        wouldn't actually reset. Restoring the captured home state does.
        """
        if (
            self._home_position is None
            or self._home_focal is None
            or self._home_view_up is None
        ):
            self._renderer.ResetCamera()
        else:
            cam = self._renderer.GetActiveCamera()
            cam.SetPosition(*self._home_position)
            cam.SetFocalPoint(*self._home_focal)
            cam.SetViewUp(*self._home_view_up)
            if self._home_parallel_scale is not None:
                cam.SetParallelScale(self._home_parallel_scale)
        self._renderer.ResetCameraClippingRange()
        self._needs_fit = False
        self.render()
