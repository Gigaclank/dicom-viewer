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
        self._volume_property: vtk.vtkVolumeProperty | None = None
        self._overlay_actor: vtk.vtkActor | None = None
        self._region_actor: vtk.vtkActor | None = None
        self._render_window: vtk.vtkRenderWindow | None = None
        # Cached so set_windowing can rebuild the transfer function whenever
        # the user picks a different preset without re-uploading the volume.
        self._windowing = None  # type: ignore[assignment]
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
        # Crosshair planes — three thin translucent rectangles indicating
        # where the AX/COR/SAG 2D slices currently are. Created lazily on
        # the first set_volume call. Voxel indices are stored so updates
        # don't need to re-pass them.
        self._crosshair_voxel: tuple[int, int, int] | None = None
        self._crosshair_sources: tuple[vtk.vtkPlaneSource, vtk.vtkPlaneSource, vtk.vtkPlaneSource] | None = None
        self._crosshair_actors: tuple[vtk.vtkActor, vtk.vtkActor, vtk.vtkActor] | None = None

        if os.environ.get("DICOM_VIEWER_OFFSCREEN") == "1" and sys.platform != "win32":
            rw = vtk.vtkRenderWindow()
            rw.SetOffScreenRendering(1)
            rw.AddRenderer(self._renderer)
            rw.SetSize(64, 64)
            self._render_window = rw

    def attach_render_window(self, render_window: vtk.vtkRenderWindow) -> None:
        render_window.AddRenderer(self._renderer)
        self._render_window = render_window
        # Force the trackball-camera style: rotate only while the left button
        # is held AND the mouse is moving; stop immediately on release. The
        # VTK default for a 3D scene is vtkInteractorStyleSwitch, which boots
        # in "joystick" mode — that mode keeps rotating as long as the cursor
        # is held offset from the click point, which feels like the camera
        # has a mind of its own. Explicitly installing TrackballCamera makes
        # rotation strictly mouse-motion-driven and stops the moment the
        # user releases or stops moving.
        interactor = render_window.GetInteractor()
        if interactor is not None:
            interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    def set_volume(self, volume: Volume) -> None:
        self._volume = volume
        if self._volume_actor is not None:
            self._renderer.RemoveVolume(self._volume_actor)
            self._volume_actor = None
        # Drop any crosshair actors from the previous volume — they'll be
        # rebuilt below with the new volume's dimensions.
        self._destroy_crosshair_actors()
        # Volume rendering only makes sense for >=2 slices. 2D modalities
        # (mammograms, plain X-rays) have z=1 — show nothing in the 3D pane.
        # The slice views still display them normally.
        if volume.shape[0] < 2:
            self._capture_home_camera()
            return
        image = _volume_to_vtk_image(volume)
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(image)
        prop = vtk.vtkVolumeProperty()
        prop.ShadeOn()
        actor = vtk.vtkVolume()
        actor.SetMapper(mapper)
        actor.SetProperty(prop)
        self._renderer.AddVolume(actor)
        self._volume_actor = actor
        self._volume_property = prop
        # Build the transfer functions from the current windowing (None means
        # use the modality default). They get rebuilt whenever windowing
        # changes so the 3D view tracks Bone/Lung/Soft-tissue presets etc.
        self._apply_transfer_functions(self._windowing)
        # Set a known default orientation BEFORE letting ResetCamera fit the
        # scene. ResetCamera preserves view direction and view-up, so without
        # this any rotation the user did on the previous volume would carry
        # over to the new one (and then get captured as that volume's "home").
        # Focal point goes at the volume's geometric center so the trackball
        # rotates around the middle of the scene, not the corner at origin.
        z, y, x = volume.shape
        sz, sy, sx = volume.spacing_mm
        cx, cy, cz = x * sx * 0.5, y * sy * 0.5, z * sz * 0.5
        cam = self._renderer.GetActiveCamera()
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(cx, cy - 1.0, cz)  # looking anteriorly (-Y)
        cam.SetViewUp(0.0, 0.0, 1.0)       # Z is up
        self._renderer.ResetCamera()
        self._capture_home_camera()
        # Crosshair planes — start at the volume's geometric center so the
        # user immediately sees where each 2D pane is pointing. Updates flow
        # in from MainWindow as the user scrubs.
        self._build_crosshair_actors()
        self.set_crosshair_position(z // 2, y // 2, x // 2)

    # --- crosshair planes -------------------------------------------------
    def _build_crosshair_actors(self) -> None:
        """Create three translucent axis-aligned plane actors marking where
        the AX/COR/SAG 2D slices currently are. Geometry is set later by
        ``set_crosshair_position`` — here we just instantiate the actors
        with their colors and opacity."""
        # Convention: AX = red (looking down Z), COR = green (along Y),
        # SAG = blue (along X). Matches most clinical viewers.
        colors = ((1.0, 0.25, 0.25), (0.25, 1.0, 0.25), (0.25, 0.45, 1.0))
        sources: list[vtk.vtkPlaneSource] = []
        actors: list[vtk.vtkActor] = []
        for color in colors:
            src = vtk.vtkPlaneSource()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(src.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(*color)
            # ~18% so the volume render is still legible underneath.
            prop.SetOpacity(0.18)
            # Disable lighting so the plane keeps its color regardless of
            # camera angle — these are indicator overlays, not lit surfaces.
            prop.LightingOff()
            self._renderer.AddActor(actor)
            sources.append(src)
            actors.append(actor)
        self._crosshair_sources = (sources[0], sources[1], sources[2])
        self._crosshair_actors = (actors[0], actors[1], actors[2])

    def _destroy_crosshair_actors(self) -> None:
        if self._crosshair_actors is not None:
            for a in self._crosshair_actors:
                self._renderer.RemoveActor(a)
        self._crosshair_actors = None
        self._crosshair_sources = None
        self._crosshair_voxel = None

    def set_windowing(self, center: float, width: float) -> None:
        """Rebuild the 3D opacity / colour transfer functions for a new
        windowing preset. The iso threshold is the windowing center, so
        switching from Bone (center≈500) to Lung (center≈-600) re-isos the
        rendered surface accordingly. No-op when no volume is loaded.
        """
        # Stash so subsequent set_volume re-applies the same window before
        # the next external set_windowing call.
        from dicom_viewer.core.document import WindowingState

        self._windowing = WindowingState(center=float(center), width=float(width))
        if self._volume is None or self._volume_property is None:
            return
        self._apply_transfer_functions(self._windowing)
        self.render()

    def _apply_transfer_functions(self, windowing) -> None:
        """Rebuild and install the opacity + colour transfer functions on
        the current volume property using ``windowing`` to set the iso
        threshold. ``windowing=None`` uses the modality default. Centralised
        here so set_volume and set_windowing share the math.
        """
        if self._volume is None or self._volume_property is None:
            return
        from dicom_viewer.core.mesh_export import iso_threshold_for_view

        volume = self._volume
        iso = iso_threshold_for_view(volume, windowing)
        lo, hi = volume.intensity_range()
        opacity = vtk.vtkPiecewiseFunction()
        color = vtk.vtkColorTransferFunction()
        if volume.modality == "CT":
            opacity.AddPoint(lo - 1, 0.0)
            opacity.AddPoint(iso - 1, 0.0)
            opacity.AddPoint(iso, 0.5)
            opacity.AddPoint(max(hi, iso + 1), 0.9)
            color.AddRGBPoint(iso - 1, 0.4, 0.2, 0.1)
            color.AddRGBPoint(iso, 0.9, 0.8, 0.7)
            color.AddRGBPoint(max(hi, iso + 1), 1.0, 1.0, 1.0)
        else:
            opacity.AddPoint(lo, 0.0)
            opacity.AddPoint(iso, 0.05)
            opacity.AddPoint(hi, 0.8)
            color.AddRGBPoint(lo, 0.1, 0.1, 0.2)
            color.AddRGBPoint(hi, 1.0, 1.0, 1.0)
        self._volume_property.SetColor(color)
        self._volume_property.SetScalarOpacity(opacity)

    def set_crosshair_position(self, z: int, y: int, x: int) -> None:
        """Move the AX/COR/SAG indicator planes to the given voxel indices.
        Coordinates are clamped to the volume bounds so the planes never
        leave the data. No-op when no volume is loaded."""
        if self._volume is None or self._crosshair_sources is None:
            return
        Z, Y, X = self._volume.shape
        sz, sy, sx = self._volume.spacing_mm
        # Clamp so callers can pass un-checked slice indices.
        z = max(0, min(int(z), Z - 1))
        y = max(0, min(int(y), Y - 1))
        x = max(0, min(int(x), X - 1))
        self._crosshair_voxel = (z, y, x)

        W = X * sx
        H = Y * sy
        D = Z * sz
        # Use mid-voxel world coords so the plane sits between texels rather
        # than on the boundary (matches how slice renderers position images).
        wz = (z + 0.5) * sz
        wy = (y + 0.5) * sy
        wx = (x + 0.5) * sx

        ax_src, co_src, sa_src = self._crosshair_sources
        # Axial plane: spans X and Y, fixed at world Z = wz.
        ax_src.SetOrigin(0.0, 0.0, wz)
        ax_src.SetPoint1(W, 0.0, wz)
        ax_src.SetPoint2(0.0, H, wz)
        # Coronal plane: spans X and Z, fixed at world Y = wy.
        co_src.SetOrigin(0.0, wy, 0.0)
        co_src.SetPoint1(W, wy, 0.0)
        co_src.SetPoint2(0.0, wy, D)
        # Sagittal plane: spans Y and Z, fixed at world X = wx.
        sa_src.SetOrigin(wx, 0.0, 0.0)
        sa_src.SetPoint1(wx, H, 0.0)
        sa_src.SetPoint2(wx, 0.0, D)
        self.render()

    @property
    def crosshair_voxel(self) -> tuple[int, int, int] | None:
        return self._crosshair_voxel

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


# Conservative upper bound for any 3D-texture axis. Most desktop GPUs allow
# 2048 or more, but a tighter cap leaves headroom for VTK's internal
# allocations and avoids surprises on older hardware. The slice views and
# the STL export keep full resolution — only the 3D volume render is
# downsampled. Hitting this limit isn't a quality problem for a quick visual
# overview; the user still has the MPR panes for full detail.
_MAX_3D_TEXTURE_DIM = 1024


def _volume_to_vtk_image(volume: Volume) -> vtk.vtkImageData:
    arr = volume.array
    z, y, x = arr.shape
    sz, sy, sx = volume.spacing_mm

    # Downsample any axis that would exceed the GPU 3D-texture limit. Stride
    # downsampling is cheap (just a view); the spacing scales by the same
    # factor so the volume stays at the correct anatomical size in world space.
    factors = (
        max(1, _ceil_div(z, _MAX_3D_TEXTURE_DIM)),
        max(1, _ceil_div(y, _MAX_3D_TEXTURE_DIM)),
        max(1, _ceil_div(x, _MAX_3D_TEXTURE_DIM)),
    )
    if any(f > 1 for f in factors):
        fz, fy, fx = factors
        arr = arr[::fz, ::fy, ::fx]
        sz, sy, sx = sz * fz, sy * fy, sx * fx
        z, y, x = arr.shape

    image = vtk.vtkImageData()
    image.SetDimensions(x, y, z)
    image.SetSpacing(sx, sy, sz)
    image.SetOrigin(0, 0, 0)
    vtk_type = vtk.VTK_SHORT if arr.dtype == np.int16 else vtk.VTK_FLOAT
    flat = numpy_support.numpy_to_vtk(arr.ravel(order="C"), deep=True, array_type=vtk_type)
    image.GetPointData().SetScalars(flat)
    return image


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


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
