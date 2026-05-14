"""Smoke tests for SliceRenderer using offscreen VTK."""
import os

import numpy as np
import pytest

from dicom_viewer.core.volume import Orientation, Volume
from dicom_viewer.rendering.slice_renderer import SliceRenderer


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    monkeypatch.setenv("DICOM_VIEWER_OFFSCREEN", "1")


def _vol() -> Volume:
    arr = np.zeros((6, 6, 6), dtype=np.int16)
    arr[1:5, 1:5, 1:5] = 500
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_slice_renderer_instantiates_offscreen():
    r = SliceRenderer(orientation=Orientation.AXIAL)
    r.set_volume(_vol())
    r.set_slice_index(3)
    r.set_windowing(center=250, width=500)
    r.render()
    assert r.current_index == 3


def test_slice_renderer_clamps_index():
    r = SliceRenderer(orientation=Orientation.AXIAL)
    r.set_volume(_vol())
    r.set_slice_index(999)
    assert r.current_index == 5  # last valid axial index
    r.set_slice_index(-5)
    assert r.current_index == 0


def test_slice_renderer_overlay_does_not_crash():
    r = SliceRenderer(orientation=Orientation.AXIAL)
    r.set_volume(_vol())
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[2:5, 2:5, 2:5] = True
    r.set_overlay_mask(mask)
    r.render()


def test_overlay_refreshes_when_slice_index_changes():
    """Regression: scrolling slices must update the overlay actor's image data,
    not leave it stale on the previously-displayed slice."""
    r = SliceRenderer(orientation=Orientation.AXIAL)
    r.set_volume(_vol())
    # Mask varies between slices so we can detect whether the overlay tracks them.
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[2, :, :] = True   # axial slice 2 fully masked
    mask[4, :, :] = False  # axial slice 4 fully unmasked
    r.set_overlay_mask(mask)
    r.set_slice_index(2)
    overlay_at_2 = r._overlay_actor.GetInput().GetPointData().GetScalars()
    sum_at_2 = sum(overlay_at_2.GetTuple4(i)[3] for i in range(overlay_at_2.GetNumberOfTuples()))
    r.set_slice_index(4)
    overlay_at_4 = r._overlay_actor.GetInput().GetPointData().GetScalars()
    sum_at_4 = sum(overlay_at_4.GetTuple4(i)[3] for i in range(overlay_at_4.GetNumberOfTuples()))
    assert sum_at_2 > 0    # slice 2 has alpha > 0 where mask is true
    assert sum_at_4 == 0   # slice 4 has no mask -> alpha all zero


def test_set_volume_auto_resets_camera_view_up():
    """Loading a new volume must snap zoom/orientation back to default — any
    zoom/pan from the previous volume should NOT carry over."""
    r = SliceRenderer(orientation=Orientation.AXIAL)
    r.set_volume(_vol())
    cam = r._renderer.GetActiveCamera()
    # Mess with the camera as if the user dragged it.
    cam.SetViewUp(1.0, 0.0, 0.0)
    cam.Zoom(3.0)
    assert cam.GetViewUp() != (0.0, 1.0, 0.0)

    # Loading another volume must restore the default view-up.
    r.set_volume(_vol())
    assert cam.GetViewUp() == (0.0, 1.0, 0.0)


def test_scroll_does_not_crash_when_mask_shape_predates_volume_swap():
    """Regression: when a new (larger) volume is set but the previous mask is
    still cached (no segmentation event has fired yet), scrolling past the
    old mask's z extent crashed with IndexError. The renderer should drop
    the mismatched mask instead."""
    r = SliceRenderer(orientation=Orientation.AXIAL)
    small = Volume(array=np.zeros((4, 4, 4), dtype=np.int16), spacing_mm=(1, 1, 1), modality="CT")
    big = Volume(array=np.zeros((20, 4, 4), dtype=np.int16), spacing_mm=(1, 1, 1), modality="CT")
    r.set_volume(small)
    mask = np.zeros(small.shape, dtype=bool)
    mask[1, :, :] = True
    r.set_overlay_mask(mask)
    # Now swap to the bigger volume WITHOUT updating the mask.
    r.set_volume(big)
    # Scrolling past the small mask's z extent must not crash.
    r.set_slice_index(15)
    assert r.current_index == 15


def test_slice_renderer_reset_view_restores_default_camera():
    r = SliceRenderer(orientation=Orientation.AXIAL)
    r.set_volume(_vol())
    # Move the camera arbitrarily; reset_view should restore a sensible default.
    cam = r._renderer.GetActiveCamera()
    cam.Zoom(3.0)
    cam.SetViewUp(0.0, 0.0, 1.0)  # nonsense up-vector for an axial slice
    r.reset_view()
    # View-up is restored to (0, 1, 0).
    assert cam.GetViewUp() == (0.0, 1.0, 0.0)


def test_zoom_at_pins_world_point_under_cursor():
    """Cursor-centered zoom invariant: the world point that lay under the
    cursor before the zoom must lie under the same cursor pixel afterwards.
    Otherwise a Ctrl+wheel zoom would feel like 'jump-then-scale'."""
    r = SliceRenderer(orientation=Orientation.AXIAL)
    r.set_volume(_vol())
    # Force a known window size so display-to-world is meaningful.
    if r._render_window is not None:
        r._render_window.SetSize(400, 300)
        r._render_window.Render()
    widget_h = 300
    cursor_x, cursor_y = 100, 80  # Qt coords (top-left origin)

    vtk_y = widget_h - cursor_y
    world_before = r._display_to_world(cursor_x, vtk_y)
    r.zoom_at(0.5, cursor_x, cursor_y, widget_h)  # zoom in 2x
    world_after_zoom_at_same_pixel = r._display_to_world(cursor_x, vtk_y)

    # The same screen pixel must map to (approximately) the same world point.
    assert world_after_zoom_at_same_pixel[0] == pytest.approx(world_before[0], abs=1e-3)
    assert world_after_zoom_at_same_pixel[1] == pytest.approx(world_before[1], abs=1e-3)


def test_zoom_at_changes_parallel_scale():
    r = SliceRenderer(orientation=Orientation.AXIAL)
    r.set_volume(_vol())
    if r._render_window is not None:
        r._render_window.SetSize(400, 300)
        r._render_window.Render()
    cam = r._renderer.GetActiveCamera()
    before = cam.GetParallelScale()
    r.zoom_at(0.5, 200, 150, 300)
    assert cam.GetParallelScale() == pytest.approx(before * 0.5, abs=1e-6)
