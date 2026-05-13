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
