import os

import numpy as np
import pytest

from dicom_viewer.core.mesh_export import ExportOptions, generate_mesh
from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.volume import Volume
from dicom_viewer.rendering.mesh_preview import MeshPreview


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    monkeypatch.setenv("DICOM_VIEWER_OFFSCREEN", "1")


def _cube_mesh():
    arr = np.zeros((16, 16, 16), dtype=np.int16)
    arr[4:12, 4:12, 4:12] = 1000
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    seg = threshold(v, low=500, high=2000)
    return generate_mesh(v, seg, region=v.bbox(), options=ExportOptions())


def test_mesh_preview_displays_mesh():
    mp = MeshPreview()
    mp.set_mesh(_cube_mesh())
    mp.render()


def test_mesh_preview_clears():
    mp = MeshPreview()
    mp.set_mesh(_cube_mesh())
    mp.set_mesh(None)
    mp.render()


def test_reset_view_restores_orientation_not_just_distance():
    """Regression: clicking Reset View on the STL preview must actually undo
    a rotation. vtkRenderer.ResetCamera() alone preserves view direction and
    view-up, so a rotated camera stayed rotated and only the zoom got reset."""
    mp = MeshPreview()
    mp.set_mesh(_cube_mesh())
    cam = mp._renderer.GetActiveCamera()
    home_position = cam.GetPosition()
    home_view_up = cam.GetViewUp()

    # Simulate the user dragging the preview.
    cam.SetPosition(home_position[0] + 50, home_position[1] + 25, home_position[2] - 10)
    cam.SetViewUp(1.0, 0.0, 0.0)
    assert cam.GetPosition() != home_position
    assert cam.GetViewUp() != home_view_up

    mp.reset_view()
    assert cam.GetPosition() == home_position
    assert cam.GetViewUp() == home_view_up


def test_reset_view_without_any_mesh_does_not_crash():
    """No mesh loaded yet: reset_view must be a no-op rather than blowing up
    trying to dereference a never-captured home camera."""
    MeshPreview().reset_view()
