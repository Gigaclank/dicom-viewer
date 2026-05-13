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
