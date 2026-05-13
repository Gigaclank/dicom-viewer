import os

import numpy as np
import pytest

from dicom_viewer.core.region import Region
from dicom_viewer.core.volume import Volume
from dicom_viewer.rendering.volume_renderer import VolumeRenderer


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    monkeypatch.setenv("DICOM_VIEWER_OFFSCREEN", "1")


def _vol() -> Volume:
    arr = np.zeros((8, 8, 8), dtype=np.int16)
    arr[2:6, 2:6, 2:6] = 500
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_volume_renderer_smoke():
    r = VolumeRenderer()
    r.set_volume(_vol())
    r.render()


def test_volume_renderer_region_box():
    r = VolumeRenderer()
    r.set_volume(_vol())
    r.set_region(Region(z=(1, 5), y=(1, 5), x=(1, 5)))
    r.render()


def test_volume_renderer_handles_no_volume():
    r = VolumeRenderer()
    r.render()  # must not raise


def test_volume_renderer_reset_view_does_not_crash():
    r = VolumeRenderer()
    r.set_volume(_vol())
    r.reset_view()
