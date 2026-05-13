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


def test_overlay_mask_hides_volume_and_shows_surface():
    r = VolumeRenderer()
    r.set_volume(_vol())
    assert r._volume_actor is not None
    assert r._volume_actor.GetVisibility() == 1

    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[3:5, 3:5, 3:5] = True
    r.set_overlay_mask(mask)
    # Volume render hidden, surface actor present.
    assert r._volume_actor.GetVisibility() == 0
    assert r._overlay_actor is not None

    # Clearing the mask restores the volume render.
    r.set_overlay_mask(None)
    assert r._volume_actor.GetVisibility() == 1
    assert r._overlay_actor is None


def test_overlay_re_renders_when_region_changes():
    r = VolumeRenderer()
    r.set_volume(_vol())
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    r.set_overlay_mask(mask)
    first_actor = r._overlay_actor
    assert first_actor is not None

    # Tightening the region should produce a fresh surface actor (different
    # geometry under the hood).
    r.set_region(Region(z=(2, 4), y=(2, 6), x=(2, 6)))
    assert r._overlay_actor is not None
    assert r._overlay_actor is not first_actor
