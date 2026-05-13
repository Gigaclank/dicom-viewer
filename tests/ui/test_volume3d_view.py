import numpy as np
import pytest

from dicom_viewer.core.region import Region
from dicom_viewer.core.volume import Volume
from dicom_viewer.ui.widgets.volume3d_view import Volume3DView


@pytest.fixture
def vol() -> Volume:
    arr = np.zeros((8, 8, 8), dtype=np.int16)
    arr[2:6, 2:6, 2:6] = 500
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_volume3d_view_set_volume_updates_label(qtbot, vol):
    view = Volume3DView()
    qtbot.addWidget(view)
    view.set_volume(vol)
    # Label includes the shape after a study loads.
    text = view._label.text()
    assert "8" in text
    assert "CT" in text


def test_volume3d_view_set_region_does_not_crash(qtbot, vol):
    view = Volume3DView()
    qtbot.addWidget(view)
    view.set_volume(vol)
    view.set_region(Region(z=(1, 5), y=(1, 5), x=(1, 5)))


def test_volume3d_view_overlay_does_not_crash(qtbot, vol):
    view = Volume3DView()
    qtbot.addWidget(view)
    view.set_volume(vol)
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[3:5, 3:5, 3:5] = True
    view.set_overlay_mask(mask)


def test_volume3d_view_reset_button_callable(qtbot, vol):
    view = Volume3DView()
    qtbot.addWidget(view)
    view.set_volume(vol)
    view.reset_button.click()  # must not raise
