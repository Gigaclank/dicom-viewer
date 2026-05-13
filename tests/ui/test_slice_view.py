import numpy as np
import pytest

from dicom_viewer.core.volume import Orientation, Volume
from dicom_viewer.ui.widgets.slice_view import SliceView


@pytest.fixture
def vol() -> Volume:
    arr = np.zeros((6, 6, 6), dtype=np.int16)
    arr[1:5, 1:5, 1:5] = 500
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_slice_view_scrollbar_range_reflects_volume(qtbot, vol):
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    assert view.scrollbar.minimum() == 0
    assert view.scrollbar.maximum() == 5  # axial: z=6 -> max 5


def test_slice_view_scrollbar_updates_index(qtbot, vol):
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    view.scrollbar.setValue(3)
    assert view.current_index == 3


def test_slice_view_emits_slice_changed_signal(qtbot, vol):
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    with qtbot.waitSignal(view.slice_changed, timeout=500) as blocker:
        view.scrollbar.setValue(4)
    assert blocker.args == [4]
