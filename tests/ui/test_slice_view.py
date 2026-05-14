import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent

from dicom_viewer.core.volume import Orientation, Volume
from dicom_viewer.ui.widgets.slice_view import SliceView


def _wheel(view: SliceView, delta_y: int, modifiers=Qt.KeyboardModifier.NoModifier) -> bool:
    """Synthesize a QWheelEvent on the SliceView's VTK widget and route it
    through the installed event filter."""
    pos = QPointF(10.0, 10.0)
    global_pos = QPointF(10.0, 10.0)
    event = QWheelEvent(
        pos,
        global_pos,
        QPoint(0, delta_y),
        QPoint(0, delta_y),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    return view.eventFilter(view._vtk_widget, event)


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


def test_slice_view_reset_button_present_and_resets(qtbot, vol):
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    assert view.reset_button.text() == "Reset view"
    # Perturb the camera, then click reset.
    cam = view._renderer._renderer.GetActiveCamera()
    cam.Zoom(2.0)
    cam.SetViewUp(0.0, 0.0, 1.0)
    view.reset_button.click()
    assert cam.GetViewUp() == (0.0, 1.0, 0.0)


def test_wheel_event_scrubs_slices(qtbot):
    """Mouse wheel on the slice viewport should step through slices, not zoom.
    Default step is configurable; we just verify direction + magnitude > 1
    (single-slice steps proved too slow on real CT volumes)."""
    arr = np.zeros((40, 6, 6), dtype=np.int16)
    big = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(big)
    view.scrollbar.setValue(20)
    # Wheel "back" (negative delta) advances by the configured step.
    consumed = _wheel(view, -120)
    assert consumed is True
    assert view.scrollbar.value() == 20 + view._SCRUB_STEP
    assert view._SCRUB_STEP > 1, "single-slice scrub is too slow on real CTs"
    # Wheel "forward" (positive delta) steps back by the same amount.
    _wheel(view, 120)
    assert view.scrollbar.value() == 20


def test_wheel_with_shift_scrubs_in_bigger_steps(qtbot):
    """Shift+wheel scrubs faster (multiplied step) for long scans."""
    arr = np.zeros((400, 6, 6), dtype=np.int16)
    big = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(big)
    view.scrollbar.setValue(100)
    expected_step = view._SCRUB_STEP * view._SCRUB_SHIFT_MULTIPLIER
    _wheel(view, -120, modifiers=Qt.KeyboardModifier.ShiftModifier)
    assert view.scrollbar.value() == 100 + expected_step


def test_wheel_with_ctrl_zooms_and_does_not_scrub(qtbot, vol):
    """Ctrl+wheel must zoom (changing ParallelScale) and not change the slice
    index. The cursor-pinning behavior is exercised separately in
    test_slice_renderer with a sized render window — here we only check that
    the SliceView routes Ctrl+wheel into zoom_at instead of falling through."""
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    view.scrollbar.setValue(3)
    cam = view._renderer._renderer.GetActiveCamera()
    scale_before = cam.GetParallelScale()
    consumed = _wheel(view, -120, modifiers=Qt.KeyboardModifier.ControlModifier)
    assert consumed is True
    assert view.scrollbar.value() == 3
    assert cam.GetParallelScale() != pytest.approx(scale_before)
    # Zoom direction sanity: wheel back zooms out (parallel scale increases).
    assert cam.GetParallelScale() > scale_before


def test_wheel_clamps_at_volume_edges(qtbot, vol):
    """Scrolling past the last slice does not wrap or overflow."""
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    view.scrollbar.setValue(view.scrollbar.maximum())
    _wheel(view, -120)  # would go past max
    assert view.scrollbar.value() == view.scrollbar.maximum()
    view.scrollbar.setValue(0)
    _wheel(view, 120)  # would go below 0
    assert view.scrollbar.value() == 0
