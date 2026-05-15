import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent

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
    Step size is configurable on the SliceView class (_SCRUB_STEP); the test
    asserts the direction and consistency without pinning the exact value."""
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


def test_set_brush_mode_validates(qtbot, vol):
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_brush_mode("add")
    assert view.brush_mode == "add"
    view.set_brush_mode("off")
    assert view.brush_mode == "off"
    with pytest.raises(ValueError):
        view.set_brush_mode("draw")


def test_mouse_click_in_off_mode_does_not_emit_seed(qtbot, vol):
    """Default 'off' mode preserves VTK's normal left-click interaction
    (window/level) — the brush signal must not fire."""
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    captured: list = []
    view.seed_clicked.connect(lambda seed, mode: captured.append((seed, mode)))
    me = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(10.0, 10.0),
        QPointF(10.0, 10.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    consumed = view.eventFilter(view._vtk_widget, me)
    assert consumed is False
    assert captured == []


def test_mouse_click_in_add_mode_emits_seed_when_vtk_attached(qtbot, vol, monkeypatch):
    """In add mode with a real VTK render window, a left-click emits
    seed_clicked. Headless test mode doesn't construct the VTK widget (the
    code falls back to a QLabel placeholder), so we stub the brush gate
    instead: pretend there's a render window, monkey-patch display-to-world
    to return a known voxel, and verify the signal."""
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    view.set_brush_mode("add")
    # Simulate the production "VTK widget present" branch.
    monkeypatch.setattr(view._vtk_widget, "GetRenderWindow", lambda: object(), raising=False)
    monkeypatch.setattr(view._vtk_widget, "height", lambda: 100)
    # Force display→world to a known in-volume point (col=3, row=4).
    monkeypatch.setattr(view._renderer, "_display_to_world", lambda x, y: (3.0, 4.0, 0.0))
    captured: list = []
    view.seed_clicked.connect(lambda seed, mode: captured.append((seed, mode)))
    me = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(5.0, 5.0),
        QPointF(5.0, 5.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    consumed = view.eventFilter(view._vtk_widget, me)
    assert consumed is True
    (seed, mode) = captured[0]
    z, y, x = seed
    # Axial: slice index = z; row → y; col → x.
    assert z == view.current_index
    assert (y, x) == (4, 3)
    assert mode == "add"


def test_mouse_click_in_add_mode_without_vtk_widget_is_safe(qtbot, vol):
    """In headless test mode the VTK widget is a placeholder QLabel — the
    brush handler must gate on the real widget presence so a synthesized
    click doesn't crash and doesn't emit a bogus seed."""
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    view.set_brush_mode("add")
    captured: list = []
    view.seed_clicked.connect(lambda seed, mode: captured.append((seed, mode)))
    me = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(5.0, 5.0),
        QPointF(5.0, 5.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.eventFilter(view._vtk_widget, me)
    assert captured == []


def test_drag_in_brush_mode_emits_seed_dragged_stream(qtbot, vol, monkeypatch):
    """Holding left button + moving the mouse in brush mode should fire
    seed_dragged for each move event so the 2D paint brush can stream
    stroke positions. seed_drag_ended fires on release."""
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    view.set_brush_mode("add")
    monkeypatch.setattr(view._vtk_widget, "GetRenderWindow", lambda: object(), raising=False)
    monkeypatch.setattr(view._vtk_widget, "height", lambda: 100)
    # Display→world returns moving positions so each MouseMove maps to a
    # different voxel.
    world_seq = iter([(2.0, 2.0, 0.0), (3.0, 3.0, 0.0), (4.0, 4.0, 0.0)])
    monkeypatch.setattr(view._renderer, "_display_to_world", lambda x, y: next(world_seq))

    clicked: list = []
    dragged: list = []
    ended: list = []
    view.seed_clicked.connect(lambda s, m: clicked.append((s, m)))
    view.seed_dragged.connect(lambda s, m: dragged.append((s, m)))
    view.seed_drag_ended.connect(lambda m: ended.append(m))

    # Press → click event.
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(5.0, 5.0), QPointF(5.0, 5.0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.eventFilter(view._vtk_widget, press)
    # Move twice while button held — each emits a drag.
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(6.0, 6.0), QPointF(6.0, 6.0),
        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.eventFilter(view._vtk_widget, move)
    move2 = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(7.0, 7.0), QPointF(7.0, 7.0),
        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.eventFilter(view._vtk_widget, move2)
    # Release → drag-ended event.
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(7.0, 7.0), QPointF(7.0, 7.0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.eventFilter(view._vtk_widget, release)

    assert len(clicked) == 1
    assert len(dragged) == 2
    assert ended == ["add"]
    # Drag tracking should have stopped now — a move with no held button
    # must NOT fire another drag.
    monkeypatch.setattr(view._renderer, "_display_to_world", lambda x, y: (5.0, 5.0, 0.0))
    bare_move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(8.0, 8.0), QPointF(8.0, 8.0),
        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.eventFilter(view._vtk_widget, bare_move)
    assert len(dragged) == 2  # unchanged


def test_drag_in_off_mode_does_not_emit(qtbot, vol, monkeypatch):
    """When brush is Off, drag tracking must be inactive — no seed events."""
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    # brush_mode remains 'off' by default.
    monkeypatch.setattr(view._vtk_widget, "GetRenderWindow", lambda: object(), raising=False)
    monkeypatch.setattr(view._vtk_widget, "height", lambda: 100)
    monkeypatch.setattr(view._renderer, "_display_to_world", lambda x, y: (2.0, 2.0, 0.0))
    captured: list = []
    view.seed_dragged.connect(lambda s, m: captured.append((s, m)))
    # A MouseMove with no preceding brush-mode press shouldn't fire anything.
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(6.0, 6.0), QPointF(6.0, 6.0),
        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.eventFilter(view._vtk_widget, move)
    assert captured == []


def test_right_click_in_brush_mode_does_not_emit(qtbot, vol):
    """Brush only listens for LEFT clicks — middle/right stay free for
    VTK's pan/contrast interactions."""
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    view.set_brush_mode("add")
    captured: list = []
    view.seed_clicked.connect(lambda seed, mode: captured.append((seed, mode)))
    me = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(5.0, 5.0),
        QPointF(5.0, 5.0),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.eventFilter(view._vtk_widget, me)
    assert captured == []


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
