"""Tests for the global StatusModel — the 'currently doing' bus that
panels register tasks against and the MainWindow renders into the status bar."""
from PyQt6.QtWidgets import QApplication

from dicom_viewer.ui.status_model import StatusModel


def _ensure_app():
    """StatusModel inherits QObject so it needs a QApplication context. The
    qt-bot fixture handles this automatically, but we don't need a widget
    here — just a process-wide app."""
    return QApplication.instance() or QApplication([])


def test_idle_when_no_tasks(qtbot):
    _ensure_app()
    m = StatusModel()
    assert m.is_idle
    assert m.render() == "Idle"


def test_single_task_renders_with_label(qtbot):
    _ensure_app()
    m = StatusModel()
    m.begin("a", "Doing thing A")
    assert not m.is_idle
    assert m.render() == "Currently doing: Doing thing A"


def test_multiple_tasks_render_with_count(qtbot):
    """Two simultaneous workers should be summarized: count + first label."""
    _ensure_app()
    m = StatusModel()
    m.begin("a", "Task A")
    m.begin("b", "Task B")
    s = m.render()
    assert "2 tasks" in s
    assert "Task A" in s  # first-started shown verbatim


def test_update_changes_label_only_if_task_active(qtbot):
    """Late progress signals arriving after end() must not resurrect a task."""
    _ensure_app()
    m = StatusModel()
    m.begin("a", "Initial")
    m.update("a", "Progressing")
    assert m.render() == "Currently doing: Progressing"
    m.end("a")
    m.update("a", "After end")
    assert m.is_idle


def test_changed_signal_fires_on_each_mutation(qtbot):
    _ensure_app()
    m = StatusModel()
    fired: list[int] = []
    m.changed.connect(lambda: fired.append(1))
    m.begin("a", "A")
    m.update("a", "A2")
    m.end("a")
    # 3 mutations -> 3 signals.
    assert len(fired) == 3


def test_end_unknown_task_is_a_noop(qtbot):
    _ensure_app()
    m = StatusModel()
    fired: list[int] = []
    m.changed.connect(lambda: fired.append(1))
    m.end("missing")
    assert fired == []
