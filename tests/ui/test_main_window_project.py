"""End-to-end: settings round-trip through the MainWindow project save/load."""
import pytest

from dicom_viewer.io.project import PROJECT_EXTENSION, load_project, save_project
from dicom_viewer.ui.main_window import MainWindow
from tests.fixtures.make_synthetic_series import make_synthetic_ct_series


@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path, monkeypatch):
    """Keep tests from clobbering the user's real ~/.config/dicom-viewer."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


def test_collect_project_from_empty_window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    project = win.collect_project()
    assert project.version == 1
    assert project.source.kind == ""
    assert project.windowing.width > 0


def test_round_trip_project_through_main_window(qtbot, tmp_path):
    series_dir = make_synthetic_ct_series(tmp_path, shape=(6, 16, 16))

    # First window: open the folder, tweak settings, save project.
    win1 = MainWindow()
    qtbot.addWidget(win1)
    assert win1.open_folder_path(series_dir)
    # Pick values inside the slider's adapted range (synthetic CT has data in
    # roughly [-1024, -24] after rescale, with ~5% pad on each side).
    win1.segmentation_panel.low_slider.setValue(-500)
    win1.segmentation_panel.high_slider.setValue(0)
    win1.export_panel.smoothing_slider.setValue(7)
    win1.export_panel.decimation_slider.setFloatValue(0.25)

    project_path = tmp_path / f"proj{PROJECT_EXTENSION}"
    save_project(project_path, win1.collect_project())

    # Second window: load the project, expect settings + segmentation to come back.
    win2 = MainWindow()
    qtbot.addWidget(win2)
    assert win2.load_project_from_path(project_path)
    assert win2.export_panel.smoothing_slider.value() == 7
    assert abs(win2.export_panel.decimation_slider.float_value() - 0.25) < 1e-6
    assert win2.segmentation_panel.low_slider.value() == -500
    assert win2.segmentation_panel.high_slider.value() == 0
    assert win2.document.volume is not None
    # Applying the project triggered the live-preview, so a segmentation exists.
    assert win2.document.segmentation is not None


def test_save_project_records_last_project_pointer(qtbot, tmp_path):
    from dicom_viewer.io.config import load_last_project

    win = MainWindow()
    qtbot.addWidget(win)
    proj_path = tmp_path / f"last{PROJECT_EXTENSION}"
    win._save_project_to(proj_path)
    assert load_last_project() == proj_path


def test_load_corrupt_project_does_not_crash(qtbot, tmp_path, monkeypatch):
    bad = tmp_path / f"bad{PROJECT_EXTENSION}"
    bad.write_text("not even json", encoding="utf-8")
    # Silence the modal warning dialog so the test doesn't block.
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **kw: None))
    win = MainWindow()
    qtbot.addWidget(win)
    assert not win.load_project_from_path(bad)
