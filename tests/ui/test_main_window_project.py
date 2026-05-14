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


def test_series_combo_populated_after_loading_multi_series_folder(qtbot, tmp_path):
    """Loading a folder with several series populates the toolbar dropdown,
    and switching the dropdown changes the active study without reloading."""
    from tests.fixtures.make_synthetic_series import make_synthetic_mr_series

    make_synthetic_ct_series(tmp_path, shape=(3, 4, 4))
    make_synthetic_mr_series(tmp_path, shape=(3, 4, 4))

    win = MainWindow()
    qtbot.addWidget(win)
    assert win.open_folder_path(tmp_path)

    # Both series live in the dropdown.
    assert win.series_combo.count() == 2
    assert win.series_combo.isEnabled()
    initial_uid = win.document.study.series_uid

    # Pick the other series; the document's active study should change.
    other_idx = 1 if win.series_combo.currentIndex() == 0 else 0
    win.series_combo.setCurrentIndex(other_idx)
    assert win.document.study.series_uid != initial_uid


def test_project_remembers_active_series_uid(qtbot, tmp_path):
    """Saving a project then loading it must restore the same active series."""
    from tests.fixtures.make_synthetic_series import make_synthetic_mr_series

    make_synthetic_ct_series(tmp_path, shape=(3, 4, 4))
    make_synthetic_mr_series(tmp_path, shape=(3, 4, 4))
    project_path = tmp_path / f"proj{PROJECT_EXTENSION}"

    win1 = MainWindow()
    qtbot.addWidget(win1)
    assert win1.open_folder_path(tmp_path)
    # Switch to whichever series is "second" in the dropdown.
    second_idx = 1 if win1.series_combo.currentIndex() == 0 else 0
    win1.series_combo.setCurrentIndex(second_idx)
    expected_uid = win1.document.study.series_uid
    save_project(project_path, win1.collect_project())

    win2 = MainWindow()
    qtbot.addWidget(win2)
    assert win2.load_project_from_path(project_path)
    assert win2.document.study.series_uid == expected_uid


def test_open_folder_path_returns_true_on_real_load(qtbot, tmp_path):
    """Regression: QProgressDialog.setValue(100) fired by the worker's final
    progress event used to auto-close + emit canceled(), which the open
    handler treated as a user cancel. The function returned False, the study
    never reached the document, and the views stayed empty. We now disable
    autoClose/autoReset and let result-set win over cancel-set."""
    series_dir = make_synthetic_ct_series(tmp_path, shape=(4, 8, 8))
    win = MainWindow()
    qtbot.addWidget(win)
    assert win.open_folder_path(series_dir) is True
    assert win.document.study is not None
    assert win.document.volume.shape == (4, 8, 8)


def test_mask_library_round_trips_through_project(qtbot, tmp_path):
    """Save a project with two named masks, reload it, verify both masks land
    in the document and the active one becomes the segmentation."""
    import numpy as np

    from dicom_viewer.core.segmentation.threshold import threshold

    series_dir = make_synthetic_ct_series(tmp_path / "scan", shape=(4, 8, 8))
    project_path = tmp_path / f"masks{PROJECT_EXTENSION}"

    win1 = MainWindow()
    qtbot.addWidget(win1)
    assert win1.open_folder_path(series_dir)
    vol = win1.document.volume
    # Two distinct masks.
    win1.document.set_segmentation(threshold(vol, low=-2000, high=10000))
    win1.document.save_mask_as("A")
    win1.document.set_segmentation(threshold(vol, low=-100, high=10000))
    win1.document.save_mask_as("B")
    assert win1.document.active_mask_name == "B"

    win1._save_project_to(project_path)

    # Companion files should have been written next to the project.
    assert (tmp_path / f"masks.A.nii.gz").exists()
    assert (tmp_path / f"masks.B.nii.gz").exists()

    win2 = MainWindow()
    qtbot.addWidget(win2)
    assert win2.load_project_from_path(project_path)
    # Both masks present, B active (it was active at save time).
    assert set(win2.document.mask_names) == {"A", "B"}
    assert win2.document.active_mask_name == "B"
    assert win2.document.segmentation is not None
    # Switching to A produces a different mask (the void-area threshold).
    win2.document.activate_mask("A")
    assert win2.document.active_mask_name == "A"
    assert win2.document.segmentation is not None
    # The two masks have distinct voxel counts (the thresholds were different).
    assert win2.document.get_mask("A").voxel_count != win2.document.get_mask("B").voxel_count


def test_load_corrupt_project_does_not_crash(qtbot, tmp_path, monkeypatch):
    bad = tmp_path / f"bad{PROJECT_EXTENSION}"
    bad.write_text("not even json", encoding="utf-8")
    # Silence the modal warning dialog so the test doesn't block.
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **kw: None))
    win = MainWindow()
    qtbot.addWidget(win)
    assert not win.load_project_from_path(bad)
