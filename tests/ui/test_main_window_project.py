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

    # Second window: load the project. Settings must come back, but the
    # segmentation must NOT auto-run on load — the user has to hit Apply.
    # This is the explicit contract: avoid spending CPU cycles on a
    # threshold the user may not even want until they ask for it.
    win2 = MainWindow()
    qtbot.addWidget(win2)
    assert win2.load_project_from_path(project_path)
    assert win2.export_panel.smoothing_slider.value() == 7
    assert abs(win2.export_panel.decimation_slider.float_value() - 0.25) < 1e-6
    assert win2.segmentation_panel.low_slider.value() == -500
    assert win2.segmentation_panel.high_slider.value() == 0
    assert win2.document.volume is not None
    # No segmentation until the user clicks Apply (or loads a companion mask).
    assert win2.document.segmentation is None
    # Apply explicitly to confirm the loaded settings can run.
    win2.segmentation_panel.run_apply_blocking()
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


def test_quad_view_contains_all_four_anatomical_panes(qtbot):
    """The central widget is a tab widget whose first tab is the 2×2 quad
    layout. All four anatomical panes (axial, coronal, sagittal, 3D) live
    inside it as descendants — no longer hidden behind tab switches."""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win.view_tabs.count() == 2
    assert win.view_tabs.tabText(0) == "Multi-view"
    assert win.view_tabs.tabText(1) == "STL Preview"
    # All four panes are descendants of the quad widget.
    for pane in (win.axial, win.coronal, win.sagittal, win.volume3d):
        assert win.quad_view.isAncestorOf(pane), pane


def test_changing_windowing_preset_rebuilds_3d_opacity(qtbot, tmp_path):
    """Regression: switching to a non-default windowing preset (e.g. Lung)
    used to leave the 3D pane frozen on the bone iso because set_volume
    built the opacity ramp once and never rebuilt. Today set_windowing
    re-applies the transfer function, so the 3D view tracks the preset."""
    from dicom_viewer.core.document import WindowingState

    series_dir = make_synthetic_ct_series(tmp_path, shape=(6, 8, 8))
    win = MainWindow()
    qtbot.addWidget(win)
    assert win.open_folder_path(series_dir)
    vr = win.volume3d._renderer
    # Probe the opacity TF at the synthetic CT's value (1000 raw → -24 HU
    # after rescale on these fixtures); switching to a different center
    # must change what's opaque there.
    op_default = vr._volume_property.GetScalarOpacity().GetValue(-24.0)
    win.document.set_windowing(WindowingState(center=-600.0, width=1500.0))
    op_lung = vr._volume_property.GetScalarOpacity().GetValue(-24.0)
    assert op_default != pytest.approx(op_lung)


def test_scrubbing_a_2d_pane_updates_the_3d_crosshair(qtbot, tmp_path):
    """Each anatomical pane drives one axis of the 3D crosshair: scrubbing
    axial moves the red plane, etc. Verify by changing the axial scrollbar
    and checking the 3D renderer's recorded voxel."""
    series_dir = make_synthetic_ct_series(tmp_path, shape=(8, 8, 8))
    win = MainWindow()
    qtbot.addWidget(win)
    assert win.open_folder_path(series_dir)
    win.axial.scrollbar.setValue(2)
    win.coronal.scrollbar.setValue(5)
    win.sagittal.scrollbar.setValue(6)
    z, y, x = win.volume3d._renderer.crosshair_voxel
    assert (z, y, x) == (2, 5, 6)


def test_status_bar_starts_idle_and_reflects_status_model(qtbot):
    """The 'Currently doing' label at the bottom of the window mirrors the
    StatusModel. Start idle; flipping a fake task on the model updates the
    label; ending it returns to Idle. This is the contract MainWindow uses
    to surface long ops from the brush, STL preview, folder load, etc."""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._status_label.text() == "Idle"
    win.status_model.begin("fake", "Fake work")
    assert "Fake work" in win._status_label.text()
    win.status_model.end("fake")
    assert win._status_label.text() == "Idle"


def test_status_bar_updates_when_folder_loader_runs(qtbot, tmp_path):
    """Loading a folder must register a task on the status model so the user
    sees 'Currently doing: Loading DICOM folder…' instead of staring at an
    OS-frozen window. We capture the label sequence and assert it transitions
    out-of and back-to Idle."""
    series_dir = make_synthetic_ct_series(tmp_path, shape=(3, 4, 4))
    win = MainWindow()
    qtbot.addWidget(win)
    seen: list[str] = []
    win.status_model.changed.connect(
        lambda: seen.append(win.status_model.render())
    )
    assert win.open_folder_path(series_dir)
    # At some point during the load, render() should have included the label.
    assert any("Loading DICOM folder" in s for s in seen), seen
    # And we end back at idle.
    assert win.status_model.is_idle
    assert win._status_label.text() == "Idle"


def test_rapid_segmentation_changes_collapse_into_one_3d_overlay_rebuild(qtbot, tmp_path):
    """Regression: every segmentation event used to fire volume3d.set_overlay_mask
    synchronously, which runs marching cubes on the full mask. Brush bursts (one
    click ≈ one segmentation event) made the window freeze. The MainWindow now
    debounces 3D-overlay rebuilds with a QTimer; 2D overlays stay immediate. We
    verify by counting set_overlay_mask calls on the 3D view across N rapid
    changes — they should collapse to a single call."""
    import numpy as np

    from dicom_viewer.core.segmentation.base import Segmentation

    series_dir = make_synthetic_ct_series(tmp_path, shape=(4, 8, 8))
    win = MainWindow()
    qtbot.addWidget(win)
    assert win.open_folder_path(series_dir)

    calls: list[object] = []
    original = win.volume3d.set_overlay_mask
    def spy(mask):
        calls.append(mask)
        return original(mask)
    win.volume3d.set_overlay_mask = spy  # type: ignore[assignment]
    # Drop any debounce-triggered call queued by the initial load.
    qtbot.wait(win._overlay_3d_debounce.interval() + 100)
    calls.clear()

    vol = win.document.volume
    mask = np.zeros(vol.shape, dtype=bool)
    mask[1, 1, 1] = True
    # Fire several segmentation events in quick succession.
    for i in range(5):
        m = mask.copy()
        m[1, i % vol.shape[1], 1] = True
        win.document.set_segmentation(Segmentation(mask=m, method="test"))
    # Before the debounce expires, the 3D overlay must not have rebuilt yet.
    assert calls == []
    # After the debounce, exactly one rebuild for the most recent state.
    qtbot.wait(win._overlay_3d_debounce.interval() + 100)
    assert len(calls) == 1


def test_load_corrupt_project_does_not_crash(qtbot, tmp_path, monkeypatch):
    bad = tmp_path / f"bad{PROJECT_EXTENSION}"
    bad.write_text("not even json", encoding="utf-8")
    # Silence the modal warning dialog so the test doesn't block.
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **kw: None))
    win = MainWindow()
    qtbot.addWidget(win)
    assert not win.load_project_from_path(bad)
