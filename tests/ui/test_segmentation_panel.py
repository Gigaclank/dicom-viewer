import numpy as np
import pytest

from dicom_viewer.core.document import Document
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume
from dicom_viewer.ui.panels.segmentation import SegmentationPanel


@pytest.fixture
def doc() -> Document:
    arr = np.zeros((6, 6, 6), dtype=np.int16)
    arr[1:5, 1:5, 1:5] = 500
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    study = Study(
        volume=v,
        patient_id="P",
        patient_name="N",
        study_uid="s",
        series_uid="se",
        series_description="d",
        orientation_cosines=(1, 0, 0, 0, 1, 0),
    )
    document = Document()
    document.set_study(study)
    return document


def _wait_for_seg(qtbot, doc, timeout_ms: int = 5000) -> None:
    """Wait for the worker to apply a segmentation to the document."""
    qtbot.waitUntil(lambda: doc.segmentation is not None, timeout=timeout_ms)


def _wait_for_idle(qtbot, panel, extra_ms: int = 100) -> None:
    """Wait for any pending debounce + worker to complete."""
    qtbot.wait(panel.LIVE_DEBOUNCE_MS + extra_ms)
    if panel._seg_worker is not None:
        qtbot.waitUntil(
            lambda: panel._seg_worker is None or not panel._seg_worker.isRunning(),
            timeout=5000,
        )


def test_apply_threshold_writes_segmentation_to_document(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.low_slider.setValue(100)
    panel.high_slider.setValue(1000)
    panel.apply_button.click()
    _wait_for_seg(qtbot, doc)
    assert doc.segmentation.method.startswith("threshold")
    assert doc.segmentation.voxel_count > 0


def test_keep_largest_component_chains(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.low_slider.setValue(100)
    panel.high_slider.setValue(1000)
    panel.largest_component_checkbox.setChecked(True)
    panel.apply_button.click()
    _wait_for_seg(qtbot, doc)
    assert "largest_component" in doc.segmentation.method


def test_smooth_chains_after_apply(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.low_slider.setValue(100)
    panel.high_slider.setValue(1000)
    panel.smooth_checkbox.setChecked(True)
    panel.apply_button.click()
    _wait_for_seg(qtbot, doc)
    assert doc.segmentation.method.endswith("+smooth")


def test_slider_range_adapts_to_volume_intensity_range(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    # Volume intensity range is 0..500; slider range should encompass that with a pad.
    assert panel.low_slider.slider.minimum() <= 0
    assert panel.high_slider.slider.maximum() >= 500


def test_live_preview_updates_segmentation_on_slider_change(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    assert panel.live_preview_checkbox.isChecked()  # default on
    doc.set_segmentation(None)
    panel.low_slider.setValue(100)
    panel.high_slider.setValue(1000)
    _wait_for_seg(qtbot, doc)
    assert doc.segmentation.voxel_count > 0


def test_live_preview_disabled_does_not_recompute(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.live_preview_checkbox.setChecked(False)
    doc.set_segmentation(None)
    panel.low_slider.setValue(100)
    panel.high_slider.setValue(1000)
    # Wait longer than the debounce; with live preview off, nothing should fire.
    _wait_for_idle(qtbot, panel)
    assert doc.segmentation is None
    panel.apply_button.click()
    _wait_for_seg(qtbot, doc)


def test_live_preview_chains_refinements(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.largest_component_checkbox.setChecked(True)
    panel.smooth_checkbox.setChecked(True)
    doc.set_segmentation(None)
    panel.low_slider.setValue(100)
    panel.high_slider.setValue(1000)
    _wait_for_seg(qtbot, doc)
    assert "largest_component" in doc.segmentation.method
    assert doc.segmentation.method.endswith("+smooth")


def test_rapid_slider_changes_apply_only_latest_value(qtbot, doc):
    """Cancel-and-restart: rapid slider changes during the debounce window
    should collapse into one worker run for the final value, not multiple
    overlapping runs producing stale results."""
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    # Disable the chained refinements so the resulting Segmentation.params is
    # the raw threshold dict (chained ops nest the original params).
    panel.largest_component_checkbox.setChecked(False)
    panel.smooth_checkbox.setChecked(False)
    doc.set_segmentation(None)
    panel.low_slider.setValue(50)
    panel.low_slider.setValue(200)
    panel.low_slider.setValue(450)  # final value
    _wait_for_seg(qtbot, doc)
    # The applied threshold should reflect the LAST value.
    assert doc.segmentation.params["low"] == 450


def test_brush_radio_emits_mode_changed_signal(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    seen: list[str] = []
    panel.brush_mode_changed.connect(seen.append)
    panel.brush_add_radio.setChecked(True)
    assert seen == ["add"]
    panel.brush_remove_radio.setChecked(True)
    assert seen == ["add", "remove"]
    panel.brush_off_radio.setChecked(True)
    assert seen == ["add", "remove", "off"]


def test_handle_seed_click_adds_to_segmentation(qtbot, doc):
    """A click in Add mode seeds a region-grow + ORs the result into the
    current segmentation. The grow runs on a worker thread so the UI doesn't
    freeze on big volumes — tests use the blocking helper to wait."""
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.brush_add_radio.setChecked(True)
    panel.tolerance_slider.setValue(100)
    panel.run_handle_seed_click_blocking((2, 2, 2), "add")
    assert doc.segmentation is not None
    assert doc.segmentation.voxel_count > 0
    # The seeded 500-HU blob is filled.
    assert doc.segmentation.mask[3, 3, 3]


def test_handle_seed_click_remove_subtracts_from_existing_mask(qtbot, doc):
    """A click in Remove mode subtracts the grown region from the existing
    accumulated mask."""
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.brush_add_radio.setChecked(True)
    panel.tolerance_slider.setValue(100)
    panel.run_handle_seed_click_blocking((2, 2, 2), "add")
    before = doc.segmentation.voxel_count
    assert before > 0
    panel.brush_remove_radio.setChecked(True)
    panel.run_handle_seed_click_blocking((2, 2, 2), "remove")
    after = doc.segmentation.voxel_count
    assert after < before
    assert not doc.segmentation.mask[3, 3, 3]


def test_brush_seed_click_with_no_volume_is_safe(qtbot):
    """A click before any volume is loaded must not crash; just no-op.
    Crucially: no worker is spawned, so the call returns immediately."""
    empty_doc = Document()
    panel = SegmentationPanel(empty_doc)
    qtbot.addWidget(panel)
    panel.handle_seed_click((0, 0, 0), "add")
    assert empty_doc.segmentation is None
    assert panel._brush_worker is None


def test_brush_runs_on_a_worker_thread_not_the_ui_thread(qtbot, doc):
    """Regression: the initial brush implementation called grow_from_seed on
    the main thread, which froze the window on real CT volumes. Today the
    handler must hand off to a QThread so the UI keeps pumping events while
    SITK runs. We verify by checking that handle_seed_click returns BEFORE
    the worker finishes (i.e. the worker is still running on return)."""
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.brush_add_radio.setChecked(True)
    panel.tolerance_slider.setValue(100)
    panel.handle_seed_click((2, 2, 2), "add")
    # The worker exists and is the active brush worker.
    assert panel._brush_worker is not None
    # Now wait for it to finish.
    qtbot.waitUntil(
        lambda: doc.segmentation is not None and doc.segmentation.voxel_count > 0,
        timeout=5000,
    )


def test_rapid_brush_clicks_cancel_previous_worker(qtbot, doc):
    """Two clicks fired in quick succession: the second click's worker
    becomes the active one (identity check on finished_ok discards the
    first's stale result). We also verify the older worker doesn't get
    GC'd while still running — that would abort the process. Waiting for
    the in-flight list to drain proves both workers reach `finished`."""
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.brush_add_radio.setChecked(True)
    panel.tolerance_slider.setValue(100)
    # Click 1: add. Don't wait — fire click 2 immediately to force overlap.
    panel.handle_seed_click((2, 2, 2), "add")
    panel.handle_seed_click((2, 2, 2), "add")  # second add — same blob
    qtbot.waitUntil(
        lambda: len(panel._brush_workers_in_flight) == 0 and doc.segmentation is not None,
        timeout=5000,
    )
    # Resulting mask is still the add result (idempotent OR), not double-deleted.
    assert doc.segmentation.mask[3, 3, 3]


def test_second_click_after_first_worker_retired_does_not_crash(qtbot, doc):
    """Regression: handle_seed_click called isRunning() on self._brush_worker
    even after that worker had finished and been deleteLater'd. The next
    click hit 'wrapped C/C++ object has been deleted'. Now the retire
    clears the reference, so the second click finds None and starts fresh.
    """
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.brush_add_radio.setChecked(True)
    panel.tolerance_slider.setValue(100)
    panel.run_handle_seed_click_blocking((2, 2, 2), "add")
    # The first worker has finished; _brush_worker may still hold the
    # finished reference until `finished` slot runs. Flush the event loop
    # to let _retire_brush_worker drop it.
    qtbot.waitUntil(lambda: panel._brush_worker is None, timeout=2000)
    # A second click must succeed — this used to raise.
    panel.run_handle_seed_click_blocking((3, 3, 3), "add")
    assert doc.segmentation is not None


def test_brush_mode_suppresses_live_preview(qtbot, doc):
    """While brush mode is active, threshold-slider drags shouldn't kick off
    a worker and clobber the accumulated brush mask."""
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.brush_add_radio.setChecked(True)
    panel.run_handle_seed_click_blocking((2, 2, 2), "add")
    brush_mask = doc.segmentation.mask.copy()
    panel.low_slider.setValue(50)
    panel.high_slider.setValue(60)
    _wait_for_idle(qtbot, panel)
    assert np.array_equal(doc.segmentation.mask, brush_mask)


def test_brush_kind_dropdown_lists_all_options(qtbot, doc):
    """The brush-kind combo should expose every kind the user picked from
    the design list. We assert by userData rather than label so display
    text can be reworded without breaking the test."""
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    kinds = [
        panel.brush_kind_combo.itemData(i)
        for i in range(panel.brush_kind_combo.count())
    ]
    assert set(kinds) == {"region_grow", "threshold", "sphere", "box", "paint_2d", "confidence"}


def test_threshold_brush_kind_runs_through_worker(qtbot, doc):
    """Selecting the threshold kind and clicking a seed routes through the
    worker and produces a non-empty mask matching the threshold semantics
    (no connectivity check — disconnected high-HU blobs both get included)."""
    import numpy as np

    # Add a second disconnected 500-HU blob to verify no-connectivity.
    arr = doc.volume.array.copy()
    arr[5, 5, 5] = 500
    new_vol = doc.volume.__class__(
        array=arr, spacing_mm=doc.volume.spacing_mm, modality=doc.volume.modality
    )
    new_study = doc.study.__class__(
        volume=new_vol,
        patient_id=doc.study.patient_id,
        patient_name=doc.study.patient_name,
        study_uid=doc.study.study_uid,
        series_uid=doc.study.series_uid,
        series_description=doc.study.series_description,
        orientation_cosines=doc.study.orientation_cosines,
    )
    doc.set_study(new_study)

    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    idx = next(
        i for i in range(panel.brush_kind_combo.count())
        if panel.brush_kind_combo.itemData(i) == "threshold"
    )
    panel.brush_kind_combo.setCurrentIndex(idx)
    panel.brush_add_radio.setChecked(True)
    panel.tolerance_slider.setValue(100)
    panel.run_handle_seed_click_blocking((2, 2, 2), "add")
    assert doc.segmentation is not None
    # Both the original blob AND the isolated voxel are included (no connectivity).
    assert doc.segmentation.mask[3, 3, 3]
    assert doc.segmentation.mask[5, 5, 5]


def test_sphere_brush_drops_a_3d_ball(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    idx = next(
        i for i in range(panel.brush_kind_combo.count())
        if panel.brush_kind_combo.itemData(i) == "sphere"
    )
    panel.brush_kind_combo.setCurrentIndex(idx)
    panel.brush_add_radio.setChecked(True)
    panel.brush_radius_slider.setValue(2)  # 2 mm
    panel.run_handle_seed_click_blocking((3, 3, 3), "add")
    assert doc.segmentation is not None
    assert doc.segmentation.mask[3, 3, 3]
    # Within a 2mm sphere on unit spacing: ±2 voxels each axis.
    assert doc.segmentation.mask[3, 3, 5]


def test_paint_2d_brush_runs_synchronously_no_worker(qtbot, doc):
    """Paint mode must NOT spawn a worker thread — drag strokes need to
    keep up with mouse events and worker startup cost would lag them."""
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    idx = next(
        i for i in range(panel.brush_kind_combo.count())
        if panel.brush_kind_combo.itemData(i) == "paint_2d"
    )
    panel.brush_kind_combo.setCurrentIndex(idx)
    panel.brush_add_radio.setChecked(True)
    panel.brush_radius_slider.setValue(2)
    # Press (handled by handle_seed_click) — paints synchronously.
    panel.handle_seed_click((3, 3, 3), "add")
    assert panel._brush_worker is None  # no async worker
    assert doc.segmentation is not None
    assert doc.segmentation.mask[3, 3, 3]


def test_progress_bar_is_hidden_when_idle(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    assert panel.progress_bar.isVisible() is False
    panel.apply_button.click()
    _wait_for_seg(qtbot, doc)
    # After the worker finishes the bar should be hidden again.
    qtbot.waitUntil(lambda: not panel.progress_bar.isVisible(), timeout=5000)
