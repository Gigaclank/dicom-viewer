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


def test_progress_bar_is_hidden_when_idle(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    assert panel.progress_bar.isVisible() is False
    panel.apply_button.click()
    _wait_for_seg(qtbot, doc)
    # After the worker finishes the bar should be hidden again.
    qtbot.waitUntil(lambda: not panel.progress_bar.isVisible(), timeout=5000)
