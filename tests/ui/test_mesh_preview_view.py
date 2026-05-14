"""Smoke + integration tests for the auto-refreshing STL preview tab."""
import numpy as np
import pytest

from dicom_viewer.core.document import Document, WindowingState
from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume
from dicom_viewer.ui.panels.export import ExportPanel
from dicom_viewer.ui.widgets.mesh_preview_view import MeshPreviewView


def _cube_study() -> Study:
    arr = np.zeros((16, 16, 16), dtype=np.int16)
    arr[4:12, 4:12, 4:12] = 1000
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    return Study(
        volume=v,
        patient_id="P",
        patient_name="N",
        study_uid="s",
        series_uid="se",
        series_description="cube",
        orientation_cosines=(1, 0, 0, 0, 1, 0),
    )


def _doc_with_segmentation() -> Document:
    study = _cube_study()
    doc = Document()
    doc.set_study(study)
    doc.set_segmentation(threshold(study.volume, low=500, high=2000))
    return doc


def _doc_without_segmentation() -> Document:
    """A document with a volume but no user-applied segmentation. Window
    center is set so the iso-surface captures the embedded cube."""
    study = _cube_study()
    doc = Document()
    doc.set_study(study)
    doc.set_windowing(WindowingState(center=500.0, width=1000.0))
    return doc


def test_meshpreviewview_instantiates_and_idle(qtbot):
    doc = Document()
    panel = ExportPanel(doc)
    view = MeshPreviewView(doc, panel)
    qtbot.addWidget(panel)
    qtbot.addWidget(view)
    assert view._info_label.text() == "No study loaded"


def test_meshpreviewview_renders_iso_surface_without_segmentation(qtbot):
    """Regression: when no user segmentation exists, the preview should still
    render an iso-surface mesh of the volume — same fallback the STL export
    uses — so 'export 3D view' has something to look at before saving."""
    doc = _doc_without_segmentation()
    panel = ExportPanel(doc)
    view = MeshPreviewView(doc, panel)
    qtbot.addWidget(panel)
    qtbot.addWidget(view)
    view.set_tab_visible(True)
    qtbot.waitUntil(
        lambda: "triangles" in view._info_label.text()
        or "Mesh error" in view._info_label.text(),
        timeout=10000,
    )
    text = view._info_label.text()
    assert "triangles" in text
    assert "iso" in text.lower()


def test_meshpreviewview_renders_when_visible_and_segmentation_present(qtbot):
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    view = MeshPreviewView(doc, panel)
    qtbot.addWidget(panel)
    qtbot.addWidget(view)
    # Simulate tab activation so the view is allowed to do work.
    view.set_tab_visible(True)
    # Run the debounce timeout synchronously: the worker is QThread-based,
    # so we wait for the info label to update.
    qtbot.waitUntil(
        lambda: "triangles" in view._info_label.text() or "Mesh error" in view._info_label.text(),
        timeout=10000,
    )
    assert "triangles" in view._info_label.text()


def test_meshpreviewview_lazy_when_tab_hidden(qtbot):
    """If the tab isn't visible, the worker shouldn't fire on every change."""
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    view = MeshPreviewView(doc, panel)
    qtbot.addWidget(panel)
    qtbot.addWidget(view)
    # Tab is NOT visible.
    assert view._tab_visible is False
    # Settings change while hidden — no worker should be running.
    panel.smoothing_slider.setValue(10)
    panel.smoothing_slider.setValue(20)
    panel.smoothing_slider.setValue(30)
    # Wait a bit longer than the debounce interval; nothing should have run.
    qtbot.wait(view.DEBOUNCE_MS + 100)
    # The info label shouldn't have moved to "Computing…" or a result.
    assert "triangles" not in view._info_label.text()


def test_meshpreviewview_preserves_camera_on_settings_change(qtbot):
    """Refining settings should NOT re-fit the camera — only an explicit
    study change or reset should."""
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    view = MeshPreviewView(doc, panel)
    qtbot.addWidget(panel)
    qtbot.addWidget(view)
    view.set_tab_visible(True)
    qtbot.waitUntil(
        lambda: "triangles" in view._info_label.text(),
        timeout=10000,
    )
    # Mess with the camera the way a user would.
    cam = view._preview._renderer.GetActiveCamera()
    cam.SetViewUp(1.0, 0.0, 0.0)
    cam.Azimuth(40)
    after_user_rotation = (cam.GetViewUp(), cam.GetPosition())

    # Now change a setting — preview must refresh but NOT reset the camera.
    panel.smoothing_slider.setValue(5)
    qtbot.waitUntil(
        lambda: not view._refresh_pending and (
            view._worker is None or not view._worker.isRunning()
        ),
        timeout=10000,
    )
    assert (cam.GetViewUp(), cam.GetPosition()) == after_user_rotation
