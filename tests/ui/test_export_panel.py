import numpy as np
import pytest

from dicom_viewer.core.document import Document
from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume
from dicom_viewer.ui.panels.export import ExportPanel


def _doc_with_segmentation() -> Document:
    arr = np.zeros((16, 16, 16), dtype=np.int16)
    arr[4:12, 4:12, 4:12] = 1000
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    study = Study(
        volume=v,
        patient_id="P",
        patient_name="N",
        study_uid="s",
        series_uid="se",
        series_description="cube",
        orientation_cosines=(1, 0, 0, 0, 1, 0),
    )
    doc = Document()
    doc.set_study(study)
    doc.set_segmentation(threshold(v, low=500, high=2000))
    return doc


def test_export_panel_disabled_without_segmentation(qtbot):
    doc = Document()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    assert not panel.export_button.isEnabled()


def test_export_panel_enabled_with_segmentation(qtbot):
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    assert panel.export_button.isEnabled()


def test_export_writes_stl_file(qtbot, tmp_path):
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    out = tmp_path / "out.stl"
    panel.run_export(out)  # synchronous helper used by the button slot
    assert out.exists()
    assert out.stat().st_size > 84


def test_settings_changed_signal_fires_on_slider_movement(qtbot):
    """The STL Preview tab listens to this signal to auto-refresh."""
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    fires: list[int] = []
    panel.settings_changed.connect(lambda: fires.append(1))

    panel.smoothing_slider.setValue(20)
    panel.decimation_slider.setFloatValue(0.3)
    panel.manifold_checkbox.setChecked(False)

    assert sum(fires) >= 3


def test_export_writes_stl_and_deep_copies_polydata(qtbot, tmp_path):
    """Regression: generate_mesh used to return a polydata still owned by
    the pipeline filters; once that pipeline went out of scope the data
    could end up inconsistent. Now it's deep-copied — verify."""
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    received: list[object] = []
    panel.mesh_ready.connect(received.append)

    out_a = tmp_path / "a.stl"
    out_b = tmp_path / "b.stl"
    panel.run_export(out_a)
    panel.run_export(out_b)

    assert out_a.exists() and out_b.exists()
    assert len(received) == 2
    for mesh in received:
        assert mesh.triangle_count > 0
        assert mesh.polydata.GetNumberOfPolys() == mesh.triangle_count


def test_status_label_shows_filename_not_full_path(qtbot, tmp_path):
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    out = tmp_path / "deeply" / "nested" / "directory" / "result.stl"
    out.parent.mkdir(parents=True)
    panel.run_export(out)
    text = panel._status.text()
    assert "result.stl" in text
    assert str(out.parent) not in text  # full path NOT in label
    assert panel._status.toolTip() == str(out)  # full path on hover instead
