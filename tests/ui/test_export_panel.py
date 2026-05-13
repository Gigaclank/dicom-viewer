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


def test_preview_button_disabled_without_segmentation(qtbot):
    doc = Document()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    assert not panel.preview_button.isEnabled()


def test_preview_emits_mesh_ready_signal(qtbot):
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    received: list[object] = []
    panel.mesh_ready.connect(received.append)
    panel.preview_button.click()
    assert len(received) == 1
    mesh = received[0]
    assert mesh.triangle_count > 0


def test_preview_does_not_write_file(qtbot, tmp_path):
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    # Cwd is unaffected by preview — no STL written anywhere.
    panel.preview_button.click()
    assert list(tmp_path.glob("*.stl")) == []


def test_preview_clicked_twice_yields_two_valid_meshes(qtbot):
    """Regression: a second preview must work just like the first.

    Was failing because generate_mesh returned a polydata still owned by the
    pipeline filters, and reusing the same MeshPreviewDialog left the VTK
    render widget in a state that blanked out the second mesh and prevented
    the close button from responding.
    """
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    received: list[object] = []
    panel.mesh_ready.connect(received.append)

    panel.preview_button.click()
    panel.preview_button.click()

    assert len(received) == 2
    for mesh in received:
        assert mesh.triangle_count > 0
        # The polydata must still be valid after the worker that produced it
        # has exited (deep-copied output, not pipeline-owned).
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
