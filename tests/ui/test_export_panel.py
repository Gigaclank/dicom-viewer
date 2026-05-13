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
