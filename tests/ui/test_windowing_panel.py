import numpy as np
import pytest

from dicom_viewer.core.document import Document
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume
from dicom_viewer.ui.panels.windowing import WindowingPanel


@pytest.fixture
def doc_ct() -> Document:
    arr = np.zeros((4, 4, 4), dtype=np.int16)
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
    doc = Document()
    doc.set_study(study)
    return doc


def test_windowing_panel_ct_presets_present(qtbot, doc_ct):
    panel = WindowingPanel(doc_ct)
    qtbot.addWidget(panel)
    items = [panel.preset_combo.itemText(i) for i in range(panel.preset_combo.count())]
    assert "Bone" in items
    assert "Soft Tissue" in items
    assert "Lung" in items
    assert "Brain" in items


def test_windowing_preset_updates_document(qtbot, doc_ct):
    panel = WindowingPanel(doc_ct)
    qtbot.addWidget(panel)
    panel.apply_preset("Bone")
    assert doc_ct.windowing.center == 400
    assert doc_ct.windowing.width == 1500


def test_windowing_sliders_drive_document(qtbot, doc_ct):
    panel = WindowingPanel(doc_ct)
    qtbot.addWidget(panel)
    panel.center_slider.setValue(50)
    panel.width_slider.setValue(700)
    assert doc_ct.windowing.center == 50
    assert doc_ct.windowing.width == 700
