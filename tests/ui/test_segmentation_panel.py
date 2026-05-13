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


def test_apply_threshold_writes_segmentation_to_document(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.low_spin.setValue(100)
    panel.high_spin.setValue(1000)
    panel.apply_button.click()
    assert doc.segmentation is not None
    assert doc.segmentation.method.startswith("threshold")
    assert doc.segmentation.voxel_count > 0


def test_keep_largest_component_chains(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.low_spin.setValue(100)
    panel.high_spin.setValue(1000)
    panel.largest_component_checkbox.setChecked(True)
    panel.apply_button.click()
    assert "largest_component" in doc.segmentation.method


def test_smooth_chains_after_apply(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.low_spin.setValue(100)
    panel.high_spin.setValue(1000)
    panel.smooth_checkbox.setChecked(True)
    panel.apply_button.click()
    assert doc.segmentation.method.endswith("+smooth")
