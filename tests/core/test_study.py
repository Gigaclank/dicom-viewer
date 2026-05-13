import numpy as np

from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume


def test_study_wraps_volume_and_metadata():
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    volume = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    study = Study(
        volume=volume,
        patient_id="TEST001",
        patient_name="Test^Synthetic",
        study_uid="1.2.3.4",
        series_uid="1.2.3.4.5",
        series_description="synthetic-ct",
        orientation_cosines=(1, 0, 0, 0, 1, 0),
    )
    assert study.modality == "CT"
    assert study.spacing_mm == (1.0, 1.0, 1.0)
    assert study.volume is volume
    assert study.display_name == "TEST001 / synthetic-ct"


def test_study_anonymized_name_when_no_patient_id():
    volume = Volume(
        array=np.zeros((2, 2, 2), dtype=np.int16),
        spacing_mm=(1.0, 1.0, 1.0),
        modality="MR",
    )
    study = Study(
        volume=volume,
        patient_id="",
        patient_name="",
        study_uid="x",
        series_uid="y",
        series_description="anon",
        orientation_cosines=(1, 0, 0, 0, 1, 0),
    )
    assert study.display_name == "<anonymized> / anon"
