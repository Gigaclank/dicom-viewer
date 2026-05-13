import numpy as np

from dicom_viewer.core.document import Document, WindowingState
from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume


def _study() -> Study:
    arr = np.zeros((8, 8, 8), dtype=np.int16)
    arr[2:6, 2:6, 2:6] = 500
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    return Study(
        volume=v,
        patient_id="P1",
        patient_name="X",
        study_uid="s",
        series_uid="ser",
        series_description="test",
        orientation_cosines=(1, 0, 0, 0, 1, 0),
    )


def test_set_study_notifies_observers():
    events: list[str] = []
    doc = Document()
    doc.subscribe(lambda kind: events.append(kind))
    doc.set_study(_study())
    assert "study" in events
    assert doc.study is not None
    assert doc.volume is not None


def test_set_segmentation_notifies():
    doc = Document()
    doc.set_study(_study())
    events: list[str] = []
    doc.subscribe(lambda kind: events.append(kind))
    seg = threshold(doc.volume, low=100, high=1000)
    doc.set_segmentation(seg)
    assert "segmentation" in events
    assert doc.segmentation is seg


def test_set_region_notifies_and_clamps():
    doc = Document()
    doc.set_study(_study())
    events: list[str] = []
    doc.subscribe(lambda kind: events.append(kind))
    doc.set_region(Region(z=(-5, 100), y=(0, 4), x=(0, 4)))
    assert "region" in events
    # Clamped to volume bbox.
    assert doc.region == Region(z=(0, 8), y=(0, 4), x=(0, 4))


def test_windowing_defaults_and_update():
    doc = Document()
    doc.set_study(_study())
    assert doc.windowing.width > 0
    events: list[str] = []
    doc.subscribe(lambda kind: events.append(kind))
    doc.set_windowing(WindowingState(center=400, width=1500))
    assert "windowing" in events
    assert doc.windowing.center == 400


def test_unsubscribe():
    doc = Document()
    seen: list[str] = []
    handle = doc.subscribe(lambda kind: seen.append(kind))
    handle()  # unsubscribe
    doc.set_study(_study())
    assert seen == []
