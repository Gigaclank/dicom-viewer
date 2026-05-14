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


def test_set_study_emits_segmentation_so_stale_masks_clear():
    """Regression: switching DCMs left the old segmentation mask cached in
    slice renderers because no 'segmentation' event was emitted, causing an
    IndexError when the new volume had different shape and the user scrolled
    into a slice the old mask didn't cover."""
    doc = Document()
    doc.set_study(_study())
    seg = threshold(doc.volume, low=100, high=1000)
    doc.set_segmentation(seg)
    events: list[str] = []
    doc.subscribe(lambda kind: events.append(kind))
    # Loading a new study must signal observers that the segmentation is gone.
    doc.set_study(_study())
    assert "segmentation" in events
    assert doc.segmentation is None


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


def test_mask_library_save_activate_delete():
    """Document tracks a named mask catalogue distinct from the active mask."""
    doc = Document()
    doc.set_study(_study())
    seg_a = threshold(doc.volume, low=100, high=1000)
    doc.set_segmentation(seg_a)
    doc.save_mask_as("A")
    assert doc.mask_names == ["A"]
    assert doc.active_mask_name == "A"

    seg_b = threshold(doc.volume, low=200, high=2000)
    doc.set_segmentation(seg_b)
    doc.save_mask_as("B")
    assert doc.mask_names == ["A", "B"]
    assert doc.active_mask_name == "B"

    # Switching active swaps the document segmentation.
    doc.activate_mask("A")
    assert doc.segmentation is doc.get_mask("A")
    assert doc.active_mask_name == "A"

    # Deleting the active mask clears segmentation and active name.
    doc.delete_mask("A")
    assert "A" not in doc.mask_names
    assert doc.active_mask_name == ""
    assert doc.segmentation is None


def test_replace_masks_bulk_loads_library():
    doc = Document()
    doc.set_study(_study())
    seg_a = threshold(doc.volume, low=100, high=1000)
    seg_b = threshold(doc.volume, low=200, high=2000)
    events: list[str] = []
    doc.subscribe(events.append)
    doc.replace_masks({"A": seg_a, "B": seg_b}, active_name="B")
    assert set(doc.mask_names) == {"A", "B"}
    assert doc.active_mask_name == "B"
    assert doc.segmentation is seg_b
    assert "mask_library" in events


def test_set_study_clears_mask_library():
    """Loading a new study invalidates the catalogue (mask shapes wouldn't match)."""
    doc = Document()
    doc.set_study(_study())
    doc.set_segmentation(threshold(doc.volume, low=100, high=1000))
    doc.save_mask_as("A")
    assert doc.mask_names == ["A"]
    doc.set_study(_study())
    assert doc.mask_names == []
    assert doc.active_mask_name == ""


def test_unsubscribe():
    doc = Document()
    seen: list[str] = []
    handle = doc.subscribe(lambda kind: seen.append(kind))
    handle()  # unsubscribe
    doc.set_study(_study())
    assert seen == []
