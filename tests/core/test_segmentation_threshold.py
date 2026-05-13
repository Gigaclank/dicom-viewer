import numpy as np

from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.volume import Volume


def _cube_volume() -> Volume:
    arr = np.zeros((8, 8, 8), dtype=np.int16)
    arr[2:6, 2:6, 2:6] = 500
    arr[0, 0, 0] = 9999  # an outlier voxel
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_threshold_inclusive_low_high():
    v = _cube_volume()
    seg = threshold(v, low=100, high=1000)
    assert seg.mask.shape == v.shape
    assert seg.mask.dtype == bool
    # All cube voxels selected.
    assert seg.mask[3, 3, 3]
    # Background not selected.
    assert not seg.mask[0, 1, 0]
    # Outlier above high is excluded.
    assert not seg.mask[0, 0, 0]


def test_threshold_records_provenance():
    v = _cube_volume()
    seg = threshold(v, low=100, high=1000)
    assert seg.method == "threshold"
    assert seg.params == {"low": 100, "high": 1000}


def test_threshold_handles_low_equals_high():
    v = _cube_volume()
    seg = threshold(v, low=500, high=500)
    assert int(seg.mask.sum()) == int((v.array == 500).sum())
