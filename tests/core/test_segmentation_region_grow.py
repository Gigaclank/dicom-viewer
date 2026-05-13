import numpy as np

from dicom_viewer.core.segmentation.region_grow import region_grow
from dicom_viewer.core.volume import Volume


def test_region_grow_fills_connected_region():
    arr = np.zeros((10, 10, 10), dtype=np.int16)
    arr[3:7, 3:7, 3:7] = 500     # foreground blob
    arr[0, 0, 0] = 500           # isolated voxel at far corner
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    seg = region_grow(v, seed=(5, 5, 5), tolerance=10)
    # The connected blob is selected.
    assert seg.mask[3, 3, 3]
    assert seg.mask[6, 6, 6]
    # Isolated voxel is NOT (not connected).
    assert not seg.mask[0, 0, 0]
    # Background isn't.
    assert not seg.mask[8, 8, 8]
    assert seg.method == "region_grow"
    assert seg.params == {"seed": (5, 5, 5), "tolerance": 10}


def test_region_grow_seed_out_of_bounds():
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    import pytest
    with pytest.raises(ValueError):
        region_grow(v, seed=(99, 0, 0), tolerance=10)
