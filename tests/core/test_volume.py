import numpy as np
import pytest

from dicom_viewer.core.region import Region
from dicom_viewer.core.volume import Orientation, Volume


def _cube_volume() -> Volume:
    arr = np.zeros((10, 10, 10), dtype=np.int16)
    arr[3:7, 3:7, 3:7] = 1000
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_volume_shape_and_bbox():
    v = _cube_volume()
    assert v.shape == (10, 10, 10)
    assert v.bbox() == Region(z=(0, 10), y=(0, 10), x=(0, 10))


def test_volume_slice_axial():
    v = _cube_volume()
    s = v.slice(Orientation.AXIAL, 5)
    assert s.shape == (10, 10)
    assert s[5, 5] == 1000
    assert s[0, 0] == 0


def test_volume_slice_coronal():
    v = _cube_volume()
    s = v.slice(Orientation.CORONAL, 5)
    assert s.shape == (10, 10)
    assert s[5, 5] == 1000


def test_volume_slice_sagittal():
    v = _cube_volume()
    s = v.slice(Orientation.SAGITTAL, 5)
    assert s.shape == (10, 10)
    assert s[5, 5] == 1000


def test_volume_slice_out_of_range():
    v = _cube_volume()
    with pytest.raises(IndexError):
        v.slice(Orientation.AXIAL, 999)


def test_volume_windowed_uint8():
    v = _cube_volume()
    s = v.windowed(Orientation.AXIAL, 5, center=500, width=1000)
    assert s.dtype == np.uint8
    assert s[5, 5] == 255  # 1000 maps to top of window
    assert s[0, 0] == 0     # 0 maps to bottom


def test_volume_crop_is_a_view():
    v = _cube_volume()
    region = Region(z=(3, 7), y=(3, 7), x=(3, 7))
    cropped = v.crop(region)
    assert cropped.shape == (4, 4, 4)
    assert int(cropped.array.min()) == 1000
    assert int(cropped.array.max()) == 1000


def test_volume_intensity_range():
    v = _cube_volume()
    lo, hi = v.intensity_range()
    assert lo == 0
    assert hi == 1000


def test_volume_intensity_percentiles_for_mri_presets():
    arr = np.random.default_rng(0).integers(0, 4096, size=(8, 8, 8), dtype=np.int16)
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="MR")
    lo, hi = v.intensity_percentiles(1, 99)
    assert lo < hi
    assert lo >= 0 and hi <= 4096
