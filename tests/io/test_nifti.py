"""NIfTI loader + segmentation import/export round-trip."""
import numpy as np
import pytest

from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.volume import Volume
from dicom_viewer.io.dicom_loader import load_series_from_file
from dicom_viewer.io.nifti import (
    is_nifti_path,
    load_segmentation_from_nifti,
    load_study_from_nifti,
    save_segmentation_to_nifti,
)


def test_is_nifti_path():
    from pathlib import Path

    assert is_nifti_path(Path("foo.nii"))
    assert is_nifti_path(Path("foo.nii.gz"))
    assert is_nifti_path(Path("FOO.NII"))
    assert not is_nifti_path(Path("foo.dcm"))


def test_load_nifti_volume_roundtrips_shape_and_spacing(tmp_path):
    import nibabel as nib

    # Author a NIfTI volume in (X, Y, Z) order with known spacing.
    data = np.zeros((8, 16, 32), dtype=np.int16)
    data[4, 8, 16] = 1000
    affine = np.diag([2.0, 1.0, 0.5, 1.0])  # spacing in (x, y, z)
    img = nib.Nifti1Image(data, affine=affine)
    path = tmp_path / "vol.nii.gz"
    nib.save(img, str(path))

    study = load_study_from_nifti(path)
    vol = study.volume
    # (X, Y, Z) -> (Z, Y, X)
    assert vol.shape == (32, 16, 8)
    assert vol.spacing_mm == pytest.approx((0.5, 1.0, 2.0))
    # The hot voxel survives the transpose at the same physical location.
    assert vol.array[16, 8, 4] == 1000


def test_load_series_from_file_dispatches_to_nifti(tmp_path):
    """The single-file entry point picks NIfTI based on extension."""
    import nibabel as nib

    data = np.zeros((4, 4, 4), dtype=np.int16)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(tmp_path / "x.nii.gz"))
    result = load_series_from_file(tmp_path / "x.nii.gz")
    assert len(result.studies) == 1
    assert result.studies[0].volume.shape == (4, 4, 4)


def test_save_and_load_segmentation_nifti_round_trip(tmp_path):
    arr = np.zeros((4, 6, 8), dtype=np.int16)
    vol = Volume(array=arr, spacing_mm=(1.5, 1.0, 0.5), modality="CT")
    mask = np.zeros(arr.shape, dtype=bool)
    mask[1:3, 2:5, 3:6] = True
    seg = Segmentation(mask=mask, method="threshold", params={"low": 100, "high": 200})

    out = tmp_path / "mask.nii.gz"
    save_segmentation_to_nifti(seg, vol, out)
    assert out.exists()

    loaded = load_segmentation_from_nifti(out, vol)
    assert loaded.mask.shape == vol.shape
    assert (loaded.mask == mask).all()
    assert loaded.method.startswith("imported")


def test_load_segmentation_rejects_shape_mismatch(tmp_path):
    import nibabel as nib

    small_vol = Volume(
        array=np.zeros((4, 4, 4), dtype=np.int16), spacing_mm=(1, 1, 1), modality="CT"
    )
    # Write a mismatched mask.
    big_mask = np.ones((8, 8, 8), dtype=np.uint8)
    nib.save(nib.Nifti1Image(big_mask, affine=np.eye(4)), str(tmp_path / "wrong.nii"))

    with pytest.raises(ValueError):
        load_segmentation_from_nifti(tmp_path / "wrong.nii", small_vol)
