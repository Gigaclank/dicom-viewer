import numpy as np
import pytest

from dicom_viewer.io.dicom_loader import LoaderError, load_series_from_folder
from tests.fixtures.make_synthetic_series import (
    make_synthetic_ct_series,
    make_synthetic_mr_series,
)


def test_load_ct_series(tmp_path):
    folder = make_synthetic_ct_series(
        tmp_path, shape=(6, 8, 8), spacing=(2.0, 1.0, 1.0)
    )
    result = load_series_from_folder(folder)
    assert len(result.studies) == 1
    study = result.studies[0]
    assert study.modality == "CT"
    assert study.spacing_mm == pytest.approx((2.0, 1.0, 1.0))
    # CT was written with raw pixel values 0/1000; rescale -1024 => -1024 / -24.
    assert study.volume.array.min() == -1024
    assert study.volume.array.max() == -24
    assert study.volume.array.dtype == np.int16


def test_load_mr_series(tmp_path):
    folder = make_synthetic_mr_series(tmp_path, shape=(4, 4, 4))
    result = load_series_from_folder(folder)
    assert len(result.studies) == 1
    assert result.studies[0].modality == "MR"
    assert result.studies[0].volume.array.dtype == np.float32


def test_load_skips_non_dicom_files(tmp_path):
    folder = make_synthetic_ct_series(tmp_path, shape=(3, 4, 4))
    (folder / "README.txt").write_text("hello")
    (folder / "junk.bin").write_bytes(b"\x00\x01\x02")
    result = load_series_from_folder(folder)
    assert len(result.studies) == 1
    assert result.skipped_non_dicom == 2


def test_load_multiple_series_returns_all(tmp_path):
    make_synthetic_ct_series(tmp_path, shape=(3, 4, 4))
    make_synthetic_mr_series(tmp_path, shape=(3, 4, 4))
    result = load_series_from_folder(tmp_path)
    assert len(result.studies) == 2
    modalities = sorted(s.modality for s in result.studies)
    assert modalities == ["CT", "MR"]


def test_load_empty_folder_raises(tmp_path):
    with pytest.raises(LoaderError):
        load_series_from_folder(tmp_path)


def test_slice_sorting_uses_image_position(tmp_path):
    folder = make_synthetic_ct_series(
        tmp_path, shape=(5, 4, 4), spacing=(3.0, 1.0, 1.0)
    )
    # Shuffle filenames to verify InstanceNumber isn't relied on.
    files = sorted(folder.glob("*.dcm"))
    renamed = []
    for i, f in enumerate(files):
        new = folder / f"x_{(i * 37) % 5:02d}_{f.name}"
        f.rename(new)
        renamed.append(new)
    result = load_series_from_folder(folder)
    study = result.studies[0]
    # Spacing recomputed from positions, not SliceThickness.
    assert study.spacing_mm[0] == pytest.approx(3.0)
