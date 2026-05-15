import numpy as np
import pytest

from dicom_viewer.io.dicom_loader import (
    LoaderError,
    load_series_from_file,
    load_series_from_folder,
)
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


def test_load_single_ct_file(tmp_path):
    folder = make_synthetic_ct_series(tmp_path, shape=(4, 6, 6))
    one_file = next(folder.glob("*.dcm"))
    result = load_series_from_file(one_file)
    assert len(result.studies) == 1
    study = result.studies[0]
    assert study.modality == "CT"
    assert study.volume.shape == (1, 6, 6)
    assert study.volume.array.dtype == np.int16


def test_load_single_mr_file(tmp_path):
    folder = make_synthetic_mr_series(tmp_path, shape=(3, 8, 8))
    one_file = next(folder.glob("*.dcm"))
    result = load_series_from_file(one_file)
    assert len(result.studies) == 1
    assert result.studies[0].modality == "MR"
    assert result.studies[0].volume.array.dtype == np.float32


def test_load_single_file_non_dicom_raises(tmp_path):
    bad = tmp_path / "not_dicom.txt"
    bad.write_text("hello")
    with pytest.raises(LoaderError):
        load_series_from_file(bad)


def test_load_single_file_missing_raises(tmp_path):
    with pytest.raises(LoaderError):
        load_series_from_file(tmp_path / "nope.dcm")


def test_load_tolerates_none_slice_thickness(tmp_path):
    """Regression: pydicom returns None for tags that exist but are empty.
    float(None) blew up the loader on the pydicom test corpus."""
    import pydicom

    folder = make_synthetic_ct_series(tmp_path, shape=(3, 4, 4), spacing=(2.0, 1.0, 1.0))
    # Rewrite one slice with SliceThickness explicitly None — pydicom serializes
    # this as an empty-value tag, which deserializes back to None.
    one = next(folder.glob("*.dcm"))
    ds = pydicom.dcmread(one)
    ds.SliceThickness = None
    ds.save_as(one)

    result = load_series_from_folder(folder)
    assert len(result.studies) == 1
    # Z spacing should still be derived from ImagePositionPatient (2 mm).
    assert result.studies[0].spacing_mm[0] == pytest.approx(2.0)


def test_loader_progress_callback_is_invoked(tmp_path):
    folder = make_synthetic_ct_series(tmp_path, shape=(4, 8, 8))
    events: list[tuple[str, float]] = []
    load_series_from_folder(folder, progress=lambda stage, frac: events.append((stage, frac)))
    assert events  # at least one progress callback
    assert events[-1] == ("Done", 1.0)
    # All fractions must be in [0, 1] and the sequence must be non-decreasing.
    for stage, frac in events:
        assert 0.0 <= frac <= 1.0
        assert isinstance(stage, str)
    fractions = [f for _s, f in events]
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))


def test_loader_should_cancel_raises_loader_cancelled(tmp_path):
    from dicom_viewer.io.dicom_loader import LoaderCancelled

    folder = make_synthetic_ct_series(tmp_path, shape=(8, 8, 8))
    with pytest.raises(LoaderCancelled):
        load_series_from_folder(folder, should_cancel=lambda: True)


def test_loader_should_cancel_callback_failure_is_swallowed(tmp_path):
    """If the user-provided should_cancel raises, the loader treats that as
    'not cancelled' rather than failing the load."""
    folder = make_synthetic_ct_series(tmp_path, shape=(3, 4, 4))

    def boom() -> bool:
        raise RuntimeError("intentional")

    result = load_series_from_folder(folder, should_cancel=boom)
    assert len(result.studies) == 1


def test_loader_progress_callback_failure_does_not_break_load(tmp_path):
    folder = make_synthetic_ct_series(tmp_path, shape=(3, 4, 4))

    def boom(_stage: str, _frac: float) -> None:
        raise RuntimeError("intentional")

    result = load_series_from_folder(folder, progress=boom)
    assert len(result.studies) == 1


def test_folder_loads_2d_files_without_image_position_patient(tmp_path):
    """Mammograms and other 2D modalities don't carry ImagePositionPatient.
    The folder loader must still pick them up as standalone one-slice studies
    rather than skipping them as 'incomplete'."""
    import numpy as np
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import DigitalMammographyXRayImageStorageForPresentation
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    folder = tmp_path / "mg"
    folder.mkdir()

    def write_mg(name: str, rows: int, cols: int, laterality: str, view: str) -> None:
        sop_uid = DigitalMammographyXRayImageStorageForPresentation
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = sop_uid
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / name), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = sop_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = generate_uid()
        ds.StudyInstanceUID = generate_uid()
        ds.Modality = "MG"
        ds.PatientID = "P"
        ds.PatientName = "Test"
        ds.Rows, ds.Columns = rows, cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.ImageLaterality = laterality
        ds.ViewPosition = view
        ds.PixelData = np.zeros((rows, cols), dtype=np.uint16).tobytes()
        # NOTE: deliberately no ImagePositionPatient or ImageOrientationPatient.
        ds.save_as(folder / name, enforce_file_format=True)

    write_mg("0250.LEFT_CC.dcm", 64, 32, "L", "CC")
    write_mg("0250.LEFT_MLO.dcm", 64, 32, "L", "MLO")
    write_mg("0250.RIGHT_CC.dcm", 64, 32, "R", "CC")
    write_mg("0250.RIGHT_MLO.dcm", 64, 32, "R", "MLO")

    result = load_series_from_folder(folder)
    assert len(result.studies) == 4
    assert result.skipped_incomplete == 0
    # Display names must be distinct so the user can pick a view.
    names = {s.display_name for s in result.studies}
    assert len(names) == 4
    # All should have stem in the name — filename is the reliable identifier.
    for s in result.studies:
        assert "0250." in s.display_name


def test_folder_scan_uses_thread_pool_for_metadata_read(tmp_path, monkeypatch):
    """The folder header scan is parallelized — regression for the previous
    single-threaded scan that dominated load time on big folders. We assert
    by patching ThreadPoolExecutor and confirming it gets instantiated."""
    from concurrent import futures as _futures

    folder = make_synthetic_ct_series(tmp_path, shape=(8, 4, 4))

    calls: list[int] = []
    orig_executor = _futures.ThreadPoolExecutor

    class _CountingExecutor(orig_executor):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            calls.append(kw.get("max_workers", 0))
            super().__init__(*a, **kw)

    monkeypatch.setattr(
        "dicom_viewer.io.dicom_loader.ThreadPoolExecutor",
        _CountingExecutor,
    )
    result = load_series_from_folder(folder)
    # Two executors fire on a multi-slice load: one for header scan (phase 1),
    # one for pixel decode (phase 2). Both should use >1 worker.
    assert len(calls) >= 1
    assert all(w > 1 for w in calls), calls
    assert len(result.studies) == 1
    assert result.studies[0].volume.shape == (8, 4, 4)


def test_load_folder_skips_a_broken_series_keeps_good_ones(tmp_path):
    """A pathological series shouldn't take down the whole folder load."""
    import pydicom

    good = make_synthetic_ct_series(tmp_path / "good_subdir", shape=(3, 4, 4))
    bad = make_synthetic_mr_series(tmp_path / "bad_subdir", shape=(3, 4, 4))
    # Move both into one combined folder so rglob picks them up together.
    combined = tmp_path / "combined"
    combined.mkdir()
    for src in good.glob("*.dcm"):
        src.rename(combined / src.name)
    for src in bad.glob("*.dcm"):
        new = combined / f"bad_{src.name}"
        src.rename(new)
        # Corrupt the file's Rows tag so assembly will fail.
        ds = pydicom.dcmread(new)
        ds.Rows = None
        ds.save_as(new)

    result = load_series_from_folder(combined)
    assert len(result.studies) == 1
    assert result.studies[0].modality == "CT"
    assert result.skipped_incomplete >= 3
