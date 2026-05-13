import pydicom

from tests.fixtures.make_synthetic_series import make_synthetic_ct_series


def test_synthetic_ct_series_writes_expected_files(tmp_path):
    out_dir = make_synthetic_ct_series(
        tmp_path, shape=(8, 16, 16), spacing=(2.0, 1.0, 1.0)
    )
    files = sorted(out_dir.glob("*.dcm"))
    assert len(files) == 8

    ds = pydicom.dcmread(files[0])
    assert ds.Modality == "CT"
    assert ds.Rows == 16
    assert ds.Columns == 16
    assert ds.PixelSpacing == [1.0, 1.0]
    assert ds.SeriesInstanceUID == pydicom.dcmread(files[1]).SeriesInstanceUID

    # Adjacent slices should be 2mm apart in z.
    p0 = ds.ImagePositionPatient
    p1 = pydicom.dcmread(files[1]).ImagePositionPatient
    assert abs(float(p1[2]) - float(p0[2]) - 2.0) < 1e-6
