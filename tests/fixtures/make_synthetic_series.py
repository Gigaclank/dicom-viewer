"""Generate synthetic DICOM series for tests.

The pixel data is deterministic: a centered cube of high intensity inside
low-intensity background. Use a small `shape` to keep tests fast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, MRImageStorage, generate_uid


def _build_volume(shape: tuple[int, int, int], cube_fraction: float = 0.4) -> np.ndarray:
    """Background = 0, embedded centered cube = 1000. Returns int16 (z, y, x)."""
    z, y, x = shape
    vol = np.zeros(shape, dtype=np.int16)
    cz, cy, cx = z // 2, y // 2, x // 2
    hz = max(int(z * cube_fraction / 2), 1)
    hy = max(int(y * cube_fraction / 2), 1)
    hx = max(int(x * cube_fraction / 2), 1)
    vol[cz - hz : cz + hz, cy - hy : cy + hy, cx - hx : cx + hx] = 1000
    return vol


def _write_slice(
    out_dir: Path,
    index: int,
    pixels: np.ndarray,
    modality: str,
    sop_class_uid: str,
    series_uid: str,
    study_uid: str,
    z_position_mm: float,
    pixel_spacing: tuple[float, float],
) -> Path:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        filename_or_obj=str(out_dir / f"slice_{index:04d}.dcm"),
        dataset={},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = study_uid
    ds.PatientID = "TEST001"
    ds.PatientName = "Test^Synthetic"
    ds.Modality = modality
    ds.InstanceNumber = index + 1
    ds.SeriesNumber = 1
    ds.SeriesDescription = f"synthetic-{modality.lower()}"
    ds.Rows, ds.Columns = pixels.shape
    ds.PixelSpacing = [pixel_spacing[0], pixel_spacing[1]]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0.0, 0.0, z_position_mm]
    ds.SliceThickness = pixel_spacing[0]  # intentionally separate from spacing
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    if modality == "CT":
        ds.RescaleSlope = 1
        ds.RescaleIntercept = -1024
    ds.PixelData = pixels.astype(np.int16).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    path = out_dir / f"slice_{index:04d}.dcm"
    ds.save_as(path, write_like_original=False)
    return path


def make_synthetic_ct_series(
    out_root: Path,
    shape: tuple[int, int, int] = (16, 32, 32),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """Write a CT series to out_root/ct_series and return that directory.

    spacing is (z_mm, y_mm, x_mm). Pixel intensities are stored as raw values;
    after RescaleIntercept = -1024 they become -1024 (background) and -24 (cube).
    """
    return _make_series(out_root, "ct_series", "CT", CTImageStorage, shape, spacing)


def make_synthetic_mr_series(
    out_root: Path,
    shape: tuple[int, int, int] = (16, 32, 32),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """Write an MR series to out_root/mr_series and return that directory."""
    return _make_series(out_root, "mr_series", "MR", MRImageStorage, shape, spacing)


def _make_series(
    out_root: Path,
    subdir: str,
    modality: str,
    sop_class_uid: str,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> Path:
    out_dir = out_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    study_uid = generate_uid()
    volume = _build_volume(shape)
    z_spacing, y_spacing, x_spacing = spacing
    for i in range(shape[0]):
        _write_slice(
            out_dir,
            index=i,
            pixels=volume[i],
            modality=modality,
            sop_class_uid=sop_class_uid,
            series_uid=series_uid,
            study_uid=study_uid,
            z_position_mm=i * z_spacing,
            pixel_spacing=(y_spacing, x_spacing),
        )
    return out_dir
