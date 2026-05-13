"""DICOM loader — folder (multi-slice series) and single-file entry points.

`load_series_from_folder` walks a directory tree, groups files by
SeriesInstanceUID, sorts each series by ImagePositionPatient projected onto
the slice-normal axis. `load_series_from_file` reads a single DICOM file as
a one-slice study (or multi-frame volume), tolerating missing geometry tags
that one-off `.dcm` files often skip.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume


class LoaderError(Exception):
    """Raised when a folder yields no loadable DICOM series."""


@dataclass(frozen=True)
class LoadResult:
    studies: list[Study]
    skipped_non_dicom: int
    skipped_incomplete: int


def load_series_from_folder(folder: Path) -> LoadResult:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise LoaderError(f"not a directory: {folder}")

    groups: dict[str, list[pydicom.Dataset]] = defaultdict(list)
    skipped_non_dicom = 0
    skipped_incomplete = 0

    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, defer_size="1 KB", force=False)
        except (InvalidDicomError, OSError):
            skipped_non_dicom += 1
            continue
        try:
            uid = str(ds.SeriesInstanceUID)
            _ = ds.PixelData  # ensure present
            _ = ds.ImagePositionPatient
        except AttributeError:
            skipped_incomplete += 1
            continue
        groups[uid].append(ds)

    studies: list[Study] = []
    for uid, datasets in groups.items():
        try:
            study = _assemble_series(datasets)
        except _IncompleteSeries:
            skipped_incomplete += len(datasets)
            continue
        except Exception:
            # One pathological series (decoding failure, mismatched frames,
            # etc.) must not abort loading the rest of the folder.
            skipped_incomplete += len(datasets)
            continue
        studies.append(study)

    if not studies:
        raise LoaderError(
            f"no loadable DICOM series in {folder} "
            f"(skipped {skipped_non_dicom} non-DICOM, {skipped_incomplete} incomplete)"
        )

    studies.sort(key=lambda s: (s.modality, s.series_description))
    return LoadResult(
        studies=studies,
        skipped_non_dicom=skipped_non_dicom,
        skipped_incomplete=skipped_incomplete,
    )


def load_series_from_file(path: Path) -> LoadResult:
    """Load a single DICOM file as a one-slice (or multi-frame) study.

    More permissive than folder loading: tolerates missing ImageOrientationPatient,
    PixelSpacing, and SliceThickness by substituting standard-axial defaults.
    Use this for one-off `.dcm`/`.dicom` files that lack full geometry headers.
    """
    path = Path(path)
    if not path.is_file():
        raise LoaderError(f"not a file: {path}")
    try:
        ds = pydicom.dcmread(path, force=False)
    except (InvalidDicomError, OSError) as e:
        raise LoaderError(f"not a valid DICOM file: {path} ({e})") from e
    if not hasattr(ds, "PixelData"):
        raise LoaderError(f"no PixelData in {path}")
    try:
        study = _assemble_single(ds)
    except Exception as e:
        raise LoaderError(f"could not assemble study from {path}: {e}") from e
    return LoadResult(studies=[study], skipped_non_dicom=0, skipped_incomplete=0)


def _assemble_single(ds: pydicom.Dataset) -> Study:
    """Build a Study from one dataset, falling back to defaults for missing geometry."""
    pixels = ds.pixel_array
    modality = str(getattr(ds, "Modality", "OT"))

    if modality == "CT":
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        scaled = (pixels.astype(np.float32) * slope + intercept).astype(np.int16)
    else:
        scaled = pixels.astype(np.float32)

    # Normalize to (z, y, x). 2D single slice -> (1, y, x); 3D multi-frame already OK.
    if scaled.ndim == 2:
        scaled = scaled[np.newaxis, :, :]
    elif scaled.ndim != 3:
        raise ValueError(f"unsupported pixel array shape {scaled.shape}")

    iop_default = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    iop_raw = getattr(ds, "ImageOrientationPatient", None)
    if iop_raw is not None and len(iop_raw) == 6:
        iop = tuple(float(v) for v in iop_raw)
    else:
        iop = iop_default

    pixel_spacing = getattr(ds, "PixelSpacing", None)
    if pixel_spacing is not None and len(pixel_spacing) == 2:
        py, px = float(pixel_spacing[0]), float(pixel_spacing[1])
    else:
        py, px = 1.0, 1.0

    z_spacing = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
    if z_spacing <= 0:
        z_spacing = 1.0

    volume = Volume(
        array=np.ascontiguousarray(scaled),
        spacing_mm=(z_spacing, py, px),
        modality=modality,
    )
    return Study(
        volume=volume,
        patient_id=str(getattr(ds, "PatientID", "")),
        patient_name=str(getattr(ds, "PatientName", "")),
        study_uid=str(getattr(ds, "StudyInstanceUID", "")),
        series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
        series_description=str(getattr(ds, "SeriesDescription", "")),
        orientation_cosines=iop,
    )


class _IncompleteSeries(Exception):
    pass


def _safe_float(value: object, default: float) -> float:
    """Coerce a pydicom tag value to float, falling back when it's None/empty/garbage."""
    if value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _assemble_series(datasets: list[pydicom.Dataset]) -> Study:
    if not datasets:
        raise _IncompleteSeries()

    sample = datasets[0]
    iop_raw = getattr(sample, "ImageOrientationPatient", None)
    if iop_raw is None or len(iop_raw) != 6:
        iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    else:
        iop = [_safe_float(v, 0.0) for v in iop_raw]
    row_cosine = np.array(iop[0:3], dtype=np.float64)
    col_cosine = np.array(iop[3:6], dtype=np.float64)
    slice_normal = np.cross(row_cosine, col_cosine)

    def project(ds: pydicom.Dataset) -> float:
        ipp_raw = getattr(ds, "ImagePositionPatient", None)
        if ipp_raw is None or len(ipp_raw) != 3:
            return 0.0
        ipp = np.array([_safe_float(v, 0.0) for v in ipp_raw], dtype=np.float64)
        return float(np.dot(ipp, slice_normal))

    datasets.sort(key=project)

    rows, cols = int(sample.Rows), int(sample.Columns)
    n = len(datasets)

    modality = str(getattr(sample, "Modality", "OT"))
    if modality == "CT":
        out = np.empty((n, rows, cols), dtype=np.int16)
        slope = _safe_float(getattr(sample, "RescaleSlope", None), 1.0)
        intercept = _safe_float(getattr(sample, "RescaleIntercept", None), 0.0)
    else:
        out = np.empty((n, rows, cols), dtype=np.float32)
        slope = 1.0
        intercept = 0.0

    for i, ds in enumerate(datasets):
        pixels = ds.pixel_array
        if modality == "CT":
            scaled = (pixels.astype(np.float32) * slope + intercept).astype(np.int16)
            out[i] = scaled
        else:
            out[i] = pixels.astype(np.float32)

    # Spacing: recompute z from projected positions, y/x from PixelSpacing.
    if n >= 2:
        z_spacing = abs(project(datasets[1]) - project(datasets[0]))
        if z_spacing == 0.0:
            z_spacing = _safe_float(getattr(sample, "SliceThickness", None), 1.0)
    else:
        z_spacing = _safe_float(getattr(sample, "SliceThickness", None), 1.0)
    if z_spacing <= 0.0:
        z_spacing = 1.0

    px_spacing = getattr(sample, "PixelSpacing", None)
    if px_spacing is not None and len(px_spacing) == 2:
        py = _safe_float(px_spacing[0], 1.0)
        px = _safe_float(px_spacing[1], 1.0)
    else:
        py, px = 1.0, 1.0
    spacing = (z_spacing, py, px)

    volume = Volume(array=out, spacing_mm=spacing, modality=modality)

    return Study(
        volume=volume,
        patient_id=str(getattr(sample, "PatientID", "") or ""),
        patient_name=str(getattr(sample, "PatientName", "") or ""),
        study_uid=str(getattr(sample, "StudyInstanceUID", "") or ""),
        series_uid=str(getattr(sample, "SeriesInstanceUID", "") or ""),
        series_description=str(getattr(sample, "SeriesDescription", "") or ""),
        orientation_cosines=(iop[0], iop[1], iop[2], iop[3], iop[4], iop[5]),
    )
