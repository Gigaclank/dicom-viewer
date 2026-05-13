"""DICOM folder loader.

Walks a directory tree, groups DICOM files by SeriesInstanceUID, sorts each
series by ImagePositionPatient projected onto the slice-normal axis, and
returns a list of fully assembled `Study` objects.
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


class _IncompleteSeries(Exception):
    pass


def _assemble_series(datasets: list[pydicom.Dataset]) -> Study:
    if not datasets:
        raise _IncompleteSeries()

    sample = datasets[0]
    iop = [float(v) for v in sample.ImageOrientationPatient]
    row_cosine = np.array(iop[0:3], dtype=np.float64)
    col_cosine = np.array(iop[3:6], dtype=np.float64)
    slice_normal = np.cross(row_cosine, col_cosine)

    def project(ds: pydicom.Dataset) -> float:
        ipp = np.array([float(v) for v in ds.ImagePositionPatient], dtype=np.float64)
        return float(np.dot(ipp, slice_normal))

    datasets.sort(key=project)

    rows, cols = int(sample.Rows), int(sample.Columns)
    n = len(datasets)

    modality = str(sample.Modality)
    if modality == "CT":
        out = np.empty((n, rows, cols), dtype=np.int16)
        slope = float(getattr(sample, "RescaleSlope", 1.0))
        intercept = float(getattr(sample, "RescaleIntercept", 0.0))
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
    else:
        z_spacing = float(getattr(sample, "SliceThickness", 1.0))
    py, px = (float(v) for v in sample.PixelSpacing)
    spacing = (z_spacing, py, px)

    volume = Volume(array=out, spacing_mm=spacing, modality=modality)

    return Study(
        volume=volume,
        patient_id=str(getattr(sample, "PatientID", "")),
        patient_name=str(getattr(sample, "PatientName", "")),
        study_uid=str(sample.StudyInstanceUID),
        series_uid=str(sample.SeriesInstanceUID),
        series_description=str(getattr(sample, "SeriesDescription", "")),
        orientation_cosines=(iop[0], iop[1], iop[2], iop[3], iop[4], iop[5]),
    )
