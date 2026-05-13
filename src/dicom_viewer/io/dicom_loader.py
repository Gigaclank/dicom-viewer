"""DICOM loader — folder (multi-slice series) and single-file entry points.

`load_series_from_folder` walks a directory tree, groups files by
SeriesInstanceUID, sorts each series by ImagePositionPatient projected onto
the slice-normal axis. `load_series_from_file` reads a single DICOM file as
a one-slice study (or multi-frame volume), tolerating missing geometry tags
that one-off `.dcm` files often skip.
"""
from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume

ProgressCallback = Callable[[str, float], None]

# Cap parallel workers so we don't open hundreds of file handles at once on
# very large folders. 8 is enough to hide I/O latency and saturate the JPEG
# decoders on a typical laptop.
_MAX_DECODE_WORKERS = min(8, max(1, (os.cpu_count() or 4)))


class LoaderError(Exception):
    """Raised when a folder yields no loadable DICOM series."""


class LoaderCancelled(Exception):
    """Raised when the loader is interrupted via the should_cancel callback."""


@dataclass(frozen=True)
class LoadResult:
    studies: list[Study]
    skipped_non_dicom: int
    skipped_incomplete: int


def load_series_from_folder(
    folder: Path,
    *,
    progress: Optional[ProgressCallback] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> LoadResult:
    """Walk a folder, group DICOM files by SeriesInstanceUID, assemble each
    series into a Study.

    Pixel-array decoding (the slow part on big series) runs in a thread pool —
    pydicom releases the GIL inside its C decoders so parallel I/O + JPEG
    decode is a real speedup, typically 3-5x on a 500-image folder.

    `progress` (optional) receives (stage, fraction-in-0..1) at each phase.
    """
    def report(stage: str, frac: float) -> None:
        if progress is not None:
            try:
                progress(stage, frac)
            except Exception:
                pass

    def check_cancel() -> None:
        if should_cancel is not None:
            try:
                if should_cancel():
                    raise LoaderCancelled()
            except LoaderCancelled:
                raise
            except Exception:
                # If the user-provided callback itself blows up, treat it as
                # 'not cancelled' rather than failing the load.
                return

    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise LoaderError(f"not a directory: {folder}")

    # --- phase 1: scan files, group by series UID -------------------------
    report("Scanning folder", 0.0)
    all_files = [p for p in folder.rglob("*") if p.is_file()]
    total_files = max(1, len(all_files))

    # 3D series (slices share an IPP-based stacking axis) go into `groups`.
    # 2D images (no IPP — mammograms, plain X-rays, single MR localizers...)
    # go into `singletons` and become one-slice studies each.
    groups: dict[str, list[pydicom.Dataset]] = defaultdict(list)
    singletons: list[pydicom.Dataset] = []
    skipped_non_dicom = 0
    skipped_incomplete = 0

    for i, path in enumerate(all_files):
        if i % 25 == 0:
            check_cancel()
        try:
            ds = pydicom.dcmread(path, defer_size="1 KB", force=False)
        except (InvalidDicomError, OSError):
            skipped_non_dicom += 1
            continue
        if not hasattr(ds, "PixelData"):
            skipped_incomplete += 1
            continue
        if hasattr(ds, "ImagePositionPatient"):
            uid = str(getattr(ds, "SeriesInstanceUID", "") or path.stem)
            groups[uid].append(ds)
        else:
            singletons.append(ds)
        # Report progress roughly every 25 files (and at the end).
        if i % 25 == 0 or i == total_files - 1:
            report(f"Scanning ({i + 1}/{total_files})", 0.10 * (i + 1) / total_files)

    # --- phase 2: assemble each series ------------------------------------
    studies: list[Study] = []
    # Total assembly units = stacked series + standalone 2D files.
    n_units = max(1, len(groups) + len(singletons))
    unit_idx = 0
    for uid, datasets in groups.items():
        check_cancel()
        base = 0.10 + 0.85 * (unit_idx / n_units)
        slice_width = 0.85 / n_units

        def per_series_progress(stage: str, frac: float, _base=base, _w=slice_width) -> None:
            report(stage, _base + _w * frac)

        try:
            study = _assemble_series(
                datasets, progress=per_series_progress, should_cancel=should_cancel
            )
        except LoaderCancelled:
            raise
        except _IncompleteSeries:
            skipped_incomplete += len(datasets)
        except Exception:
            # One pathological series (decoding failure, mismatched frames,
            # etc.) must not abort loading the rest of the folder.
            skipped_incomplete += len(datasets)
        else:
            studies.append(study)
        unit_idx += 1

    for ds in singletons:
        check_cancel()
        base = 0.10 + 0.85 * (unit_idx / n_units)
        try:
            study = _assemble_single(ds)
        except Exception:
            skipped_incomplete += 1
        else:
            studies.append(study)
        report("Reading 2D image", base + 0.85 / n_units)
        unit_idx += 1

    if not studies:
        raise LoaderError(
            f"no loadable DICOM series in {folder} "
            f"(skipped {skipped_non_dicom} non-DICOM, {skipped_incomplete} incomplete)"
        )

    studies.sort(key=lambda s: (s.modality, s.series_description))
    report("Done", 1.0)
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
    # For 2D modalities (mammography, plain X-rays, …) the SeriesDescription
    # is often the same across every view in the folder, so the user can't
    # tell views apart in the dropdown. We always include the source filename
    # stem (the most reliable distinguishing label for ad-hoc DICOM dumps),
    # and append laterality / view tags when they're present and meaningful.
    description = str(getattr(ds, "SeriesDescription", "") or "").strip()
    fname = str(getattr(ds, "filename", "") or "")
    stem = Path(fname).stem if fname else ""
    if stem:
        description = f"{description} — {stem}" if description else stem
    laterality = str(getattr(ds, "ImageLaterality", "") or "").strip()
    view = str(getattr(ds, "ViewPosition", "") or "").strip()
    tag_bits = [b for b in (laterality, view) if b]
    if tag_bits:
        description = f"{description} [{' '.join(tag_bits)}]"
    return Study(
        volume=volume,
        patient_id=str(getattr(ds, "PatientID", "")),
        patient_name=str(getattr(ds, "PatientName", "")),
        study_uid=str(getattr(ds, "StudyInstanceUID", "")),
        series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
        series_description=description,
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


def _assemble_series(
    datasets: list[pydicom.Dataset],
    *,
    progress: Optional[ProgressCallback] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Study:
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

    # Pixel decoding is the slow part on big series. pydicom's pixel_array
    # accessor reads the (deferred) PixelData from disk and decodes it; most
    # decoders are C extensions that release the GIL, so a thread pool gives
    # a real speedup on multi-slice series. Workers stay small enough not to
    # exhaust file handles on huge folders.
    n_workers = min(_MAX_DECODE_WORKERS, n)

    def decode_one(i: int) -> tuple[int, np.ndarray]:
        return i, datasets[i].pixel_array

    def _cancelled() -> bool:
        if should_cancel is None:
            return False
        try:
            return bool(should_cancel())
        except Exception:
            return False

    if n_workers <= 1:
        # Skip the executor overhead for tiny series — common in tests.
        completed = 0
        for i in range(n):
            if _cancelled():
                raise LoaderCancelled()
            _, pixels = decode_one(i)
            if modality == "CT":
                out[i] = (pixels.astype(np.float32) * slope + intercept).astype(np.int16)
            else:
                out[i] = pixels.astype(np.float32)
            completed += 1
            if progress is not None and (completed % 25 == 0 or completed == n):
                try:
                    progress(f"Decoding slices ({completed}/{n})", completed / n)
                except Exception:
                    pass
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(decode_one, i): i for i in range(n)}
            completed = 0
            for fut in as_completed(futures):
                if _cancelled():
                    # We can't interrupt in-flight decoders (no abort hook in
                    # the C extensions), but stop processing any further.
                    for f in futures:
                        f.cancel()
                    raise LoaderCancelled()
                i, pixels = fut.result()
                if modality == "CT":
                    out[i] = (pixels.astype(np.float32) * slope + intercept).astype(np.int16)
                else:
                    out[i] = pixels.astype(np.float32)
                completed += 1
                if progress is not None and (completed % 25 == 0 or completed == n):
                    try:
                        progress(f"Decoding slices ({completed}/{n})", completed / n)
                    except Exception:
                        pass

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
