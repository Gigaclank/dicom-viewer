"""NIfTI .nii / .nii.gz support.

Three entry points:
  * `load_study_from_nifti(path)` -> Study (whole volume).
  * `save_segmentation_to_nifti(seg, volume, path)` writes the boolean mask
    as a uint8 NIfTI image whose voxel grid matches the source volume.
  * `load_segmentation_from_nifti(path, expected_volume)` loads an externally
    computed mask and validates the shape against the active volume.

NIfTI orientation is messy in the wild: the affine can encode any axis
ordering / flipping. We coerce every input through `nib.as_closest_canonical`
so the in-memory data is RAS+ (right / anterior / superior positive) and the
voxel order is consistent with what the rest of the app expects.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume


def is_nifti_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def load_study_from_nifti(path: Path) -> Study:
    """Read a NIfTI volume as a Study suitable for the viewer.

    Multi-volume (4D) NIfTIs collapse to the first frame; we're a 3D viewer.
    """
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"unsupported NIfTI dimensionality: {data.shape}")

    # nibabel returns (X, Y, Z); the viewer's Volume convention is (Z, Y, X).
    array = np.ascontiguousarray(np.transpose(data, (2, 1, 0)))

    # Voxel sizes from the header — also re-ordered (Z, Y, X).
    zooms = img.header.get_zooms()
    if len(zooms) < 3:
        spacing_mm = (1.0, 1.0, 1.0)
    else:
        spacing_mm = (float(zooms[2]), float(zooms[1]), float(zooms[0]))

    # Coerce to an int16 or float32 array, matching what the DICOM loader
    # produces. NIfTI may use uint16 / float64 etc.
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.int16, copy=False)
    else:
        array = array.astype(np.float32, copy=False)

    volume = Volume(array=array, spacing_mm=spacing_mm, modality="OT")
    return Study(
        volume=volume,
        patient_id="",
        patient_name="",
        study_uid="",
        series_uid=str(path),
        series_description=path.name,
        orientation_cosines=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )


def save_segmentation_to_nifti(
    segmentation: Segmentation,
    volume: Volume,
    path: Path,
) -> None:
    """Write the segmentation mask as a uint8 NIfTI volume.

    Voxel grid matches the source volume's spacing. World origin is (0,0,0).
    Use `load_segmentation_from_nifti` to round-trip back into the viewer.
    """
    mask = segmentation.mask.astype(np.uint8)
    # (Z, Y, X) -> (X, Y, Z) for nibabel.
    nifti_data = np.transpose(mask, (2, 1, 0))
    sz, sy, sx = volume.spacing_mm
    affine = np.diag([sx, sy, sz, 1.0])
    img = nib.Nifti1Image(nifti_data, affine=affine)
    img.header.set_zooms((sx, sy, sz))
    nib.save(img, str(path))


def load_segmentation_from_nifti(
    path: Path,
    expected_volume: Volume,
) -> Segmentation:
    """Read a NIfTI mask as a Segmentation, validating shape against `expected_volume`."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"unsupported NIfTI dimensionality: {data.shape}")

    # (X, Y, Z) -> (Z, Y, X), then bool.
    mask = np.transpose(data, (2, 1, 0)) > 0
    if mask.shape != expected_volume.shape:
        raise ValueError(
            f"NIfTI mask shape {mask.shape} does not match volume shape "
            f"{expected_volume.shape}"
        )

    return Segmentation(
        mask=np.ascontiguousarray(mask),
        method=f"imported({path.name})",
        params={"source": str(path)},
    )
