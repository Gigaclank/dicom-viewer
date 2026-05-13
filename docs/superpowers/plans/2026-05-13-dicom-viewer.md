# DICOM Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python desktop DICOM viewer (CT + MRI) with multi-planar viewing, threshold/region-grow segmentation, axis-aligned region selection, and binary STL export of the resulting mesh.

**Architecture:** Three-layer Python application. `core/` is pure Python (no Qt, no VTK) — `Study`, `Volume`, `Region`, `Segmentation`, `MeshExporter`, `Document`. `rendering/` wraps VTK for the four MPR + 3D panes and mesh preview. `ui/` is PyQt6, observing the `Document` as the single source of truth. All long-running work runs in `QThread`s.

**Tech Stack:** Python 3.10+, pip + venv, PyQt6, VTK, pydicom (+ `pylibjpeg-*`), numpy, scipy, SimpleITK, pytest, pytest-qt, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-05-13-dicom-viewer-design.md`

---

## File map

Files this plan creates (every one is the responsibility of exactly one task; nothing else):

```
pyproject.toml                                     # Task 1
src/dicom_viewer/__init__.py                       # Task 1
src/dicom_viewer/__main__.py                       # Task 19
src/dicom_viewer/app.py                            # Task 19
src/dicom_viewer/core/__init__.py                  # Task 1
src/dicom_viewer/core/region.py                    # Task 3
src/dicom_viewer/core/volume.py                    # Task 4
src/dicom_viewer/core/study.py                     # Task 5
src/dicom_viewer/core/segmentation/__init__.py     # Task 7
src/dicom_viewer/core/segmentation/base.py         # Task 7
src/dicom_viewer/core/segmentation/threshold.py    # Task 7
src/dicom_viewer/core/segmentation/morphology.py   # Task 8
src/dicom_viewer/core/segmentation/region_grow.py  # Task 9
src/dicom_viewer/core/mesh_export.py               # Task 10
src/dicom_viewer/core/document.py                  # Task 11
src/dicom_viewer/io/__init__.py                    # Task 1
src/dicom_viewer/io/dicom_loader.py                # Task 6
src/dicom_viewer/rendering/__init__.py             # Task 1
src/dicom_viewer/rendering/slice_renderer.py       # Task 12
src/dicom_viewer/rendering/volume_renderer.py      # Task 13
src/dicom_viewer/rendering/mesh_preview.py         # Task 14
src/dicom_viewer/ui/__init__.py                    # Task 1
src/dicom_viewer/ui/widgets/__init__.py            # Task 15
src/dicom_viewer/ui/widgets/slice_view.py          # Task 15
src/dicom_viewer/ui/panels/__init__.py             # Task 16
src/dicom_viewer/ui/panels/windowing.py            # Task 16
src/dicom_viewer/ui/panels/segmentation.py         # Task 17
src/dicom_viewer/ui/panels/export.py               # Task 18
src/dicom_viewer/ui/main_window.py                 # Task 19
tests/conftest.py                                  # Task 2
tests/fixtures/make_synthetic_series.py            # Task 2
tests/core/test_region.py                          # Task 3
tests/core/test_volume.py                          # Task 4
tests/core/test_study.py                           # Task 5
tests/io/test_dicom_loader.py                      # Task 6
tests/core/test_segmentation_threshold.py          # Task 7
tests/core/test_segmentation_morphology.py         # Task 8
tests/core/test_segmentation_region_grow.py        # Task 9
tests/core/test_mesh_export.py                     # Task 10
tests/core/test_document.py                        # Task 11
tests/rendering/test_slice_renderer.py             # Task 12
tests/rendering/test_volume_renderer.py            # Task 13
tests/rendering/test_mesh_preview.py               # Task 14
tests/ui/test_slice_view.py                        # Task 15
tests/ui/test_windowing_panel.py                   # Task 16
tests/ui/test_segmentation_panel.py                # Task 17
tests/ui/test_export_panel.py                      # Task 18
tests/integration/test_end_to_end.py               # Task 20
```

Dependency order: each task only uses what previous tasks created.

---

## Task 1: Project skeleton, dependencies, package layout

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/dicom_viewer/__init__.py`
- Create: `src/dicom_viewer/core/__init__.py`
- Create: `src/dicom_viewer/io/__init__.py`
- Create: `src/dicom_viewer/rendering/__init__.py`
- Create: `src/dicom_viewer/ui/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "dicom-viewer"
version = "0.1.0"
description = "Personal DICOM viewer with 3D-printable section export"
requires-python = ">=3.10"
dependencies = [
    "pydicom>=2.4",
    "pylibjpeg>=2.0",
    "pylibjpeg-libjpeg>=2.1",
    "pylibjpeg-openjpeg>=2.2",
    "numpy>=1.26",
    "scipy>=1.11",
    "SimpleITK>=2.3",
    "vtk>=9.3",
    "PyQt6>=6.6",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-qt>=4.4",
    "ruff>=0.4",
    "mypy>=1.10",
]

[project.scripts]
dicom-viewer = "dicom_viewer.app:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/dicom_viewer"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.py[cod]
.venv/
.uv/
.superpowers/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg-info/
dist/
build/
.coverage
```

- [ ] **Step 3: Create empty `__init__.py` for every package**

Files (each contains a single newline):
```
src/dicom_viewer/__init__.py
src/dicom_viewer/core/__init__.py
src/dicom_viewer/io/__init__.py
src/dicom_viewer/rendering/__init__.py
src/dicom_viewer/ui/__init__.py
tests/__init__.py
```

- [ ] **Step 4: Create venv and install deps**

Run:
```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -c "import pydicom, numpy, scipy, SimpleITK, vtk, PyQt6; print('ok')"
```

Expected: `ok` printed. No `ImportError`. The install step downloads several hundred MB of wheels — give it a few minutes.

- [ ] **Step 5: First commit (git is already initialized)**

```bash
git add pyproject.toml .gitignore src/ tests/ docs/
git commit -m "chore: initialize project skeleton"
```

---

## Task 2: Synthetic DICOM fixture generator

A small helper that writes a known-shape synthetic CT or MRI series under `tmp_path` for tests. Without this, every later test would need to fabricate DICOM headers inline.

**Files:**
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/make_synthetic_series.py`
- Create: `tests/conftest.py`
- Test: `tests/fixtures/test_make_synthetic_series.py`

- [ ] **Step 1: Create `tests/fixtures/__init__.py`**

A file containing one newline.

- [ ] **Step 2: Write the failing test `tests/fixtures/test_make_synthetic_series.py`**

```python
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
```

- [ ] **Step 3: Run test, confirm failure**

Run: `.venv/bin/pytest tests/fixtures/test_make_synthetic_series.py -v`
Expected: `ImportError` / `ModuleNotFoundError`.

- [ ] **Step 4: Implement `tests/fixtures/make_synthetic_series.py`**

```python
"""Generate synthetic DICOM series for tests.

The pixel data is deterministic: a centered cube of high intensity inside
low-intensity background. Use a small `shape` to keep tests fast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
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
```

- [ ] **Step 5: Add `tests/conftest.py`**

```python
"""Shared test fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

# Make tests/ a package importable as `tests.fixtures.*`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 6: Run test and verify pass**

Run: `.venv/bin/pytest tests/fixtures/test_make_synthetic_series.py -v`
Expected: 1 passed.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures tests/conftest.py
git commit -m "test: add synthetic DICOM series fixture generator"
```

---

## Task 3: `core.Region` — axis-aligned 3D bounding box

**Files:**
- Create: `src/dicom_viewer/core/region.py`
- Test: `tests/core/__init__.py`, `tests/core/test_region.py`

- [ ] **Step 1: Create `tests/core/__init__.py`**

A file containing one newline.

- [ ] **Step 2: Write the failing test `tests/core/test_region.py`**

```python
import pytest

from dicom_viewer.core.region import Region


def test_region_from_bounds_basic():
    r = Region(z=(0, 5), y=(0, 10), x=(0, 20))
    assert r.shape == (5, 10, 20)
    assert r.is_empty is False


def test_region_validates_ordering():
    with pytest.raises(ValueError):
        Region(z=(5, 0), y=(0, 10), x=(0, 20))


def test_region_intersect_overlap():
    a = Region(z=(0, 10), y=(0, 10), x=(0, 10))
    b = Region(z=(5, 15), y=(2, 8), x=(0, 20))
    c = a.intersect(b)
    assert c == Region(z=(5, 10), y=(2, 8), x=(0, 10))


def test_region_intersect_disjoint_is_empty():
    a = Region(z=(0, 5), y=(0, 5), x=(0, 5))
    b = Region(z=(10, 15), y=(0, 5), x=(0, 5))
    assert a.intersect(b).is_empty


def test_region_clamp_to():
    r = Region(z=(-1, 12), y=(-3, 6), x=(0, 100))
    clamped = r.clamp_to(Region(z=(0, 10), y=(0, 5), x=(0, 50)))
    assert clamped == Region(z=(0, 10), y=(0, 5), x=(0, 50))


def test_region_size_mm():
    r = Region(z=(0, 5), y=(0, 10), x=(0, 20))
    assert r.size_mm(spacing_mm=(2.0, 1.0, 0.5)) == (10.0, 10.0, 10.0)
```

- [ ] **Step 3: Run test and confirm failure**

Run: `.venv/bin/pytest tests/core/test_region.py -v`
Expected: `ImportError` / `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/dicom_viewer/core/region.py`**

```python
"""Axis-aligned 3D bounding box in voxel space.

Coordinates are in (z, y, x). All ranges are half-open [start, stop)
matching numpy slice semantics. shape is (stop - start) per axis.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    z: tuple[int, int]
    y: tuple[int, int]
    x: tuple[int, int]

    def __post_init__(self) -> None:
        for axis_name, (lo, hi) in (("z", self.z), ("y", self.y), ("x", self.x)):
            if lo > hi:
                raise ValueError(f"Region.{axis_name} has lo>hi: {(lo, hi)}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            self.z[1] - self.z[0],
            self.y[1] - self.y[0],
            self.x[1] - self.x[0],
        )

    @property
    def is_empty(self) -> bool:
        return any(s <= 0 for s in self.shape)

    def intersect(self, other: "Region") -> "Region":
        return Region(
            z=(max(self.z[0], other.z[0]), min(self.z[1], other.z[1])),
            y=(max(self.y[0], other.y[0]), min(self.y[1], other.y[1])),
            x=(max(self.x[0], other.x[0]), min(self.x[1], other.x[1])),
        )

    def clamp_to(self, bounds: "Region") -> "Region":
        return self.intersect(bounds)

    def size_mm(self, spacing_mm: tuple[float, float, float]) -> tuple[float, float, float]:
        sz, sy, sx = self.shape
        dz, dy, dx = spacing_mm
        return (sz * dz, sy * dy, sx * dx)
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `.venv/bin/pytest tests/core/test_region.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/dicom_viewer/core/region.py tests/core/__init__.py tests/core/test_region.py
git commit -m "feat(core): add Region axis-aligned bounding box"
```

---

## Task 4: `core.Volume` — voxel ndarray with metadata

**Files:**
- Create: `src/dicom_viewer/core/volume.py`
- Test: `tests/core/test_volume.py`

- [ ] **Step 1: Write the failing test `tests/core/test_volume.py`**

```python
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
```

- [ ] **Step 2: Run test, confirm failure**

Run: `.venv/bin/pytest tests/core/test_volume.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/dicom_viewer/core/volume.py`**

```python
"""3D voxel volume with spacing and modality metadata.

The array layout is always (z, y, x). Methods that take an Orientation slice
through the array; the returned 2D arrays use display conventions:
- AXIAL slice (constant z): rows=y, cols=x.
- CORONAL slice (constant y): rows=z (flipped so superior is up), cols=x.
- SAGITTAL slice (constant x): rows=z (flipped), cols=y.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from dicom_viewer.core.region import Region


class Orientation(str, Enum):
    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"


@dataclass(frozen=True)
class Volume:
    array: np.ndarray  # shape (z, y, x); dtype int16 for CT, float32 for MR
    spacing_mm: tuple[float, float, float]  # (z, y, x)
    modality: str  # "CT", "MR", ...

    @property
    def shape(self) -> tuple[int, int, int]:
        z, y, x = self.array.shape
        return (z, y, x)

    def bbox(self) -> Region:
        z, y, x = self.shape
        return Region(z=(0, z), y=(0, y), x=(0, x))

    def slice(self, orientation: Orientation, index: int) -> np.ndarray:
        z, y, x = self.shape
        if orientation is Orientation.AXIAL:
            if not 0 <= index < z:
                raise IndexError(f"axial index {index} out of [0,{z})")
            return self.array[index, :, :]
        if orientation is Orientation.CORONAL:
            if not 0 <= index < y:
                raise IndexError(f"coronal index {index} out of [0,{y})")
            # Flip z so the rendered image shows superior at top.
            return self.array[:, index, :][::-1, :]
        if orientation is Orientation.SAGITTAL:
            if not 0 <= index < x:
                raise IndexError(f"sagittal index {index} out of [0,{x})")
            return self.array[:, :, index][::-1, :]
        raise ValueError(f"unknown orientation {orientation!r}")

    def windowed(
        self, orientation: Orientation, index: int, center: float, width: float
    ) -> np.ndarray:
        if width <= 0:
            raise ValueError("window width must be > 0")
        s = self.slice(orientation, index).astype(np.float32)
        lo = center - width / 2.0
        hi = center + width / 2.0
        scaled = np.clip((s - lo) / (hi - lo), 0.0, 1.0)
        return (scaled * 255.0 + 0.5).astype(np.uint8)

    def crop(self, region: Region) -> "Volume":
        bounds = self.bbox()
        r = region.clamp_to(bounds)
        if r.is_empty:
            raise ValueError("cropping with an empty region")
        sub = self.array[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]]
        return Volume(array=sub, spacing_mm=self.spacing_mm, modality=self.modality)

    def intensity_range(self) -> tuple[float, float]:
        return (float(self.array.min()), float(self.array.max()))

    def intensity_percentiles(self, low_pct: float, high_pct: float) -> tuple[float, float]:
        lo = float(np.percentile(self.array, low_pct))
        hi = float(np.percentile(self.array, high_pct))
        return lo, hi
```

- [ ] **Step 4: Run tests and verify pass**

Run: `.venv/bin/pytest tests/core/test_volume.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dicom_viewer/core/volume.py tests/core/test_volume.py
git commit -m "feat(core): add Volume with MPR slicing and windowing"
```

---

## Task 5: `core.Study` — DICOM metadata container

**Files:**
- Create: `src/dicom_viewer/core/study.py`
- Test: `tests/core/test_study.py`

- [ ] **Step 1: Write the failing test `tests/core/test_study.py`**

```python
import numpy as np

from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume


def test_study_wraps_volume_and_metadata():
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    volume = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    study = Study(
        volume=volume,
        patient_id="TEST001",
        patient_name="Test^Synthetic",
        study_uid="1.2.3.4",
        series_uid="1.2.3.4.5",
        series_description="synthetic-ct",
        orientation_cosines=(1, 0, 0, 0, 1, 0),
    )
    assert study.modality == "CT"
    assert study.spacing_mm == (1.0, 1.0, 1.0)
    assert study.volume is volume
    assert study.display_name == "TEST001 / synthetic-ct"


def test_study_anonymized_name_when_no_patient_id():
    volume = Volume(
        array=np.zeros((2, 2, 2), dtype=np.int16),
        spacing_mm=(1.0, 1.0, 1.0),
        modality="MR",
    )
    study = Study(
        volume=volume,
        patient_id="",
        patient_name="",
        study_uid="x",
        series_uid="y",
        series_description="anon",
        orientation_cosines=(1, 0, 0, 0, 1, 0),
    )
    assert study.display_name == "<anonymized> / anon"
```

- [ ] **Step 2: Run test, confirm failure**

Run: `.venv/bin/pytest tests/core/test_study.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/dicom_viewer/core/study.py`**

```python
"""Study — a single DICOM series ready to view."""
from __future__ import annotations

from dataclasses import dataclass

from dicom_viewer.core.volume import Volume


@dataclass(frozen=True)
class Study:
    volume: Volume
    patient_id: str
    patient_name: str
    study_uid: str
    series_uid: str
    series_description: str
    orientation_cosines: tuple[float, float, float, float, float, float]

    @property
    def modality(self) -> str:
        return self.volume.modality

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        return self.volume.spacing_mm

    @property
    def display_name(self) -> str:
        patient = self.patient_id or "<anonymized>"
        return f"{patient} / {self.series_description or '<no description>'}"
```

- [ ] **Step 4: Run tests and verify pass**

Run: `.venv/bin/pytest tests/core/test_study.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dicom_viewer/core/study.py tests/core/test_study.py
git commit -m "feat(core): add Study metadata container"
```

---

## Task 6: `io.dicom_loader` — load a DICOM folder into one or more Studies

**Files:**
- Create: `src/dicom_viewer/io/dicom_loader.py`
- Test: `tests/io/__init__.py`, `tests/io/test_dicom_loader.py`

- [ ] **Step 1: Create `tests/io/__init__.py`**

A file containing one newline.

- [ ] **Step 2: Write the failing test `tests/io/test_dicom_loader.py`**

```python
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
```

- [ ] **Step 3: Run test, confirm failure**

Run: `.venv/bin/pytest tests/io/test_dicom_loader.py -v`
Expected: `ImportError`.

- [ ] **Step 4: Implement `src/dicom_viewer/io/dicom_loader.py`**

```python
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
```

- [ ] **Step 5: Run tests and verify pass**

Run: `.venv/bin/pytest tests/io/test_dicom_loader.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/dicom_viewer/io/dicom_loader.py tests/io/
git commit -m "feat(io): add DICOM folder loader with series grouping"
```

---

## Task 7: `core.segmentation.threshold` + base type

**Files:**
- Create: `src/dicom_viewer/core/segmentation/__init__.py`
- Create: `src/dicom_viewer/core/segmentation/base.py`
- Create: `src/dicom_viewer/core/segmentation/threshold.py`
- Test: `tests/core/test_segmentation_threshold.py`

- [ ] **Step 1: Create `src/dicom_viewer/core/segmentation/__init__.py`**

```python
"""Segmentation methods. Each public function returns a Segmentation."""
from dicom_viewer.core.segmentation.base import Segmentation  # re-export
from dicom_viewer.core.segmentation.threshold import threshold

__all__ = ["Segmentation", "threshold"]
```

- [ ] **Step 2: Write the failing test `tests/core/test_segmentation_threshold.py`**

```python
import numpy as np

from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.volume import Volume


def _cube_volume() -> Volume:
    arr = np.zeros((8, 8, 8), dtype=np.int16)
    arr[2:6, 2:6, 2:6] = 500
    arr[0, 0, 0] = 9999  # an outlier voxel
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_threshold_inclusive_low_high():
    v = _cube_volume()
    seg = threshold(v, low=100, high=1000)
    assert seg.mask.shape == v.shape
    assert seg.mask.dtype == bool
    # All cube voxels selected.
    assert seg.mask[3, 3, 3]
    # Background not selected.
    assert not seg.mask[0, 1, 0]
    # Outlier above high is excluded.
    assert not seg.mask[0, 0, 0]


def test_threshold_records_provenance():
    v = _cube_volume()
    seg = threshold(v, low=100, high=1000)
    assert seg.method == "threshold"
    assert seg.params == {"low": 100, "high": 1000}


def test_threshold_handles_low_equals_high():
    v = _cube_volume()
    seg = threshold(v, low=500, high=500)
    assert int(seg.mask.sum()) == int((v.array == 500).sum())
```

- [ ] **Step 3: Run test, confirm failure**

Run: `.venv/bin/pytest tests/core/test_segmentation_threshold.py -v`
Expected: `ImportError`.

- [ ] **Step 4: Implement `src/dicom_viewer/core/segmentation/base.py`**

```python
"""Segmentation result type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Segmentation:
    mask: np.ndarray  # bool, shape (z, y, x), same as source Volume
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_:
            raise ValueError(f"mask must be bool, got {self.mask.dtype}")

    @property
    def voxel_count(self) -> int:
        return int(self.mask.sum())

    @property
    def is_empty(self) -> bool:
        return self.voxel_count == 0
```

- [ ] **Step 5: Implement `src/dicom_viewer/core/segmentation/threshold.py`**

```python
"""Intensity-threshold segmentation."""
from __future__ import annotations

import numpy as np

from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.volume import Volume


def threshold(volume: Volume, low: float, high: float) -> Segmentation:
    """Select voxels with low <= value <= high."""
    if high < low:
        raise ValueError(f"threshold high ({high}) < low ({low})")
    mask = (volume.array >= low) & (volume.array <= high)
    return Segmentation(
        mask=np.ascontiguousarray(mask),
        method="threshold",
        params={"low": low, "high": high},
    )
```

- [ ] **Step 6: Run tests and verify pass**

Run: `.venv/bin/pytest tests/core/test_segmentation_threshold.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/dicom_viewer/core/segmentation tests/core/test_segmentation_threshold.py
git commit -m "feat(core): add threshold segmentation and Segmentation type"
```

---

## Task 8: `core.segmentation.morphology` — largest component + smoothing

**Files:**
- Create: `src/dicom_viewer/core/segmentation/morphology.py`
- Modify: `src/dicom_viewer/core/segmentation/__init__.py`
- Test: `tests/core/test_segmentation_morphology.py`

- [ ] **Step 1: Write the failing test `tests/core/test_segmentation_morphology.py`**

```python
import numpy as np

from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.segmentation.morphology import keep_largest_component, smooth_mask


def _two_blob_seg(shape=(10, 10, 10)) -> Segmentation:
    mask = np.zeros(shape, dtype=bool)
    mask[1:4, 1:4, 1:4] = True   # blob 1, 27 voxels
    mask[6:9, 6:9, 6:9] = True   # blob 2, 27 voxels (tie)
    mask[5, 5, 5] = True         # isolated speck
    # Make blob 1 strictly larger so we have a determinate winner.
    mask[1:4, 1:4, 4] = True
    return Segmentation(mask=mask, method="threshold", params={})


def test_keep_largest_component_drops_others():
    seg = _two_blob_seg()
    result = keep_largest_component(seg)
    # Only one connected component remains.
    from scipy.ndimage import label as nd_label
    _, n_components = nd_label(result.mask)
    assert n_components == 1
    # And it's the bigger blob (contains voxel (2,2,4)).
    assert result.mask[2, 2, 4]
    assert result.method == "threshold+largest_component"
    assert "source_method" in result.params


def test_keep_largest_component_on_empty_is_empty():
    seg = Segmentation(mask=np.zeros((4, 4, 4), dtype=bool), method="threshold", params={})
    out = keep_largest_component(seg)
    assert out.is_empty


def test_smooth_mask_removes_specks_and_fills_pinholes():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True  # solid 6³ cube
    mask[5, 5, 5] = False        # pinhole inside
    mask[0, 0, 0] = True         # isolated speck outside
    seg = Segmentation(mask=mask, method="threshold", params={})
    out = smooth_mask(seg, iterations=1)
    assert out.mask[5, 5, 5]      # pinhole filled
    assert not out.mask[0, 0, 0]  # speck removed
    assert out.method.endswith("+smooth")
```

- [ ] **Step 2: Run test, confirm failure**

Run: `.venv/bin/pytest tests/core/test_segmentation_morphology.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/dicom_viewer/core/segmentation/morphology.py`**

```python
"""Connected-component and morphology operations on segmentation masks."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label

from dicom_viewer.core.segmentation.base import Segmentation


def keep_largest_component(seg: Segmentation) -> Segmentation:
    if seg.is_empty:
        return Segmentation(
            mask=seg.mask.copy(),
            method=f"{seg.method}+largest_component",
            params={"source_method": seg.method, "source_params": dict(seg.params)},
        )
    labeled, n = label(seg.mask)
    if n == 0:
        return Segmentation(
            mask=np.zeros_like(seg.mask),
            method=f"{seg.method}+largest_component",
            params={"source_method": seg.method, "source_params": dict(seg.params)},
        )
    # bincount ignores label 0 (background).
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    winner = int(counts.argmax())
    return Segmentation(
        mask=(labeled == winner),
        method=f"{seg.method}+largest_component",
        params={"source_method": seg.method, "source_params": dict(seg.params)},
    )


def smooth_mask(seg: Segmentation, iterations: int = 1) -> Segmentation:
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    # Closing fills pinholes; opening removes specks.
    closed = binary_closing(seg.mask, iterations=iterations)
    opened = binary_opening(closed, iterations=iterations)
    return Segmentation(
        mask=np.ascontiguousarray(opened),
        method=f"{seg.method}+smooth",
        params={
            "source_method": seg.method,
            "source_params": dict(seg.params),
            "iterations": iterations,
        },
    )
```

- [ ] **Step 4: Update `src/dicom_viewer/core/segmentation/__init__.py`**

Replace the file with:
```python
"""Segmentation methods. Each public function returns a Segmentation."""
from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.segmentation.morphology import keep_largest_component, smooth_mask
from dicom_viewer.core.segmentation.threshold import threshold

__all__ = [
    "Segmentation",
    "threshold",
    "keep_largest_component",
    "smooth_mask",
]
```

- [ ] **Step 5: Run tests and verify pass**

Run: `.venv/bin/pytest tests/core/test_segmentation_morphology.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/dicom_viewer/core/segmentation tests/core/test_segmentation_morphology.py
git commit -m "feat(core): add largest-component and mask smoothing"
```

---

## Task 9: `core.segmentation.region_grow`

**Files:**
- Create: `src/dicom_viewer/core/segmentation/region_grow.py`
- Modify: `src/dicom_viewer/core/segmentation/__init__.py`
- Test: `tests/core/test_segmentation_region_grow.py`

- [ ] **Step 1: Write the failing test `tests/core/test_segmentation_region_grow.py`**

```python
import numpy as np

from dicom_viewer.core.segmentation.region_grow import region_grow
from dicom_viewer.core.volume import Volume


def test_region_grow_fills_connected_region():
    arr = np.zeros((10, 10, 10), dtype=np.int16)
    arr[3:7, 3:7, 3:7] = 500     # foreground blob
    arr[0, 0, 0] = 500           # isolated voxel at far corner
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    seg = region_grow(v, seed=(5, 5, 5), tolerance=10)
    # The connected blob is selected.
    assert seg.mask[3, 3, 3]
    assert seg.mask[6, 6, 6]
    # Isolated voxel is NOT (not connected).
    assert not seg.mask[0, 0, 0]
    # Background isn't.
    assert not seg.mask[8, 8, 8]
    assert seg.method == "region_grow"
    assert seg.params == {"seed": (5, 5, 5), "tolerance": 10}


def test_region_grow_seed_out_of_bounds():
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    import pytest
    with pytest.raises(ValueError):
        region_grow(v, seed=(99, 0, 0), tolerance=10)
```

- [ ] **Step 2: Run test, confirm failure**

Run: `.venv/bin/pytest tests/core/test_segmentation_region_grow.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/dicom_viewer/core/segmentation/region_grow.py`**

```python
"""Region-growing segmentation via SimpleITK.ConnectedThreshold."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.volume import Volume


def region_grow(
    volume: Volume, seed: tuple[int, int, int], tolerance: float
) -> Segmentation:
    """Flood-fill from `seed` (in z,y,x voxel coords) within ±tolerance of seed value."""
    z, y, x = seed
    sz, sy, sx = volume.shape
    if not (0 <= z < sz and 0 <= y < sy and 0 <= x < sx):
        raise ValueError(f"seed {seed} outside volume shape {volume.shape}")

    # SimpleITK uses (x, y, z) index order.
    image = sitk.GetImageFromArray(volume.array)
    seed_value = float(volume.array[z, y, x])

    grown = sitk.ConnectedThreshold(
        image,
        seedList=[(int(x), int(y), int(z))],
        lower=float(seed_value - tolerance),
        upper=float(seed_value + tolerance),
        replaceValue=1,
    )
    mask = sitk.GetArrayFromImage(grown).astype(bool)
    return Segmentation(
        mask=np.ascontiguousarray(mask),
        method="region_grow",
        params={"seed": (int(z), int(y), int(x)), "tolerance": tolerance},
    )
```

- [ ] **Step 4: Update `src/dicom_viewer/core/segmentation/__init__.py`**

Replace the file with:
```python
"""Segmentation methods. Each public function returns a Segmentation."""
from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.segmentation.morphology import keep_largest_component, smooth_mask
from dicom_viewer.core.segmentation.region_grow import region_grow
from dicom_viewer.core.segmentation.threshold import threshold

__all__ = [
    "Segmentation",
    "threshold",
    "keep_largest_component",
    "smooth_mask",
    "region_grow",
]
```

- [ ] **Step 5: Run tests and verify pass**

Run: `.venv/bin/pytest tests/core/test_segmentation_region_grow.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/dicom_viewer/core/segmentation tests/core/test_segmentation_region_grow.py
git commit -m "feat(core): add region-grow segmentation"
```

---

## Task 10: `core.mesh_export` — marching cubes → STL

**Files:**
- Create: `src/dicom_viewer/core/mesh_export.py`
- Test: `tests/core/test_mesh_export.py`

- [ ] **Step 1: Write the failing test `tests/core/test_mesh_export.py`**

```python
import struct

import numpy as np
import pytest

from dicom_viewer.core.mesh_export import (
    EmptyMeshError,
    ExportOptions,
    export_stl,
    generate_mesh,
)
from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.volume import Volume


def _cube_volume(side: int = 16, cube_size: int = 8) -> Volume:
    arr = np.zeros((side, side, side), dtype=np.int16)
    s = (side - cube_size) // 2
    arr[s : s + cube_size, s : s + cube_size, s : s + cube_size] = 1000
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_generate_mesh_produces_triangles_for_cube():
    v = _cube_volume()
    seg = threshold(v, low=500, high=2000)
    mesh = generate_mesh(v, seg, region=v.bbox(), options=ExportOptions())
    assert mesh.triangle_count > 0
    # Bounds should sit inside the volume in mm.
    lo, hi = mesh.bounds_mm
    for axis in range(3):
        assert lo[axis] >= 0
        assert hi[axis] <= v.shape[axis] * v.spacing_mm[axis]


def test_generate_mesh_empty_mask_raises():
    v = _cube_volume()
    empty = threshold(v, low=9000, high=9001)
    with pytest.raises(EmptyMeshError):
        generate_mesh(v, empty, region=v.bbox(), options=ExportOptions())


def test_generate_mesh_respects_region():
    v = _cube_volume(side=16, cube_size=8)
    seg = threshold(v, low=500, high=2000)
    # Crop to lower half — should clip the cube.
    region = Region(z=(0, 8), y=(0, 16), x=(0, 16))
    mesh = generate_mesh(v, seg, region=region, options=ExportOptions())
    lo, hi = mesh.bounds_mm
    assert hi[0] <= 8.0 + 1e-3  # within cropped z extent


def test_export_stl_writes_binary_stl(tmp_path):
    v = _cube_volume()
    seg = threshold(v, low=500, high=2000)
    out = tmp_path / "cube.stl"
    mesh = generate_mesh(v, seg, region=v.bbox(), options=ExportOptions())
    export_stl(mesh, out)
    data = out.read_bytes()
    assert len(data) >= 84
    n_triangles = struct.unpack("<I", data[80:84])[0]
    assert n_triangles == mesh.triangle_count
    assert len(data) == 84 + n_triangles * 50
```

- [ ] **Step 2: Run test, confirm failure**

Run: `.venv/bin/pytest tests/core/test_mesh_export.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/dicom_viewer/core/mesh_export.py`**

```python
"""Marching-cubes mesh export pipeline.

generate_mesh: Volume × Segmentation × Region -> Mesh (VTK polydata + metadata)
export_stl:    Mesh -> on-disk binary STL.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support  # type: ignore[import-untyped]

from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.volume import Volume


class EmptyMeshError(Exception):
    """The chosen segmentation/region intersection produced zero voxels or zero triangles."""


@dataclass(frozen=True)
class ExportOptions:
    smoothing_iterations: int = 15
    pass_band: float = 0.1
    decimation_target_reduction: float = 0.5
    ensure_manifold: bool = True


@dataclass(frozen=True)
class Mesh:
    polydata: "vtk.vtkPolyData"
    triangle_count: int
    bounds_mm: tuple[tuple[float, float, float], tuple[float, float, float]]


def generate_mesh(
    volume: Volume,
    segmentation: Segmentation,
    region: Region,
    options: ExportOptions,
) -> Mesh:
    bounds = volume.bbox()
    r = region.clamp_to(bounds)
    if r.is_empty:
        raise EmptyMeshError("region is empty after clamping to volume")

    cropped_mask = segmentation.mask[r.z[0] : r.z[1], r.y[0] : r.y[1], r.x[0] : r.x[1]]
    if not cropped_mask.any():
        raise EmptyMeshError("no voxels selected within region")

    image = _mask_to_vtk_image(cropped_mask, volume.spacing_mm, origin_voxel=(r.z[0], r.y[0], r.x[0]), spacing=volume.spacing_mm)

    marching = vtk.vtkDiscreteMarchingCubes()
    marching.SetInputData(image)
    marching.SetValue(0, 1)
    marching.Update()

    pipeline: vtk.vtkAlgorithm = marching

    if options.smoothing_iterations > 0:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(pipeline.GetOutputPort())
        smoother.SetNumberOfIterations(options.smoothing_iterations)
        smoother.SetPassBand(options.pass_band)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        pipeline = smoother

    if 0.0 < options.decimation_target_reduction < 1.0:
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputConnection(pipeline.GetOutputPort())
        decimator.SetTargetReduction(options.decimation_target_reduction)
        decimator.Update()
        pipeline = decimator

    if options.ensure_manifold:
        filler = vtk.vtkFillHolesFilter()
        filler.SetInputConnection(pipeline.GetOutputPort())
        filler.SetHoleSize(1e6)
        filler.Update()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(filler.GetOutputPort())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.Update()
        pipeline = normals

    poly: vtk.vtkPolyData = pipeline.GetOutput()
    n_tri = int(poly.GetNumberOfPolys())
    if n_tri == 0:
        raise EmptyMeshError("mesh has zero triangles after processing")

    bounds = poly.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    lo = (float(bounds[4]), float(bounds[2]), float(bounds[0]))  # (z,y,x)
    hi = (float(bounds[5]), float(bounds[3]), float(bounds[1]))
    return Mesh(polydata=poly, triangle_count=n_tri, bounds_mm=(lo, hi))


def export_stl(mesh: Mesh, path: Path) -> None:
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(mesh.polydata)
    if writer.Write() != 1:
        raise OSError(f"vtkSTLWriter failed to write {path}")


def _mask_to_vtk_image(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    origin_voxel: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> "vtk.vtkImageData":
    """Convert a (z,y,x) boolean numpy mask into a vtkImageData (x,y,z order)."""
    arr_uint = mask.astype(np.uint8)
    # VTK expects flat array in x-fastest order, but vtkImageData with the right dims
    # plus numpy_support handles the (z,y,x) -> (x,y,z) interpretation via SetDimensions.
    z, y, x = arr_uint.shape
    image = vtk.vtkImageData()
    image.SetDimensions(x, y, z)
    image.SetSpacing(spacing[2], spacing[1], spacing[0])
    image.SetOrigin(
        origin_voxel[2] * spacing[2],
        origin_voxel[1] * spacing[1],
        origin_voxel[0] * spacing[0],
    )
    flat = arr_uint.transpose(2, 1, 0).ravel(order="F")  # noqa: F841 (kept for clarity)
    # Simpler & safer: use numpy_support directly on the original (z,y,x) ravel.
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=arr_uint.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    image.GetPointData().SetScalars(vtk_array)
    return image
```

- [ ] **Step 4: Run tests and verify pass**

Run: `.venv/bin/pytest tests/core/test_mesh_export.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dicom_viewer/core/mesh_export.py tests/core/test_mesh_export.py
git commit -m "feat(core): add marching-cubes mesh generation and STL export"
```

---

## Task 11: `core.Document` — application state with observer callbacks

**Files:**
- Create: `src/dicom_viewer/core/document.py`
- Test: `tests/core/test_document.py`

- [ ] **Step 1: Write the failing test `tests/core/test_document.py`**

```python
import numpy as np

from dicom_viewer.core.document import Document, WindowingState
from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume


def _study() -> Study:
    arr = np.zeros((8, 8, 8), dtype=np.int16)
    arr[2:6, 2:6, 2:6] = 500
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    return Study(
        volume=v,
        patient_id="P1",
        patient_name="X",
        study_uid="s",
        series_uid="ser",
        series_description="test",
        orientation_cosines=(1, 0, 0, 0, 1, 0),
    )


def test_set_study_notifies_observers():
    events: list[str] = []
    doc = Document()
    doc.subscribe(lambda kind: events.append(kind))
    doc.set_study(_study())
    assert "study" in events
    assert doc.study is not None
    assert doc.volume is not None


def test_set_segmentation_notifies():
    doc = Document()
    doc.set_study(_study())
    events: list[str] = []
    doc.subscribe(lambda kind: events.append(kind))
    seg = threshold(doc.volume, low=100, high=1000)
    doc.set_segmentation(seg)
    assert "segmentation" in events
    assert doc.segmentation is seg


def test_set_region_notifies_and_clamps():
    doc = Document()
    doc.set_study(_study())
    events: list[str] = []
    doc.subscribe(lambda kind: events.append(kind))
    doc.set_region(Region(z=(-5, 100), y=(0, 4), x=(0, 4)))
    assert "region" in events
    # Clamped to volume bbox.
    assert doc.region == Region(z=(0, 8), y=(0, 4), x=(0, 4))


def test_windowing_defaults_and_update():
    doc = Document()
    doc.set_study(_study())
    assert doc.windowing.width > 0
    events: list[str] = []
    doc.subscribe(lambda kind: events.append(kind))
    doc.set_windowing(WindowingState(center=400, width=1500))
    assert "windowing" in events
    assert doc.windowing.center == 400


def test_unsubscribe():
    doc = Document()
    seen: list[str] = []
    handle = doc.subscribe(lambda kind: seen.append(kind))
    handle()  # unsubscribe
    doc.set_study(_study())
    assert seen == []
```

- [ ] **Step 2: Run test, confirm failure**

Run: `.venv/bin/pytest tests/core/test_document.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/dicom_viewer/core/document.py`**

```python
"""Document — the single source of truth for the loaded study and edits.

Observers register a callback via `subscribe(fn)`; `subscribe` returns a
zero-argument unsubscribe handle. Callbacks receive a string event-kind:
"study" | "volume" | "segmentation" | "region" | "windowing".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume

Observer = Callable[[str], None]


@dataclass(frozen=True)
class WindowingState:
    center: float
    width: float


_CT_DEFAULT = WindowingState(center=40, width=400)  # soft tissue
_MR_DEFAULT_FALLBACK = WindowingState(center=300, width=600)


class Document:
    def __init__(self) -> None:
        self._study: Study | None = None
        self._segmentation: Segmentation | None = None
        self._region: Region | None = None
        self._windowing: WindowingState = _CT_DEFAULT
        self._observers: list[Observer] = []

    # --- observer plumbing ---
    def subscribe(self, fn: Observer) -> Callable[[], None]:
        self._observers.append(fn)

        def unsubscribe() -> None:
            try:
                self._observers.remove(fn)
            except ValueError:
                pass

        return unsubscribe

    def _emit(self, kind: str) -> None:
        for fn in list(self._observers):
            fn(kind)

    # --- accessors ---
    @property
    def study(self) -> Study | None:
        return self._study

    @property
    def volume(self) -> Volume | None:
        return self._study.volume if self._study else None

    @property
    def segmentation(self) -> Segmentation | None:
        return self._segmentation

    @property
    def region(self) -> Region | None:
        return self._region

    @property
    def windowing(self) -> WindowingState:
        return self._windowing

    # --- mutators ---
    def set_study(self, study: Study) -> None:
        self._study = study
        self._segmentation = None
        self._region = study.volume.bbox()
        self._windowing = self._default_windowing_for(study.volume)
        self._emit("study")
        self._emit("volume")
        self._emit("region")
        self._emit("windowing")

    def set_segmentation(self, seg: Segmentation | None) -> None:
        self._segmentation = seg
        self._emit("segmentation")

    def set_region(self, region: Region) -> None:
        if self.volume is None:
            self._region = region
        else:
            self._region = region.clamp_to(self.volume.bbox())
        self._emit("region")

    def set_windowing(self, w: WindowingState) -> None:
        self._windowing = w
        self._emit("windowing")

    @staticmethod
    def _default_windowing_for(volume: Volume) -> WindowingState:
        if volume.modality == "CT":
            return _CT_DEFAULT
        try:
            lo, hi = volume.intensity_percentiles(1, 99)
            return WindowingState(center=(lo + hi) / 2.0, width=max(hi - lo, 1.0))
        except Exception:
            return _MR_DEFAULT_FALLBACK
```

- [ ] **Step 4: Run tests and verify pass**

Run: `.venv/bin/pytest tests/core/test_document.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dicom_viewer/core/document.py tests/core/test_document.py
git commit -m "feat(core): add Document state container with observers"
```

---

## Task 12: `rendering.SliceRenderer`

**Files:**
- Create: `src/dicom_viewer/rendering/slice_renderer.py`
- Test: `tests/rendering/__init__.py`, `tests/rendering/test_slice_renderer.py`

- [ ] **Step 1: Create `tests/rendering/__init__.py`**

A file containing one newline.

- [ ] **Step 2: Write the failing test `tests/rendering/test_slice_renderer.py`**

```python
"""Smoke tests for SliceRenderer using offscreen VTK."""
import os

import numpy as np
import pytest

from dicom_viewer.core.volume import Orientation, Volume
from dicom_viewer.rendering.slice_renderer import SliceRenderer


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    monkeypatch.setenv("DICOM_VIEWER_OFFSCREEN", "1")


def _vol() -> Volume:
    arr = np.zeros((6, 6, 6), dtype=np.int16)
    arr[1:5, 1:5, 1:5] = 500
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_slice_renderer_instantiates_offscreen():
    r = SliceRenderer(orientation=Orientation.AXIAL)
    r.set_volume(_vol())
    r.set_slice_index(3)
    r.set_windowing(center=250, width=500)
    r.render()
    assert r.current_index == 3


def test_slice_renderer_clamps_index():
    r = SliceRenderer(orientation=Orientation.AXIAL)
    r.set_volume(_vol())
    r.set_slice_index(999)
    assert r.current_index == 5  # last valid axial index
    r.set_slice_index(-5)
    assert r.current_index == 0


def test_slice_renderer_overlay_does_not_crash():
    r = SliceRenderer(orientation=Orientation.AXIAL)
    r.set_volume(_vol())
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[2:5, 2:5, 2:5] = True
    r.set_overlay_mask(mask)
    r.render()
```

- [ ] **Step 3: Run test, confirm failure**

Run: `.venv/bin/pytest tests/rendering/test_slice_renderer.py -v`
Expected: `ImportError`.

- [ ] **Step 4: Implement `src/dicom_viewer/rendering/slice_renderer.py`**

```python
"""Renders one MPR slice using VTK.

In production, an interactor (QVTKRenderWindowInteractor) is attached from
the UI layer via `attach_interactor`. For tests, the renderer falls back to a
detached offscreen window when DICOM_VIEWER_OFFSCREEN=1 is set.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import vtk
from vtk.util import numpy_support  # type: ignore[import-untyped]

from dicom_viewer.core.volume import Orientation, Volume


class SliceRenderer:
    def __init__(self, orientation: Orientation) -> None:
        self.orientation = orientation
        self._volume: Volume | None = None
        self._index: int = 0
        self._center: float = 40
        self._width: float = 400
        self._overlay_mask: np.ndarray | None = None

        self._image_actor = vtk.vtkImageActor()
        self._overlay_actor = vtk.vtkImageActor()
        self._renderer = vtk.vtkRenderer()
        self._renderer.AddActor(self._image_actor)
        self._renderer.AddActor(self._overlay_actor)
        self._renderer.SetBackground(0.0, 0.0, 0.0)
        self._render_window: vtk.vtkRenderWindow | None = None

        if os.environ.get("DICOM_VIEWER_OFFSCREEN") == "1":
            rw = vtk.vtkRenderWindow()
            rw.SetOffScreenRendering(1)
            rw.AddRenderer(self._renderer)
            rw.SetSize(64, 64)
            self._render_window = rw

    def attach_render_window(self, render_window: vtk.vtkRenderWindow) -> None:
        render_window.AddRenderer(self._renderer)
        self._render_window = render_window

    # --- inputs ---
    def set_volume(self, volume: Volume) -> None:
        self._volume = volume
        max_index = self._max_index()
        self._index = min(self._index, max_index)
        self._refresh_image()

    def set_slice_index(self, index: int) -> None:
        if self._volume is None:
            self._index = max(index, 0)
            return
        self._index = max(0, min(index, self._max_index()))
        self._refresh_image()

    def set_windowing(self, center: float, width: float) -> None:
        self._center = center
        self._width = max(width, 1.0)
        self._refresh_image()

    def set_overlay_mask(self, mask: Optional[np.ndarray]) -> None:
        self._overlay_mask = mask
        self._refresh_overlay()

    # --- output ---
    @property
    def current_index(self) -> int:
        return self._index

    def render(self) -> None:
        if self._render_window is not None:
            self._render_window.Render()

    # --- internals ---
    def _max_index(self) -> int:
        if self._volume is None:
            return 0
        sz, sy, sx = self._volume.shape
        if self.orientation is Orientation.AXIAL:
            return sz - 1
        if self.orientation is Orientation.CORONAL:
            return sy - 1
        return sx - 1

    def _refresh_image(self) -> None:
        if self._volume is None:
            return
        slice2d = self._volume.windowed(self.orientation, self._index, self._center, self._width)
        self._image_actor.SetInputData(_to_rgb_image(slice2d))

    def _refresh_overlay(self) -> None:
        if self._volume is None or self._overlay_mask is None:
            self._overlay_actor.SetInputData(_empty_image())
            return
        # Slice the mask the same way the volume is sliced.
        mask_volume = Volume(
            array=self._overlay_mask.astype(np.uint8),
            spacing_mm=self._volume.spacing_mm,
            modality=self._volume.modality,
        )
        slice2d = mask_volume.slice(self.orientation, self._index)
        self._overlay_actor.SetInputData(_to_rgba_overlay(slice2d))


def _to_rgb_image(slice2d: np.ndarray) -> vtk.vtkImageData:
    h, w = slice2d.shape
    rgb = np.stack([slice2d, slice2d, slice2d], axis=-1).astype(np.uint8)
    image = vtk.vtkImageData()
    image.SetDimensions(w, h, 1)
    vtk_array = numpy_support.numpy_to_vtk(
        rgb.reshape(-1, 3), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_array.SetNumberOfComponents(3)
    image.GetPointData().SetScalars(vtk_array)
    return image


def _to_rgba_overlay(slice2d: np.ndarray) -> vtk.vtkImageData:
    h, w = slice2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_bool = slice2d > 0
    rgba[mask_bool, 0] = 255   # red
    rgba[mask_bool, 3] = 96    # semi-transparent
    image = vtk.vtkImageData()
    image.SetDimensions(w, h, 1)
    vtk_array = numpy_support.numpy_to_vtk(
        rgba.reshape(-1, 4), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_array.SetNumberOfComponents(4)
    image.GetPointData().SetScalars(vtk_array)
    return image


def _empty_image() -> vtk.vtkImageData:
    image = vtk.vtkImageData()
    image.SetDimensions(1, 1, 1)
    arr = numpy_support.numpy_to_vtk(
        np.zeros((1, 4), dtype=np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    arr.SetNumberOfComponents(4)
    image.GetPointData().SetScalars(arr)
    return image
```

- [ ] **Step 5: Run tests and verify pass**

Run: `.venv/bin/pytest tests/rendering/test_slice_renderer.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/dicom_viewer/rendering/slice_renderer.py tests/rendering/
git commit -m "feat(rendering): add SliceRenderer with overlay support"
```

---

## Task 13: `rendering.VolumeRenderer`

**Files:**
- Create: `src/dicom_viewer/rendering/volume_renderer.py`
- Test: `tests/rendering/test_volume_renderer.py`

- [ ] **Step 1: Write the failing test `tests/rendering/test_volume_renderer.py`**

```python
import os

import numpy as np
import pytest

from dicom_viewer.core.region import Region
from dicom_viewer.core.volume import Volume
from dicom_viewer.rendering.volume_renderer import VolumeRenderer


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    monkeypatch.setenv("DICOM_VIEWER_OFFSCREEN", "1")


def _vol() -> Volume:
    arr = np.zeros((8, 8, 8), dtype=np.int16)
    arr[2:6, 2:6, 2:6] = 500
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_volume_renderer_smoke():
    r = VolumeRenderer()
    r.set_volume(_vol())
    r.render()


def test_volume_renderer_region_box():
    r = VolumeRenderer()
    r.set_volume(_vol())
    r.set_region(Region(z=(1, 5), y=(1, 5), x=(1, 5)))
    r.render()


def test_volume_renderer_handles_no_volume():
    r = VolumeRenderer()
    r.render()  # must not raise
```

- [ ] **Step 2: Run test, confirm failure**

Run: `.venv/bin/pytest tests/rendering/test_volume_renderer.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/dicom_viewer/rendering/volume_renderer.py`**

```python
"""3D volume rendering with optional segmentation overlay and region box."""
from __future__ import annotations

import os

import numpy as np
import vtk
from vtk.util import numpy_support  # type: ignore[import-untyped]

from dicom_viewer.core.region import Region
from dicom_viewer.core.volume import Volume


class VolumeRenderer:
    def __init__(self) -> None:
        self._volume: Volume | None = None
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.05, 0.05, 0.08)
        self._volume_actor: vtk.vtkVolume | None = None
        self._overlay_actor: vtk.vtkActor | None = None
        self._region_actor: vtk.vtkActor | None = None
        self._render_window: vtk.vtkRenderWindow | None = None

        if os.environ.get("DICOM_VIEWER_OFFSCREEN") == "1":
            rw = vtk.vtkRenderWindow()
            rw.SetOffScreenRendering(1)
            rw.AddRenderer(self._renderer)
            rw.SetSize(64, 64)
            self._render_window = rw

    def attach_render_window(self, render_window: vtk.vtkRenderWindow) -> None:
        render_window.AddRenderer(self._renderer)
        self._render_window = render_window

    def set_volume(self, volume: Volume) -> None:
        self._volume = volume
        if self._volume_actor is not None:
            self._renderer.RemoveVolume(self._volume_actor)
        image = _volume_to_vtk_image(volume)
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(image)
        prop = vtk.vtkVolumeProperty()
        opacity = vtk.vtkPiecewiseFunction()
        color = vtk.vtkColorTransferFunction()
        if volume.modality == "CT":
            # bone-emphasizing transfer function
            opacity.AddPoint(-1000, 0.0)
            opacity.AddPoint(150, 0.0)
            opacity.AddPoint(300, 0.5)
            opacity.AddPoint(1500, 0.9)
            color.AddRGBPoint(150, 0.4, 0.2, 0.1)
            color.AddRGBPoint(300, 0.9, 0.8, 0.7)
            color.AddRGBPoint(1500, 1.0, 1.0, 1.0)
        else:
            lo, hi = volume.intensity_range()
            opacity.AddPoint(lo, 0.0)
            opacity.AddPoint(lo + (hi - lo) * 0.3, 0.05)
            opacity.AddPoint(hi, 0.8)
            color.AddRGBPoint(lo, 0.1, 0.1, 0.2)
            color.AddRGBPoint(hi, 1.0, 1.0, 1.0)
        prop.SetColor(color)
        prop.SetScalarOpacity(opacity)
        prop.ShadeOn()
        actor = vtk.vtkVolume()
        actor.SetMapper(mapper)
        actor.SetProperty(prop)
        self._renderer.AddVolume(actor)
        self._volume_actor = actor
        self._renderer.ResetCamera()

    def set_region(self, region: Region) -> None:
        if self._region_actor is not None:
            self._renderer.RemoveActor(self._region_actor)
            self._region_actor = None
        if self._volume is None or region.is_empty:
            return
        sz, sy, sx = self._volume.spacing_mm
        cube = vtk.vtkCubeSource()
        cube.SetBounds(
            region.x[0] * sx,
            region.x[1] * sx,
            region.y[0] * sy,
            region.y[1] * sy,
            region.z[0] * sz,
            region.z[1] * sz,
        )
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(1.0, 0.8, 0.2)
        actor.GetProperty().SetLineWidth(2.0)
        self._renderer.AddActor(actor)
        self._region_actor = actor

    def set_overlay_mask(self, mask: np.ndarray | None) -> None:
        if self._overlay_actor is not None:
            self._renderer.RemoveActor(self._overlay_actor)
            self._overlay_actor = None
        if mask is None or self._volume is None or not mask.any():
            return
        image = _mask_to_vtk_image(mask, self._volume.spacing_mm)
        mc = vtk.vtkDiscreteMarchingCubes()
        mc.SetInputData(image)
        mc.SetValue(0, 1)
        mc.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(mc.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.95, 0.3, 0.3)
        actor.GetProperty().SetOpacity(0.6)
        self._renderer.AddActor(actor)
        self._overlay_actor = actor

    def render(self) -> None:
        if self._render_window is not None:
            self._render_window.Render()


def _volume_to_vtk_image(volume: Volume) -> vtk.vtkImageData:
    arr = volume.array
    z, y, x = arr.shape
    image = vtk.vtkImageData()
    image.SetDimensions(x, y, z)
    sz, sy, sx = volume.spacing_mm
    image.SetSpacing(sx, sy, sz)
    image.SetOrigin(0, 0, 0)
    vtk_type = vtk.VTK_SHORT if arr.dtype == np.int16 else vtk.VTK_FLOAT
    flat = numpy_support.numpy_to_vtk(arr.ravel(order="C"), deep=True, array_type=vtk_type)
    image.GetPointData().SetScalars(flat)
    return image


def _mask_to_vtk_image(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> vtk.vtkImageData:
    arr = mask.astype(np.uint8)
    z, y, x = arr.shape
    image = vtk.vtkImageData()
    image.SetDimensions(x, y, z)
    sz, sy, sx = spacing_mm
    image.SetSpacing(sx, sy, sz)
    image.SetOrigin(0, 0, 0)
    vtk_array = numpy_support.numpy_to_vtk(
        arr.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    image.GetPointData().SetScalars(vtk_array)
    return image
```

- [ ] **Step 4: Run tests and verify pass**

Run: `.venv/bin/pytest tests/rendering/test_volume_renderer.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dicom_viewer/rendering/volume_renderer.py tests/rendering/test_volume_renderer.py
git commit -m "feat(rendering): add VolumeRenderer with overlay and region box"
```

---

## Task 14: `rendering.MeshPreview`

**Files:**
- Create: `src/dicom_viewer/rendering/mesh_preview.py`
- Test: `tests/rendering/test_mesh_preview.py`

- [ ] **Step 1: Write the failing test `tests/rendering/test_mesh_preview.py`**

```python
import os

import numpy as np
import pytest

from dicom_viewer.core.mesh_export import ExportOptions, generate_mesh
from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.volume import Volume
from dicom_viewer.rendering.mesh_preview import MeshPreview


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    monkeypatch.setenv("DICOM_VIEWER_OFFSCREEN", "1")


def _cube_mesh():
    arr = np.zeros((16, 16, 16), dtype=np.int16)
    arr[4:12, 4:12, 4:12] = 1000
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    seg = threshold(v, low=500, high=2000)
    return generate_mesh(v, seg, region=v.bbox(), options=ExportOptions())


def test_mesh_preview_displays_mesh():
    mp = MeshPreview()
    mp.set_mesh(_cube_mesh())
    mp.render()


def test_mesh_preview_clears():
    mp = MeshPreview()
    mp.set_mesh(_cube_mesh())
    mp.set_mesh(None)
    mp.render()
```

- [ ] **Step 2: Run test, confirm failure**

Run: `.venv/bin/pytest tests/rendering/test_mesh_preview.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/dicom_viewer/rendering/mesh_preview.py`**

```python
"""Preview the generated mesh before STL export."""
from __future__ import annotations

import os

import vtk

from dicom_viewer.core.mesh_export import Mesh


class MeshPreview:
    def __init__(self) -> None:
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.1, 0.1, 0.12)
        self._actor: vtk.vtkActor | None = None
        self._render_window: vtk.vtkRenderWindow | None = None

        if os.environ.get("DICOM_VIEWER_OFFSCREEN") == "1":
            rw = vtk.vtkRenderWindow()
            rw.SetOffScreenRendering(1)
            rw.AddRenderer(self._renderer)
            rw.SetSize(64, 64)
            self._render_window = rw

    def attach_render_window(self, render_window: vtk.vtkRenderWindow) -> None:
        render_window.AddRenderer(self._renderer)
        self._render_window = render_window

    def set_mesh(self, mesh: Mesh | None) -> None:
        if self._actor is not None:
            self._renderer.RemoveActor(self._actor)
            self._actor = None
        if mesh is None:
            return
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh.polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.85, 0.85, 0.9)
        self._renderer.AddActor(actor)
        self._actor = actor
        self._renderer.ResetCamera()

    def render(self) -> None:
        if self._render_window is not None:
            self._render_window.Render()
```

- [ ] **Step 4: Run tests and verify pass**

Run: `.venv/bin/pytest tests/rendering/test_mesh_preview.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dicom_viewer/rendering/mesh_preview.py tests/rendering/test_mesh_preview.py
git commit -m "feat(rendering): add MeshPreview"
```

---

## Task 15: `ui.widgets.SliceView`

A single MPR-pane widget combining a `SliceRenderer` with a scrollbar and slice read-out.

**Files:**
- Create: `src/dicom_viewer/ui/widgets/__init__.py`
- Create: `src/dicom_viewer/ui/widgets/slice_view.py`
- Test: `tests/ui/__init__.py`, `tests/ui/test_slice_view.py`

- [ ] **Step 1: Create `src/dicom_viewer/ui/widgets/__init__.py`** (one newline)

- [ ] **Step 2: Create `tests/ui/__init__.py`** (one newline)

- [ ] **Step 3: Write the failing test `tests/ui/test_slice_view.py`**

```python
import numpy as np
import pytest

from dicom_viewer.core.volume import Orientation, Volume
from dicom_viewer.ui.widgets.slice_view import SliceView


@pytest.fixture
def vol() -> Volume:
    arr = np.zeros((6, 6, 6), dtype=np.int16)
    arr[1:5, 1:5, 1:5] = 500
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_slice_view_scrollbar_range_reflects_volume(qtbot, vol):
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    assert view.scrollbar.minimum() == 0
    assert view.scrollbar.maximum() == 5  # axial: z=6 -> max 5


def test_slice_view_scrollbar_updates_index(qtbot, vol):
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    view.scrollbar.setValue(3)
    assert view.current_index == 3


def test_slice_view_emits_slice_changed_signal(qtbot, vol):
    view = SliceView(orientation=Orientation.AXIAL)
    qtbot.addWidget(view)
    view.set_volume(vol)
    with qtbot.waitSignal(view.slice_changed, timeout=500) as blocker:
        view.scrollbar.setValue(4)
    assert blocker.args == [4]
```

- [ ] **Step 4: Run test, confirm failure**

Run: `.venv/bin/pytest tests/ui/test_slice_view.py -v`
Expected: `ImportError`.

- [ ] **Step 5: Implement `src/dicom_viewer/ui/widgets/slice_view.py`**

```python
"""SliceView — a QWidget showing one MPR slice with a scrollbar."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QScrollBar, QVBoxLayout, QWidget

from dicom_viewer.core.volume import Orientation, Volume
from dicom_viewer.rendering.slice_renderer import SliceRenderer


class SliceView(QWidget):
    slice_changed = pyqtSignal(int)

    def __init__(self, orientation: Orientation) -> None:
        super().__init__()
        self.orientation = orientation
        self._volume: Volume | None = None
        self._renderer = SliceRenderer(orientation=orientation)

        try:
            from vtkmodules.qt.QVTKRenderWindowInteractor import (  # type: ignore[import-untyped]
                QVTKRenderWindowInteractor,
            )
            self._vtk_widget = QVTKRenderWindowInteractor(self)
            self._renderer.attach_render_window(self._vtk_widget.GetRenderWindow())
        except Exception:
            # Headless test environment: skip VTK widget instantiation.
            self._vtk_widget = QLabel("[vtk render area]")  # type: ignore[assignment]

        self.scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(0)
        self.scrollbar.valueChanged.connect(self._on_scroll)

        self._label = QLabel(f"{orientation.value} — 0 / 0")

        row = QHBoxLayout()
        row.addWidget(self._vtk_widget, stretch=1)
        row.addWidget(self.scrollbar)
        layout = QVBoxLayout(self)
        layout.addLayout(row, stretch=1)
        layout.addWidget(self._label)

    def set_volume(self, volume: Volume) -> None:
        self._volume = volume
        self._renderer.set_volume(volume)
        max_index = self._max_index()
        self.scrollbar.setMaximum(max_index)
        self.scrollbar.setValue(max_index // 2)
        self._update_label()
        if hasattr(self._vtk_widget, "GetRenderWindow"):
            self._renderer.render()

    def set_windowing(self, center: float, width: float) -> None:
        self._renderer.set_windowing(center, width)
        if hasattr(self._vtk_widget, "GetRenderWindow"):
            self._renderer.render()

    def set_overlay_mask(self, mask: np.ndarray | None) -> None:
        self._renderer.set_overlay_mask(mask)
        if hasattr(self._vtk_widget, "GetRenderWindow"):
            self._renderer.render()

    @property
    def current_index(self) -> int:
        return int(self.scrollbar.value())

    def _on_scroll(self, value: int) -> None:
        self._renderer.set_slice_index(value)
        self._update_label()
        if hasattr(self._vtk_widget, "GetRenderWindow"):
            self._renderer.render()
        self.slice_changed.emit(value)

    def _max_index(self) -> int:
        if self._volume is None:
            return 0
        sz, sy, sx = self._volume.shape
        if self.orientation is Orientation.AXIAL:
            return sz - 1
        if self.orientation is Orientation.CORONAL:
            return sy - 1
        return sx - 1

    def _update_label(self) -> None:
        self._label.setText(
            f"{self.orientation.value} — {self.current_index} / {self._max_index()}"
        )
```

- [ ] **Step 6: Run tests and verify pass**

Run: `.venv/bin/pytest tests/ui/test_slice_view.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/dicom_viewer/ui/widgets tests/ui/__init__.py tests/ui/test_slice_view.py
git commit -m "feat(ui): add SliceView widget"
```

---

## Task 16: `ui.panels.windowing` + presets

**Files:**
- Create: `src/dicom_viewer/ui/panels/__init__.py`
- Create: `src/dicom_viewer/ui/panels/windowing.py`
- Test: `tests/ui/test_windowing_panel.py`

- [ ] **Step 1: Create `src/dicom_viewer/ui/panels/__init__.py`** (one newline)

- [ ] **Step 2: Write the failing test `tests/ui/test_windowing_panel.py`**

```python
import numpy as np
import pytest

from dicom_viewer.core.document import Document
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume
from dicom_viewer.ui.panels.windowing import WindowingPanel


@pytest.fixture
def doc_ct() -> Document:
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    study = Study(
        volume=v,
        patient_id="P",
        patient_name="N",
        study_uid="s",
        series_uid="se",
        series_description="d",
        orientation_cosines=(1, 0, 0, 0, 1, 0),
    )
    doc = Document()
    doc.set_study(study)
    return doc


def test_windowing_panel_ct_presets_present(qtbot, doc_ct):
    panel = WindowingPanel(doc_ct)
    qtbot.addWidget(panel)
    items = [panel.preset_combo.itemText(i) for i in range(panel.preset_combo.count())]
    assert "Bone" in items
    assert "Soft Tissue" in items
    assert "Lung" in items
    assert "Brain" in items


def test_windowing_preset_updates_document(qtbot, doc_ct):
    panel = WindowingPanel(doc_ct)
    qtbot.addWidget(panel)
    panel.apply_preset("Bone")
    assert doc_ct.windowing.center == 400
    assert doc_ct.windowing.width == 1500


def test_windowing_sliders_drive_document(qtbot, doc_ct):
    panel = WindowingPanel(doc_ct)
    qtbot.addWidget(panel)
    panel.center_slider.setValue(50)
    panel.width_slider.setValue(700)
    assert doc_ct.windowing.center == 50
    assert doc_ct.windowing.width == 700
```

- [ ] **Step 3: Run test, confirm failure**

Run: `.venv/bin/pytest tests/ui/test_windowing_panel.py -v`
Expected: `ImportError`.

- [ ] **Step 4: Implement `src/dicom_viewer/ui/panels/windowing.py`**

```python
"""Window/Level panel — sliders + modality-aware presets."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.document import Document, WindowingState

_CT_PRESETS: dict[str, tuple[int, int]] = {
    "Bone": (400, 1500),
    "Soft Tissue": (40, 400),
    "Lung": (-600, 1500),
    "Brain": (40, 80),
}


class WindowingPanel(QWidget):
    def __init__(self, document: Document) -> None:
        super().__init__()
        self._document = document
        self._building = False

        self.preset_combo = QComboBox()
        self._refresh_presets()
        self.preset_combo.activated.connect(
            lambda _i: self.apply_preset(self.preset_combo.currentText())
        )

        self.center_slider = QSlider(Qt.Orientation.Horizontal)
        self.center_slider.setRange(-1024, 4096)
        self.center_slider.valueChanged.connect(self._on_slider_changed)

        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider.setRange(1, 8192)
        self.width_slider.valueChanged.connect(self._on_slider_changed)

        self._readout = QLabel()

        form = QFormLayout()
        form.addRow("Preset", self.preset_combo)
        form.addRow("Center", self.center_slider)
        form.addRow("Width", self.width_slider)
        form.addRow(self._readout)
        layout = QVBoxLayout(self)
        layout.addLayout(form)

        document.subscribe(self._on_doc_event)
        self._sync_from_document()

    def apply_preset(self, name: str) -> None:
        if name not in _CT_PRESETS:
            return
        c, w = _CT_PRESETS[name]
        self._document.set_windowing(WindowingState(center=float(c), width=float(w)))

    def _on_slider_changed(self, _value: int) -> None:
        if self._building:
            return
        self._document.set_windowing(
            WindowingState(
                center=float(self.center_slider.value()),
                width=float(self.width_slider.value()),
            )
        )

    def _on_doc_event(self, kind: str) -> None:
        if kind in ("study", "windowing"):
            self._sync_from_document()
        if kind == "study":
            self._refresh_presets()

    def _refresh_presets(self) -> None:
        self.preset_combo.clear()
        if self._document.volume and self._document.volume.modality == "CT":
            for name in _CT_PRESETS:
                self.preset_combo.addItem(name)
        else:
            self.preset_combo.addItem("Auto (MRI)")

    def _sync_from_document(self) -> None:
        self._building = True
        try:
            w = self._document.windowing
            self.center_slider.setValue(int(round(w.center)))
            self.width_slider.setValue(int(round(w.width)))
            self._readout.setText(f"C={w.center:.0f} W={w.width:.0f}")
        finally:
            self._building = False
```

- [ ] **Step 5: Run tests and verify pass**

Run: `.venv/bin/pytest tests/ui/test_windowing_panel.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/dicom_viewer/ui/panels tests/ui/test_windowing_panel.py
git commit -m "feat(ui): add windowing panel with CT presets"
```

---

## Task 17: `ui.panels.segmentation`

**Files:**
- Create: `src/dicom_viewer/ui/panels/segmentation.py`
- Test: `tests/ui/test_segmentation_panel.py`

- [ ] **Step 1: Write the failing test `tests/ui/test_segmentation_panel.py`**

```python
import numpy as np
import pytest

from dicom_viewer.core.document import Document
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume
from dicom_viewer.ui.panels.segmentation import SegmentationPanel


@pytest.fixture
def doc() -> Document:
    arr = np.zeros((6, 6, 6), dtype=np.int16)
    arr[1:5, 1:5, 1:5] = 500
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    study = Study(
        volume=v,
        patient_id="P",
        patient_name="N",
        study_uid="s",
        series_uid="se",
        series_description="d",
        orientation_cosines=(1, 0, 0, 0, 1, 0),
    )
    document = Document()
    document.set_study(study)
    return document


def test_apply_threshold_writes_segmentation_to_document(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.low_spin.setValue(100)
    panel.high_spin.setValue(1000)
    panel.apply_button.click()
    assert doc.segmentation is not None
    assert doc.segmentation.method.startswith("threshold")
    assert doc.segmentation.voxel_count > 0


def test_keep_largest_component_chains(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.low_spin.setValue(100)
    panel.high_spin.setValue(1000)
    panel.largest_component_checkbox.setChecked(True)
    panel.apply_button.click()
    assert "largest_component" in doc.segmentation.method


def test_smooth_chains_after_apply(qtbot, doc):
    panel = SegmentationPanel(doc)
    qtbot.addWidget(panel)
    panel.low_spin.setValue(100)
    panel.high_spin.setValue(1000)
    panel.smooth_checkbox.setChecked(True)
    panel.apply_button.click()
    assert doc.segmentation.method.endswith("+smooth")
```

- [ ] **Step 2: Run test, confirm failure**

Run: `.venv/bin/pytest tests/ui/test_segmentation_panel.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/dicom_viewer/ui/panels/segmentation.py`**

```python
"""Segmentation panel — threshold + region-grow methods with chained refinements."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.document import Document
from dicom_viewer.core.segmentation.morphology import keep_largest_component, smooth_mask
from dicom_viewer.core.segmentation.region_grow import region_grow
from dicom_viewer.core.segmentation.threshold import threshold


class SegmentationPanel(QWidget):
    def __init__(self, document: Document) -> None:
        super().__init__()
        self._document = document

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Threshold", "Region grow"])

        self.low_spin = QSpinBox()
        self.low_spin.setRange(-2000, 10000)
        self.low_spin.setValue(300)

        self.high_spin = QSpinBox()
        self.high_spin.setRange(-2000, 10000)
        self.high_spin.setValue(3000)

        self.seed_z = QSpinBox(); self.seed_z.setRange(0, 100000)
        self.seed_y = QSpinBox(); self.seed_y.setRange(0, 100000)
        self.seed_x = QSpinBox(); self.seed_x.setRange(0, 100000)
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(0, 10000)
        self.tolerance_spin.setValue(100)

        self.largest_component_checkbox = QCheckBox("Keep largest connected component")
        self.largest_component_checkbox.setChecked(True)
        self.smooth_checkbox = QCheckBox("Smooth mask (close + open)")
        self.smooth_checkbox.setChecked(False)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._on_apply)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(lambda: document.set_segmentation(None))

        self._status = QLabel("No segmentation")

        form = QFormLayout()
        form.addRow("Method", self.method_combo)
        form.addRow("Low", self.low_spin)
        form.addRow("High", self.high_spin)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("seed z/y/x:"))
        seed_row.addWidget(self.seed_z)
        seed_row.addWidget(self.seed_y)
        seed_row.addWidget(self.seed_x)
        form.addRow(seed_row)
        form.addRow("Tolerance", self.tolerance_spin)
        form.addRow(self.largest_component_checkbox)
        form.addRow(self.smooth_checkbox)

        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.clear_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self._status)

        document.subscribe(self._on_doc_event)

    def _on_apply(self) -> None:
        volume = self._document.volume
        if volume is None:
            return
        method = self.method_combo.currentText()
        if method == "Threshold":
            seg = threshold(volume, self.low_spin.value(), self.high_spin.value())
        else:
            seg = region_grow(
                volume,
                seed=(self.seed_z.value(), self.seed_y.value(), self.seed_x.value()),
                tolerance=self.tolerance_spin.value(),
            )
        if self.largest_component_checkbox.isChecked():
            seg = keep_largest_component(seg)
        if self.smooth_checkbox.isChecked():
            seg = smooth_mask(seg, iterations=1)
        self._document.set_segmentation(seg)

    def _on_doc_event(self, kind: str) -> None:
        if kind == "segmentation":
            seg = self._document.segmentation
            if seg is None:
                self._status.setText("No segmentation")
            else:
                self._status.setText(f"{seg.method} — {seg.voxel_count} voxels")
```

- [ ] **Step 4: Run tests and verify pass**

Run: `.venv/bin/pytest tests/ui/test_segmentation_panel.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dicom_viewer/ui/panels/segmentation.py tests/ui/test_segmentation_panel.py
git commit -m "feat(ui): add segmentation panel (threshold + region-grow)"
```

---

## Task 18: `ui.panels.export`

**Files:**
- Create: `src/dicom_viewer/ui/panels/export.py`
- Test: `tests/ui/test_export_panel.py`

- [ ] **Step 1: Write the failing test `tests/ui/test_export_panel.py`**

```python
import numpy as np
import pytest

from dicom_viewer.core.document import Document
from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume
from dicom_viewer.ui.panels.export import ExportPanel


def _doc_with_segmentation() -> Document:
    arr = np.zeros((16, 16, 16), dtype=np.int16)
    arr[4:12, 4:12, 4:12] = 1000
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    study = Study(
        volume=v,
        patient_id="P",
        patient_name="N",
        study_uid="s",
        series_uid="se",
        series_description="cube",
        orientation_cosines=(1, 0, 0, 0, 1, 0),
    )
    doc = Document()
    doc.set_study(study)
    doc.set_segmentation(threshold(v, low=500, high=2000))
    return doc


def test_export_panel_disabled_without_segmentation(qtbot):
    doc = Document()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    assert not panel.export_button.isEnabled()


def test_export_panel_enabled_with_segmentation(qtbot):
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    assert panel.export_button.isEnabled()


def test_export_writes_stl_file(qtbot, tmp_path):
    doc = _doc_with_segmentation()
    panel = ExportPanel(doc)
    qtbot.addWidget(panel)
    out = tmp_path / "out.stl"
    panel.run_export(out)  # synchronous helper used by the button slot
    assert out.exists()
    assert out.stat().st_size > 84
```

- [ ] **Step 2: Run test, confirm failure**

Run: `.venv/bin/pytest tests/ui/test_export_panel.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `src/dicom_viewer/ui/panels/export.py`**

```python
"""Export panel — smoothing / decimation options + STL file dialog."""
from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.document import Document
from dicom_viewer.core.mesh_export import EmptyMeshError, ExportOptions, export_stl, generate_mesh


class _ExportWorker(QThread):
    finished_ok = pyqtSignal(str, int)
    failed = pyqtSignal(str)

    def __init__(self, document: Document, options: ExportOptions, out_path: Path) -> None:
        super().__init__()
        self._document = document
        self._options = options
        self._out_path = out_path

    def run(self) -> None:
        try:
            volume = self._document.volume
            seg = self._document.segmentation
            region = self._document.region or (volume.bbox() if volume else None)
            if volume is None or seg is None or region is None:
                raise EmptyMeshError("missing volume / segmentation / region")
            mesh = generate_mesh(volume, seg, region, self._options)
            export_stl(mesh, self._out_path)
            self.finished_ok.emit(str(self._out_path), mesh.triangle_count)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ExportPanel(QWidget):
    def __init__(self, document: Document) -> None:
        super().__init__()
        self._document = document

        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(0, 200)
        self.smoothing_spin.setValue(15)

        self.decimation_spin = QDoubleSpinBox()
        self.decimation_spin.setRange(0.0, 0.95)
        self.decimation_spin.setSingleStep(0.05)
        self.decimation_spin.setValue(0.5)

        self.manifold_checkbox = QCheckBox("Ensure manifold (recommended)")
        self.manifold_checkbox.setChecked(True)

        self.export_button = QPushButton("Export STL…")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setEnabled(False)

        self._status = QLabel("")

        form = QFormLayout()
        form.addRow("Smoothing iterations", self.smoothing_spin)
        form.addRow("Decimation reduction", self.decimation_spin)
        form.addRow(self.manifold_checkbox)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.export_button)
        layout.addWidget(self._status)

        document.subscribe(self._on_doc_event)

    def _on_doc_event(self, kind: str) -> None:
        if kind in ("segmentation", "study"):
            self.export_button.setEnabled(self._document.segmentation is not None)

    def _on_export_clicked(self) -> None:
        out_str, _ = QFileDialog.getSaveFileName(
            self, "Export STL", self._suggested_filename(), "STL files (*.stl)"
        )
        if not out_str:
            return
        self.run_export(Path(out_str))

    def run_export(self, out_path: Path) -> None:
        """Synchronous export entry point — used by the button and by tests."""
        options = ExportOptions(
            smoothing_iterations=self.smoothing_spin.value(),
            decimation_target_reduction=float(self.decimation_spin.value()),
            ensure_manifold=self.manifold_checkbox.isChecked(),
        )
        worker = _ExportWorker(self._document, options, out_path)
        worker.finished_ok.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.start()
        worker.wait()  # synchronous; UI variant runs it async via QThread normally.

    def _on_done(self, path: str, triangle_count: int) -> None:
        self._status.setText(f"Wrote {path} ({triangle_count} triangles)")

    def _on_failed(self, msg: str) -> None:
        self._status.setText(f"Export failed: {msg}")
        QMessageBox.critical(self, "Export failed", msg)

    def _suggested_filename(self) -> str:
        study = self._document.study
        if study is None:
            return "export.stl"
        method = self._document.segmentation.method if self._document.segmentation else "raw"
        raw = f"{study.patient_id or 'anon'}_{study.series_description or 'series'}_{method}"
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
        return f"{sanitized}.stl"
```

- [ ] **Step 4: Run tests and verify pass**

Run: `.venv/bin/pytest tests/ui/test_export_panel.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dicom_viewer/ui/panels/export.py tests/ui/test_export_panel.py
git commit -m "feat(ui): add export panel with threaded STL writer"
```

---

## Task 19: `ui.MainWindow`, `app.py`, `__main__.py`

Wires everything into a window the user can actually launch.

**Files:**
- Create: `src/dicom_viewer/ui/main_window.py`
- Create: `src/dicom_viewer/app.py`
- Create: `src/dicom_viewer/__main__.py`

- [ ] **Step 1: Implement `src/dicom_viewer/ui/main_window.py`**

```python
"""MainWindow — four-pane MPR + 3D layout with side dock for panels."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QGridLayout,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QWidget,
)

from dicom_viewer.core.document import Document
from dicom_viewer.core.volume import Orientation
from dicom_viewer.io.dicom_loader import LoaderError, load_series_from_folder
from dicom_viewer.ui.panels.export import ExportPanel
from dicom_viewer.ui.panels.segmentation import SegmentationPanel
from dicom_viewer.ui.panels.windowing import WindowingPanel
from dicom_viewer.ui.widgets.slice_view import SliceView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DICOM Viewer")
        self.resize(1400, 900)

        self.document = Document()

        self.axial = SliceView(Orientation.AXIAL)
        self.coronal = SliceView(Orientation.CORONAL)
        self.sagittal = SliceView(Orientation.SAGITTAL)
        # The 3D view is a placeholder QWidget; full VolumeRenderer is wired in via the
        # rendering layer in a later iteration. For groundwork, the four-pane layout
        # uses the three MPRs plus a labelled placeholder pane.
        self.placeholder_3d = QWidget()

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.addWidget(self.axial, 0, 0)
        grid.addWidget(self.coronal, 0, 1)
        grid.addWidget(self.sagittal, 1, 0)
        grid.addWidget(self.placeholder_3d, 1, 1)
        self.setCentralWidget(grid_host)

        tabs = QTabWidget()
        tabs.addTab(WindowingPanel(self.document), "Windowing")
        tabs.addTab(SegmentationPanel(self.document), "Segmentation")
        tabs.addTab(ExportPanel(self.document), "Export")
        dock = QDockWidget("Tools", self)
        dock.setWidget(tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        open_action = QAction("Open DICOM Folder…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_folder)
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_action)

        self.document.subscribe(self._on_doc_event)

    def _on_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open DICOM Folder")
        if not folder:
            return
        try:
            result = load_series_from_folder(Path(folder))
        except LoaderError as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        if len(result.studies) == 1:
            chosen = result.studies[0]
        else:
            items = [s.display_name for s in result.studies]
            picked, ok = QInputDialog.getItem(
                self, "Pick a series", "Multiple series found:", items, 0, False
            )
            if not ok:
                return
            chosen = result.studies[items.index(picked)]
        self.document.set_study(chosen)

    def _on_doc_event(self, kind: str) -> None:
        volume = self.document.volume
        if volume is None:
            return
        if kind == "study":
            self.axial.set_volume(volume)
            self.coronal.set_volume(volume)
            self.sagittal.set_volume(volume)
        if kind in ("study", "windowing"):
            w = self.document.windowing
            self.axial.set_windowing(w.center, w.width)
            self.coronal.set_windowing(w.center, w.width)
            self.sagittal.set_windowing(w.center, w.width)
        if kind == "segmentation":
            mask = self.document.segmentation.mask if self.document.segmentation else None
            self.axial.set_overlay_mask(mask)
            self.coronal.set_overlay_mask(mask)
            self.sagittal.set_overlay_mask(mask)
```

- [ ] **Step 2: Implement `src/dicom_viewer/app.py`**

```python
"""Application entry point."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from dicom_viewer.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Implement `src/dicom_viewer/__main__.py`**

```python
from dicom_viewer.app import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Smoke-test the app launches (offscreen) without errors**

Run:
```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m dicom_viewer &
APP_PID=$!
sleep 2
kill -TERM $APP_PID 2>/dev/null || true
wait $APP_PID 2>/dev/null || true
echo "exit=$?"
```
Expected: process starts, runs for ~2 seconds, is terminated. No `Traceback` output prior to termination.

- [ ] **Step 5: Commit**

```bash
git add src/dicom_viewer/ui/main_window.py src/dicom_viewer/app.py src/dicom_viewer/__main__.py
git commit -m "feat(ui): add MainWindow, app entry point, and four-pane layout"
```

---

## Task 20: End-to-end integration test

Load a synthetic DICOM series, threshold-segment, crop to a region, export STL — entire pipeline through the public API of `core/` and `io/`.

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_end_to_end.py`

- [ ] **Step 1: Create `tests/integration/__init__.py`** (one newline)

- [ ] **Step 2: Write the integration test**

```python
"""End-to-end: synthetic DICOM folder → loaded study → segmentation → STL on disk."""
import struct

from dicom_viewer.core.mesh_export import ExportOptions, export_stl, generate_mesh
from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.morphology import (
    keep_largest_component,
    smooth_mask,
)
from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.io.dicom_loader import load_series_from_folder
from tests.fixtures.make_synthetic_series import make_synthetic_ct_series


def test_full_pipeline_produces_valid_stl(tmp_path):
    series_dir = make_synthetic_ct_series(
        tmp_path, shape=(20, 32, 32), spacing=(1.0, 1.0, 1.0)
    )
    loaded = load_series_from_folder(series_dir)
    assert len(loaded.studies) == 1
    volume = loaded.studies[0].volume

    # CT pixels were 0/1000 raw, intercept -1024 -> -1024 / -24 HU.
    seg = threshold(volume, low=-100, high=10000)
    seg = keep_largest_component(seg)
    seg = smooth_mask(seg, iterations=1)
    assert seg.voxel_count > 0

    # Crop to the upper half of the volume.
    region = Region(z=(0, 10), y=(0, 32), x=(0, 32))

    mesh = generate_mesh(volume, seg, region, ExportOptions(smoothing_iterations=5))
    assert mesh.triangle_count > 0

    out = tmp_path / "result.stl"
    export_stl(mesh, out)
    data = out.read_bytes()
    assert len(data) >= 84
    n_triangles = struct.unpack("<I", data[80:84])[0]
    assert n_triangles == mesh.triangle_count

    # Mesh bounds (z) must fall inside the cropped region.
    (lo_z, _, _), (hi_z, _, _) = mesh.bounds_mm
    assert hi_z <= 10.0 + 1e-3
    assert lo_z >= 0.0 - 1e-3
```

- [ ] **Step 3: Run the integration test and verify pass**

Run: `.venv/bin/pytest tests/integration/ -v`
Expected: 1 passed.

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Final commit**

```bash
git add tests/integration/
git commit -m "test: add end-to-end DICOM-to-STL integration test"
```

---

## Self-review (run during plan-write, after the plan is complete)

**1. Spec coverage**
- Load folder, group by SeriesInstanceUID, sort by ImagePositionPatient → Task 6.
- CT rescale to HU; MRI raw float → Task 6.
- Volume + MPR slicing + windowing → Tasks 4, 12, 15.
- Document model with observers → Task 11.
- Threshold + largest-component + smoothing → Tasks 7, 8.
- Region grow (MRI) → Task 9.
- Region (axis-aligned bbox) → Task 3; used by mesh export Task 10 and main window indirectly.
- 3D volume render → Task 13 (rendering layer exists; the four-pane window currently uses a placeholder for the 3D pane — wiring it in is a stretch step the implementing engineer can do in Task 19 if time allows; the spec explicitly calls out "groundwork", and all the pipeline math is exercised by Task 20).
- Mesh preview → Task 14.
- Marching cubes + smoothing + decimation + manifold fix + binary STL → Task 10.
- UI panels (windowing, segmentation, export) → Tasks 16, 17, 18.
- Threaded export → Task 18 (`_ExportWorker`).
- Error handling (empty folder, mixed series, compressed transfer syntax, empty mask) → Tasks 6, 7, 10.
- Tests at all layers → present in every task; explicit integration test in Task 20.

**2. Placeholder scan** — no TBDs, all code blocks contain real code, no "implement later" steps.

**3. Type consistency** — `Segmentation`, `Region`, `Volume`, `Study`, `Document`, `Mesh`, `ExportOptions` are defined once and referenced with the same signatures everywhere.

**4. Known limitations the engineer should be aware of**
- The 3D pane in `MainWindow` is a placeholder `QWidget`; `VolumeRenderer` is fully implemented (Task 13) and ready to be wired in. Reason for the gap: wiring `QVTKRenderWindowInteractor` plus a `vtkBoxWidget2` into the live layout has many moving Qt-event-loop pieces; pulling it into a single task would have produced an oversized step. Adding it after Task 19 is a small, isolated change.
- 2D crosshairs and 2D region rectangles are not in this plan. They are stretch UX polish that can be added in a follow-up plan once the basic four-pane + segmentation + export flow is verified end-to-end.
