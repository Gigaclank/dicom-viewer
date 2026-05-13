# DICOM Viewer with 3D-Printable Section Export — Design

**Date:** 2026-05-13
**Status:** Approved for implementation

## Goal

Build a desktop application that loads DICOM scans (CT and MRI), lets the user view them in standard radiology multi-planar layout, and exports a chosen anatomical section as a 3D-printable STL file.

The primary user is the project owner working with their own scans for personal use. This is not a clinical tool.

## Scope

**In scope (the "groundwork" build):**

- Load a DICOM folder, pick a series if multiple are present, build a 3D volume.
- Four-pane viewer: axial, coronal, sagittal MPR + 3D volume render.
- Windowing controls with modality-aware presets.
- Threshold-based segmentation with live preview, including largest-component filtering.
- Region-grow segmentation for cases where threshold alone is insufficient.
- Axis-aligned 3D region (bounding-box) tool for selecting a section to export.
- Marching-cubes mesh generation with smoothing and decimation options.
- STL export (binary), with mesh preview.

**Out of scope (explicit non-goals):**

- Manual paint/brush segmentation editor.
- Multi-label segmentations or DICOM-SEG export.
- Network / PACS connectivity.
- Oblique (rotatable, non-axis-aligned) region cuts.
- AI/ML segmentation.
- Clinical-grade accuracy guarantees.

## Platform & stack

- Python 3.11+, managed by `uv`.
- GUI: **PyQt6**.
- Rendering and mesh pipeline: **VTK** (`vtkImageReslice`, `vtkSmartVolumeMapper`, `vtkDiscreteMarchingCubes`, `vtkWindowedSincPolyDataFilter`, `vtkQuadricDecimation`, `vtkSTLWriter`, `vtkBoxWidget2`).
- DICOM parsing: **pydicom**, with `pylibjpeg` + `pylibjpeg-libjpeg` + `pylibjpeg-openjpeg` for compressed transfer syntaxes.
- Numerics: **numpy**, **scipy** (`scipy.ndimage` for labeling and morphology).
- Segmentation: **SimpleITK** for `ConnectedThreshold` region growing.
- Dev: `pytest`, `pytest-qt`, `ruff`, `mypy`.

## Architecture

Three layers with strict, one-way dependencies: `ui` depends on `core` and `rendering`; `rendering` depends on `core`; `core` depends on nothing in this project (and crucially does not import Qt or VTK).

```
ui  ──►  rendering  ──►  core
                          ▲
                io ───────┘
```

### Layer 1 — `core/` (pure Python)

Domain model. No Qt, no VTK imports anywhere in this package. Trivially unit-testable.

- **`Study`** — a loaded DICOM series. Holds metadata (modality, patient orientation cosines, voxel spacing `(z, y, x)` in mm, intensity range, study/series descriptions) and the 3D voxel ndarray (`int16` for CT after rescale, `float32` for MRI).
- **`Volume`** — read-only view over a `Study`'s voxel array.
  - `slice(orientation, index) -> ndarray` — axial / coronal / sagittal.
  - `windowed(center, width) -> ndarray[uint8]` — for display.
  - `bbox() -> Region` — full extent.
  - `crop(region) -> Volume` — cheap view, no copy.
- **`Region`** — axis-aligned 3D bounding box in voxel space. Supports clipping and intersection.
- **`Segmentation`** — a boolean 3D mask plus provenance (source volume, method name, parameters). Supports intersection with a `Region`.
- **`segmentation/`** — submodule containing one function per method:
  - `threshold(volume, low, high) -> Segmentation`
  - `threshold_with_largest_component(volume, low, high) -> Segmentation` (uses `scipy.ndimage.label`).
  - `region_grow(volume, seed_voxel, tolerance) -> Segmentation` (uses `SimpleITK.ConnectedThreshold`).
  - `smooth_mask(seg, iterations) -> Segmentation` — binary closing + opening to remove specks and fill pinholes prior to meshing.
- **`MeshExporter`** — runs the marching-cubes → smoothing → decimation → STL pipeline on `(Volume, Segmentation, Region)`.
- **`Document`** — the single source of truth holding the active `Study`, `Volume`, `Segmentation`, `Region`, and windowing state. Exposes a plain-Python observer interface (callback registration). A Qt adapter in the `ui` layer translates those callbacks into Qt signals.

### Layer 2 — `rendering/` (VTK adapters)

- **`SliceRenderer`** — drives a `QVTKRenderWindowInteractor` showing one MPR view for a given `Volume` and orientation. Handles crosshair drawing and segmentation overlay.
- **`VolumeRenderer`** — 3D volume render with segmentation overlay and an interactive `vtkBoxWidget2` for the region. Default transfer function is modality-aware (bone ramp for CT, generic intensity ramp for MRI).
- **`MeshPreview`** — preview of the generated mesh before export, with triangle count and bounding-box dimensions in mm.

### Layer 3 — `ui/` (PyQt6)

- **`MainWindow`** — four-pane layout (axial / coronal / sagittal / 3D), menu bar, status bar, side dock with panels.
- **Panels:** `WindowingPanel`, `SegmentationPanel`, `ExportPanel`.
- **`SliceView`** widget — combines a `SliceRenderer` with a slice-index scrollbar, slice-position read-out, and per-pane right-click-drag windowing handler.

The UI does not own state directly. It reads from and dispatches to the `Document`. Renderers observe the `Document` and re-render on change.

### Cross-cutting — `io/`

- **`dicom_loader.py`** — walks a folder with `pydicom`, groups by `SeriesInstanceUID`, surfaces a series picker if more than one, sorts the chosen series by `ImagePositionPatient` projected on the slice-normal axis, applies `RescaleSlope`/`RescaleIntercept` for CT, computes slice spacing from positions (not `SliceThickness`).

## Data flow (end-to-end)

1. User drags a folder onto the window (or `File → Open DICOM Folder`).
2. `dicom_loader.load_folder(path)` returns one or more candidate series with summary metadata.
3. If multiple series, a series picker dialog is shown; the user picks one.
4. The chosen series becomes a `Study`, then the active `Volume` on the `Document`. All four views re-render.
5. The user adjusts windowing (sliders or right-click-drag on a pane). The `Document`'s window state updates; views re-render.
6. The user opens the segmentation panel, picks `threshold` (default), and adjusts low/high sliders. A `Segmentation` is computed at downsampled resolution for live preview (~100 ms target) and shown as an overlay in the 3D and 2D views.
7. The user toggles "smooth mask" if needed. Optionally, they switch method to `region_grow` and click a 2D pane to set the seed.
8. The user enables Region Mode. A `vtkBoxWidget2` appears in 3D; matching draggable rectangles appear in the three MPR panes. The user shapes the region until it surrounds the section they want.
9. The user clicks "Export STL…". The export panel shows smoothing and decimation options. On confirm, the export runs in a `QThread`:
   - Compute full-resolution segmentation.
   - Intersect mask with region; crop volume to region.
   - Run `vtkDiscreteMarchingCubes`.
   - Run `vtkWindowedSincPolyDataFilter` (smoothing iterations from the dialog).
   - Run `vtkQuadricDecimation` (target reduction from the dialog).
   - Optionally run `vtkFillHolesFilter` + recompute normals for manifold output.
   - Write binary STL via `vtkSTLWriter`.
10. The `MeshPreview` updates with the result. The user accepts and the file is saved (default name: `PatientID_SeriesDescription_method.stl`, sanitized).

## Viewing details

**Four-pane MPR + 3D layout**
- Top-left axial, top-right coronal, bottom-left sagittal, bottom-right 3D.
- Linked crosshairs: clicking in any 2D pane moves the other two to that voxel.
- Mouse-wheel scrolls slices in the focused pane; `Shift+wheel` is ±10.
- Right-click-drag adjusts window center/width on any 2D pane.

**Windowing**
- CT presets: Bone (C=400, W=1500), Soft Tissue (40/400), Lung (-600/1500), Brain (40/80).
- MRI: presets computed from intensity percentiles (e.g. p1 and p99). "Reset" recomputes them.

**3D render performance**
- The live 3D render uses a downsampled volume (max side length 256 by default) for interactivity.
- Marching-cubes / STL export always uses full resolution.
- Toggle exposed to render full-resolution for a stationary inspection view.

## Region selection (the "section extraction")

- Toolbar button toggles Region Mode.
- In 3D: `vtkBoxWidget2` (axis-aligned only — no rotation handles surfaced).
- In each MPR pane: an axis-aligned draggable rectangle that maps to the same `Region`. 2D editing is essential for precise cuts (e.g. "at the C2 vertebra").
- Size read-out in mm and voxel ranges shown next to the region controls.
- Exported mesh is the intersection of the segmentation mask and the region.

## Error handling

**Loading**
- Non-DICOM files in folder → skipped silently; count shown in the load log.
- Mixed series → series picker; never silently concatenated.
- Missing required tags (`PixelData`, `ImagePositionPatient`) → series excluded with reason.
- Compressed transfer syntax with missing decoder → clear "install X to read this file" message rather than a crash.

**Segmentation / export**
- Empty mask → no export, message hinting that threshold is likely wrong.
- Mesh with zero triangles after region intersection → same.
- Out-of-memory during marching cubes → caught; user told to crop tighter or increase decimation.
- Non-manifold output when "ensure manifold" is on → automatic fix via `vtkFillHolesFilter` and normals recomputation.

**General**
- All long-running operations run in `QThread`s; the UI stays responsive.
- Exceptions in workers are caught, logged, and surfaced as user-facing error dialogs with the underlying message.

## Project layout

```
Dicom/
├── pyproject.toml
├── uv.lock
├── README.md
├── src/dicom_viewer/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── study.py
│   │   ├── volume.py
│   │   ├── region.py
│   │   ├── segmentation/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── threshold.py
│   │   │   ├── region_grow.py
│   │   │   └── morphology.py
│   │   ├── mesh_export.py
│   │   └── document.py
│   ├── io/
│   │   ├── __init__.py
│   │   └── dicom_loader.py
│   ├── rendering/
│   │   ├── __init__.py
│   │   ├── slice_renderer.py
│   │   ├── volume_renderer.py
│   │   └── mesh_preview.py
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py
│       ├── panels/
│       │   ├── __init__.py
│       │   ├── windowing.py
│       │   ├── segmentation.py
│       │   └── export.py
│       └── widgets/
│           ├── __init__.py
│           └── slice_view.py
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   └── make_synthetic_series.py
    ├── core/
    ├── io/
    └── integration/
```

## Testing strategy

- **`core/` unit tests** — fast, headless, no GUI. Cover slice sorting by `ImagePositionPatient`, HU conversion, windowing math, threshold + largest-component on synthetic volumes (a cube embedded in noise), region intersection math, mesh export against a known synthetic cube volume (assert triangle count, bbox, manifold-ness).
- **`io/` tests** — generate synthetic DICOM series with `pydicom`'s dataset API into a `tmp_path`; cover happy path, mixed series, missing tags, compressed transfer syntaxes (skipped via `pytest.importorskip` if `pylibjpeg` is unavailable in CI).
- **`rendering/` tests** — instantiation/smoke tests using VTK's offscreen render windows. No pixel-comparison tests.
- **`ui/` tests** — `pytest-qt`. Threshold slider updates the document; "Export" triggers the worker; error states show the right dialog.
- **Integration test** — load a synthetic fixture series, threshold-segment, set a region, export STL; assert the file is a valid binary STL with more than zero triangles and bounds within the region.

Targets: `core/` ≥ 90% line coverage; UI smoke-coverage only.

## Open questions

None at the time of writing. Decisions made during brainstorming:
- Platform: Python desktop (PyQt + VTK + pydicom).
- Modality: both CT and MRI; CT-first segmentation tools.
- Scope: viewer + 3D + region selection + STL export.
- Architecture: document-model with VTK rendering.
- Region shape: axis-aligned only.
- Use case: personal own scans, non-clinical.
