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


def test_preview_mode_downsamples_when_over_budget(monkeypatch):
    """With preview_mode=True, a mask above the voxel budget gets max-pool
    downsampled before marching cubes — far fewer triangles than full mode.

    Uses a small in-memory volume but a tiny voxel budget so the downsample
    path always runs (the test stays fast).
    """
    import dicom_viewer.core.mesh_export as me
    monkeypatch.setattr(me, "PREVIEW_VOXEL_BUDGET", 1000)  # force downsample

    side = 32
    cube_size = 24
    arr = np.zeros((side, side, side), dtype=np.int16)
    s = (side - cube_size) // 2
    arr[s : s + cube_size, s : s + cube_size, s : s + cube_size] = 1000
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    seg = threshold(v, low=500, high=2000)
    opts = ExportOptions(smoothing_iterations=0, decimation_target_reduction=0.0, ensure_manifold=False)

    full = generate_mesh(v, seg, region=v.bbox(), options=opts)
    preview = generate_mesh(v, seg, region=v.bbox(), options=opts, preview_mode=True)
    assert preview.triangle_count > 0
    # Preview should be materially smaller.
    assert preview.triangle_count < full.triangle_count

    # World bounds should still describe roughly the same cube in mm space —
    # downsampling shouldn't move the mesh, only coarsen it.
    (lo_f_z, lo_f_y, lo_f_x), (hi_f_z, hi_f_y, hi_f_x) = full.bounds_mm
    (lo_p_z, lo_p_y, lo_p_x), (hi_p_z, hi_p_y, hi_p_x) = preview.bounds_mm
    for full_lo, prev_lo in zip((lo_f_z, lo_f_y, lo_f_x), (lo_p_z, lo_p_y, lo_p_x)):
        assert abs(full_lo - prev_lo) < 5.0


def test_progress_callback_is_invoked():
    v = _cube_volume()
    seg = threshold(v, low=500, high=2000)
    calls: list[tuple[str, float]] = []
    generate_mesh(
        v, seg, region=v.bbox(),
        options=ExportOptions(smoothing_iterations=5),
        progress=lambda stage, frac: calls.append((stage, frac)),
    )
    stages = [c[0] for c in calls]
    fractions = [c[1] for c in calls]
    assert "Marching cubes" in stages
    assert "Done" in stages
    assert fractions[-1] == 1.0
    # Fractions must be non-decreasing.
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))


def test_progress_callback_failure_does_not_break_pipeline():
    """The pipeline must keep going even if the progress callback raises."""
    v = _cube_volume()
    seg = threshold(v, low=500, high=2000)

    def boom(_stage: str, _frac: float) -> None:
        raise RuntimeError("intentional")

    mesh = generate_mesh(v, seg, region=v.bbox(), options=ExportOptions(), progress=boom)
    assert mesh.triangle_count > 0
