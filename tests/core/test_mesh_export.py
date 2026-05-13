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
