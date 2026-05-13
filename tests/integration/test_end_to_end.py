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
