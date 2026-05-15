"""Tests for the click-seed brush primitives (grow_from_seed +
apply_brush_stroke). These are pure functions over numpy arrays, so the
tests don't need Qt or VTK."""
import numpy as np
import pytest

from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.click_seed import apply_brush_stroke, grow_from_seed
from dicom_viewer.core.volume import Volume


def _vol_with_two_blobs() -> Volume:
    """8^3 CT-like volume with two disconnected high-intensity blobs."""
    arr = np.zeros((8, 8, 8), dtype=np.int16)
    arr[1:3, 1:3, 1:3] = 500   # blob A (corner)
    arr[5:7, 5:7, 5:7] = 500   # blob B (opposite corner)
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_grow_from_seed_fills_only_the_seeded_blob():
    """Region grow from a seed inside blob A must include all of A and
    none of B — they're disconnected so the flood-fill can't cross."""
    vol = _vol_with_two_blobs()
    mask = grow_from_seed(vol, seed=(2, 2, 2), tolerance=100)
    # All of blob A is in the mask.
    assert mask[1:3, 1:3, 1:3].all()
    # None of blob B is.
    assert not mask[5:7, 5:7, 5:7].any()
    # Background is not in the mask either (seed value is 500, bg is 0).
    assert not mask[0, 0, 0]


def test_grow_from_seed_respects_region_bbox():
    """A seed inside the region grows only within the region; voxels beyond
    the bbox stay zero even if they'd be connected at the seed value."""
    arr = np.zeros((8, 8, 8), dtype=np.int16)
    arr[1:7, 1:7, 1:7] = 500   # one big connected blob
    vol = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    bbox = Region(z=(2, 5), y=(2, 5), x=(2, 5))
    mask = grow_from_seed(vol, seed=(3, 3, 3), tolerance=100, region=bbox)
    # Only the cropped interior is filled.
    assert mask[2:5, 2:5, 2:5].all()
    # Voxels outside the bbox stay zero even though they're at intensity 500.
    assert not mask[1, 1, 1]
    assert not mask[6, 6, 6]


def test_grow_from_seed_outside_region_returns_empty():
    """User clicked outside the active region — return an empty mask
    rather than silently disabling the region constraint."""
    vol = _vol_with_two_blobs()
    region = Region(z=(0, 4), y=(0, 4), x=(0, 4))
    # Blob B is fully outside that region.
    mask = grow_from_seed(vol, seed=(6, 6, 6), tolerance=100, region=region)
    assert not mask.any()


def test_grow_from_seed_rejects_seed_outside_volume():
    vol = _vol_with_two_blobs()
    with pytest.raises(ValueError):
        grow_from_seed(vol, seed=(100, 100, 100), tolerance=50)


def test_apply_brush_stroke_add_unions_masks():
    base = np.zeros((4, 4, 4), dtype=bool)
    base[0, 0, 0] = True
    addition = np.zeros_like(base)
    addition[1, 1, 1] = True
    out = apply_brush_stroke(base, addition, "add", base.shape)
    assert out[0, 0, 0] and out[1, 1, 1]
    # base unchanged (function returns a new array, doesn't mutate).
    assert base.sum() == 1


def test_apply_brush_stroke_remove_subtracts():
    base = np.ones((4, 4, 4), dtype=bool)
    addition = np.zeros_like(base)
    addition[2, 2, 2] = True
    out = apply_brush_stroke(base, addition, "remove", base.shape)
    assert not out[2, 2, 2]
    # Everything else stayed True.
    assert out.sum() == base.size - 1


def test_apply_brush_stroke_seeds_from_none_in_add_mode():
    """First click on an empty document should produce a mask = addition."""
    addition = np.zeros((4, 4, 4), dtype=bool)
    addition[1, 2, 3] = True
    out = apply_brush_stroke(None, addition, "add", (4, 4, 4))
    assert out[1, 2, 3]
    assert out.sum() == 1


def test_apply_brush_stroke_remove_with_no_base_is_empty():
    """Remove-mode with no existing mask is a no-op — return empty rather
    than crash. Otherwise the very first click in Remove mode would error."""
    addition = np.zeros((4, 4, 4), dtype=bool)
    addition[0, 0, 0] = True
    out = apply_brush_stroke(None, addition, "remove", (4, 4, 4))
    assert not out.any()


def test_apply_brush_stroke_unknown_mode_raises():
    arr = np.zeros((2, 2, 2), dtype=bool)
    with pytest.raises(ValueError):
        apply_brush_stroke(arr, arr, "draw", (2, 2, 2))


# --- new brush kinds --------------------------------------------------------


def test_threshold_from_seed_ignores_connectivity():
    """Threshold-style brush includes every voxel in ±tolerance regardless of
    whether it's connected to the seed — the differentiator vs region grow."""
    from dicom_viewer.core.segmentation.click_seed import threshold_from_seed

    vol = _vol_with_two_blobs()
    # Seed is in blob A; without connectivity, blob B (also 500 HU) is
    # also marked. Region grow would not catch blob B; threshold does.
    mask = threshold_from_seed(vol, seed=(2, 2, 2), tolerance=100)
    assert mask[1:3, 1:3, 1:3].all()
    assert mask[5:7, 5:7, 5:7].all()


def test_threshold_from_seed_respects_region():
    from dicom_viewer.core.segmentation.click_seed import threshold_from_seed

    vol = _vol_with_two_blobs()
    region = Region(z=(0, 4), y=(0, 4), x=(0, 4))
    mask = threshold_from_seed(vol, seed=(2, 2, 2), tolerance=100, region=region)
    assert mask[1:3, 1:3, 1:3].all()
    # Blob B is outside the region — must not be in the mask.
    assert not mask[5:7, 5:7, 5:7].any()


def test_sphere_from_seed_uses_world_space_radius():
    """Sphere radius is in mm, so anisotropic spacing must shrink the sphere
    along the axis with bigger voxels. Verify by giving Z 2× the spacing of
    X/Y and checking that the sphere extends fewer voxels along Z."""
    from dicom_viewer.core.segmentation.click_seed import sphere_from_seed

    arr = np.zeros((11, 11, 11), dtype=np.int16)
    vol = Volume(array=arr, spacing_mm=(2.0, 1.0, 1.0), modality="CT")
    mask = sphere_from_seed(vol, seed=(5, 5, 5), radius_mm=3.0)
    # Center voxel is inside.
    assert mask[5, 5, 5]
    # X/Y direction with spacing 1mm: 3mm radius → 3 voxels each side.
    assert mask[5, 5, 5 + 3]
    assert mask[5, 5, 5 - 3]
    # Z direction with spacing 2mm: 3mm radius → 1 voxel each side, not 3.
    assert mask[5 + 1, 5, 5]
    assert not mask[5 + 2, 5, 5]


def test_sphere_from_seed_zero_radius_is_empty():
    from dicom_viewer.core.segmentation.click_seed import sphere_from_seed

    vol = _vol_with_two_blobs()
    mask = sphere_from_seed(vol, seed=(4, 4, 4), radius_mm=0.0)
    assert not mask.any()


def test_box_from_seed_is_axis_aligned_and_inclusive():
    """Box brush drops an axis-aligned cuboid centered on the seed. Verify
    every voxel inside the half-extent is true and the boundary is closed."""
    from dicom_viewer.core.segmentation.click_seed import box_from_seed

    arr = np.zeros((11, 11, 11), dtype=np.int16)
    vol = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    mask = box_from_seed(vol, seed=(5, 5, 5), half_extent_mm=2.0)
    # Inclusive box: voxels 3..7 along each axis.
    assert mask[3:8, 3:8, 3:8].all()
    # Outside the box stays false.
    assert not mask[2, 5, 5]
    assert not mask[8, 5, 5]


def test_paint_disc_2d_paints_on_axial_slice_only():
    """A paint stroke must only mutate the targeted slice. Other slices stay
    untouched even if the disc would cover their (y, x) coordinates."""
    from dicom_viewer.core.segmentation.click_seed import paint_disc_2d
    from dicom_viewer.core.volume import Orientation

    mask = np.zeros((6, 12, 12), dtype=bool)
    paint_disc_2d(
        mask, Orientation.AXIAL, slice_index=2, center_yx=(6, 6), radius_px=3
    )
    # Slice 2 has the disc.
    assert mask[2, 6, 6]
    assert mask[2, 6, 9]  # boundary
    assert not mask[2, 6, 10]  # just outside
    # Other slices completely untouched.
    assert not mask[1, 6, 6]
    assert not mask[3, 6, 6]


def test_paint_disc_2d_erase_mode_subtracts():
    """set_value=False erases pixels under the disc instead of adding them.
    This is how Remove mode applies paint without going through apply_brush_stroke."""
    from dicom_viewer.core.segmentation.click_seed import paint_disc_2d
    from dicom_viewer.core.volume import Orientation

    mask = np.ones((4, 8, 8), dtype=bool)
    paint_disc_2d(
        mask, Orientation.AXIAL, slice_index=1, center_yx=(4, 4), radius_px=2,
        set_value=False,
    )
    assert not mask[1, 4, 4]
    # Outside the disc stays painted.
    assert mask[1, 0, 0]
    # Other slices untouched.
    assert mask[0, 4, 4]


def test_paint_disc_2d_on_coronal_undoes_z_flip():
    """Coronal slices flip z so superior renders on top; the disc must paint
    on the SAME flipped row as the user sees, i.e. center_yx row=0 maps to
    volume z=(Z-1)."""
    from dicom_viewer.core.segmentation.click_seed import paint_disc_2d
    from dicom_viewer.core.volume import Orientation

    mask = np.zeros((6, 8, 8), dtype=bool)
    paint_disc_2d(
        mask, Orientation.CORONAL, slice_index=4, center_yx=(0, 4), radius_px=1
    )
    # row=0 → volume z = (6-1)-0 = 5.
    assert mask[5, 4, 4]
    # Other z's stay clean.
    assert not mask[0, 4, 4]


def test_confidence_grow_includes_seed_blob():
    """ConfidenceConnected from a seed inside a uniform blob fills that blob.
    Hard to assert exact extent without pinning SITK behavior; we just
    confirm the seed voxel and immediate neighbors come through."""
    from dicom_viewer.core.segmentation.click_seed import confidence_grow_from_seed

    arr = np.zeros((8, 8, 8), dtype=np.int16)
    arr[2:6, 2:6, 2:6] = 800
    # Add noise so SITK can compute a stddev > 0.
    rng = np.random.default_rng(0)
    arr[2:6, 2:6, 2:6] += rng.integers(-10, 11, size=(4, 4, 4), dtype=np.int16)
    vol = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    mask = confidence_grow_from_seed(vol, seed=(3, 3, 3), multiplier=2.5)
    assert mask[3, 3, 3]
    assert mask.sum() > 0
