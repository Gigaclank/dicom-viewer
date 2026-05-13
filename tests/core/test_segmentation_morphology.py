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
