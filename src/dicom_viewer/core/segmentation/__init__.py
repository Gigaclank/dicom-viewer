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
