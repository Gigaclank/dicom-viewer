"""Segmentation methods. Each public function returns a Segmentation."""
from dicom_viewer.core.segmentation.base import Segmentation  # re-export
from dicom_viewer.core.segmentation.threshold import threshold

__all__ = ["Segmentation", "threshold"]
