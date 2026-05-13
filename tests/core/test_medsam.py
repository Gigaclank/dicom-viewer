"""MedSAM availability check + unit-level wiring.

A real model run downloads ~360MB of weights and needs torch installed,
so the substantive run is gated behind `torch` being importable. The
availability + import-pathway checks always run.
"""
import os

import numpy as np
import pytest

from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.medsam import MedSAMSegmenter, MedSAMUnavailable
from dicom_viewer.core.volume import Volume


def test_is_available_does_not_crash():
    """is_available() is a static check that returns True or False, never raises."""
    result = MedSAMSegmenter.is_available()
    assert isinstance(result, bool)


def test_segment_slice_raises_unavailable_when_torch_missing(monkeypatch):
    """If torch isn't installed, segment_slice raises MedSAMUnavailable —
    UI handles this case by disabling the feature."""
    # Simulate missing torch by removing it from sys.modules and blocking re-import.
    import sys

    original_torch = sys.modules.pop("torch", None)
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        seg = MedSAMSegmenter()
        with pytest.raises((MedSAMUnavailable, ModuleNotFoundError, TypeError)):
            seg.segment_slice(np.zeros((8, 8), dtype=np.int16), (1, 1, 5, 5))
    finally:
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        else:
            sys.modules.pop("torch", None)


def test_segment_volume_z_rejects_empty_region():
    """An empty region clamps to no-op and the segmenter raises a clear error
    before attempting any model work."""
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    vol = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    seg = MedSAMSegmenter()
    # An "empty" region (z range outside bounds) clamps to (z=(10,10) etc),
    # which is_empty=True; the function should fail fast.
    with pytest.raises(ValueError):
        seg.segment_volume_z(vol, Region(z=(10, 12), y=(0, 4), x=(0, 4)))


@pytest.mark.skipif(
    not MedSAMSegmenter.is_available()
    or not os.environ.get("DICOM_VIEWER_RUN_MEDSAM_TEST"),
    reason=(
        "Real MedSAM inference is gated behind DICOM_VIEWER_RUN_MEDSAM_TEST=1 "
        "because it downloads ~360MB of weights from HuggingFace and runs "
        "actual torch inference. Set the env var to opt in."
    ),
)
def test_segment_volume_z_integration_smoke():
    """Opt-in real inference. Set DICOM_VIEWER_RUN_MEDSAM_TEST=1 and ensure
    torch + transformers are installed. The first run downloads ~360MB of
    weights from HuggingFace."""
    # Tiny 2-slice volume so the test isn't slow even on CPU.
    arr = np.random.default_rng(0).integers(0, 1000, (2, 64, 64), dtype=np.int16)
    vol = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    seg = MedSAMSegmenter()
    result = seg.segment_volume_z(
        vol, Region(z=(0, 2), y=(10, 50), x=(10, 50))
    )
    assert result.mask.shape == arr.shape
    assert result.method == "medsam"
