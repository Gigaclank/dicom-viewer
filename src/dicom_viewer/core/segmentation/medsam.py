"""MedSAM integration — bounding-box-prompted segmentation via SAM.

Wraps the HuggingFace SamModel + SamProcessor with weights from
`wanglab/medsam-vit-base` so you get a click-and-segment experience.
The torch / transformers / pillow stack is optional — `is_available()`
returns False if the install is missing.

Workflow:
    seg = MedSAMSegmenter()
    if seg.is_available():
        result = seg.segment_volume_z(volume, region)
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.volume import Volume

_MODEL_ID = "wanglab/medsam-vit-base"


class MedSAMUnavailable(Exception):
    """Raised when torch / transformers aren't installed."""


class MedSAMSegmenter:
    """Lazy-loading wrapper around a MedSAM (SAM-based) inference pipeline."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device: Optional[str] = None

    @staticmethod
    def is_available() -> bool:
        """Returns True if torch + transformers are importable."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import SamModel, SamProcessor
        except ImportError as e:
            raise MedSAMUnavailable(
                "MedSAM requires `torch` and `transformers`. "
                "These ship with the standard install; if you're running from "
                "source and they're missing, run: "
                "pip install torch transformers pillow"
            ) from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # First call downloads ~360MB of weights into the HF cache.
        self._model = SamModel.from_pretrained(_MODEL_ID).to(self._device)
        self._model.eval()
        self._processor = SamProcessor.from_pretrained(_MODEL_ID)

    @property
    def device(self) -> str:
        if self._device is None:
            raise MedSAMUnavailable("Model not loaded yet — call segment_slice first.")
        return self._device

    def segment_slice(
        self,
        slice_2d: np.ndarray,
        box_xyxy: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Run MedSAM on one 2D slice with a bounding-box prompt.

        Returns a bool mask the same shape as `slice_2d`. Box is in pixel
        coordinates of the slice, ordered (x0, y0, x1, y1) — SAM convention.
        """
        self._ensure_loaded()
        import torch

        # Normalize the slice to uint8 RGB — SAM was trained on natural images.
        s = slice_2d.astype(np.float32)
        lo, hi = float(s.min()), float(s.max())
        if hi > lo:
            s = (s - lo) / (hi - lo) * 255.0
        s_uint8 = s.astype(np.uint8)
        rgb = np.stack([s_uint8] * 3, axis=-1)

        # HF SamProcessor wants input_boxes shaped [batch, num_boxes, 4].
        inputs = self._processor(
            images=rgb,
            input_boxes=[[list(box_xyxy)]],
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=False)

        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # masks[0] has shape (num_boxes, num_predictions, H, W).
        mask = masks[0][0, 0].numpy().astype(bool)
        return mask

    def segment_volume_z(
        self,
        volume: Volume,
        region: Region,
        *,
        progress: Optional[Callable[[str, float], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Segmentation:
        """Run MedSAM slice-by-slice over the z-range of `region` using the
        region's xy box as the prompt for every slice. Returns a 3D
        Segmentation matching the volume shape.

        This is the simplest 3D-from-2D propagation strategy. It works well
        when the target structure has a roughly consistent footprint across
        slices (most tumors satisfy this for small z-ranges). For larger
        ranges with significantly varying footprints, expect rough edges at
        the top / bottom.
        """
        bounds = volume.bbox()
        r = region.clamp_to(bounds)
        if r.is_empty:
            raise ValueError("region is empty after clamping to volume bounds")

        z0, z1 = r.z
        x0, x1 = r.x
        y0, y1 = r.y
        n_slices = z1 - z0
        mask = np.zeros(volume.shape, dtype=bool)

        for idx, z in enumerate(range(z0, z1)):
            if should_cancel is not None:
                try:
                    if should_cancel():
                        break
                except Exception:
                    pass
            if progress is not None:
                try:
                    progress(f"MedSAM slice {idx + 1}/{n_slices}", idx / max(1, n_slices))
                except Exception:
                    pass
            slice_2d = volume.array[z, :, :]
            slice_mask = self.segment_slice(slice_2d, (x0, y0, x1, y1))
            mask[z, :, :] = slice_mask

        if progress is not None:
            try:
                progress("MedSAM done", 1.0)
            except Exception:
                pass

        return Segmentation(
            mask=np.ascontiguousarray(mask),
            method="medsam",
            params={
                "model": _MODEL_ID,
                "region_z": tuple(r.z),
                "region_y": tuple(r.y),
                "region_x": tuple(r.x),
            },
        )
