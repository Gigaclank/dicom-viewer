"""MedSAM integration — bounding-box-prompted segmentation via SAM.

Wraps the HuggingFace SamModel + SamProcessor with weights from
`wanglab/medsam-vit-base`. `is_available()` returns False if torch /
transformers couldn't be imported when this module first loaded.

Threading notes
---------------
torch / transformers are imported at module load time, NOT lazily inside
the inference call, because importing torch from a non-main thread
(specifically the Qt worker thread used in the UI) can segfault — PyTorch
initialises thread-local state on import and that initialisation is not
safe outside the main thread on every platform.

We also cap torch's internal thread pools to 1 thread. SAM inference is
mostly cuBLAS / MKL calls under the hood; running it from a Qt worker
thread on top of an OpenMP-multithreaded backend has historically caused
intermittent core dumps. One thread is plenty for the inference rates we
need on CPU.

Workflow:
    seg = MedSAMSegmenter()
    if seg.is_available():
        result = seg.segment_volume_z(volume, region)
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.volume import Volume

_MODEL_ID = "wanglab/medsam-vit-base"
# CUDA opt-in: by default we run on CPU because the pip-installed torch
# wheels bundle bleeding-edge CUDA runtimes (CUDA 13+) that very few user
# machines have drivers for; the mismatch crashes at native level (core
# dump). Users who know their GPU + drivers match can opt in with this
# env var. Setting it does NOT force GPU — it just allows CUDA if the
# usual torch.cuda.is_available() check passes.
_CUDA_OPT_IN_ENV = "DICOM_VIEWER_MEDSAM_CUDA"

# Cap thread pools BEFORE torch is imported (these env vars are read once
# at C-extension init time).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Eager import so the heavy native libs load on whichever thread first
# imports this module — in the running app that's the main thread, because
# the segmentation panel does `from .medsam import MedSAMSegmenter` in its
# button click handler (main thread) before kicking off the worker.
try:
    import torch as _torch
    from transformers import SamModel as _SamModel  # type: ignore[import-not-found]
    from transformers import SamProcessor as _SamProcessor  # type: ignore[import-not-found]

    # Defensive: keep torch single-threaded so the QThread worker doesn't
    # fight with torch's internal thread pools.
    try:
        _torch.set_num_threads(1)
    except Exception:
        pass
    try:
        _torch.set_num_interop_threads(1)
    except Exception:
        pass

    _TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _e:  # noqa: BLE001
    _torch = None  # type: ignore[assignment]
    _SamModel = None  # type: ignore[assignment]
    _SamProcessor = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _e


class MedSAMUnavailable(Exception):
    """Raised when torch / transformers aren't installed or the model fails to load."""


class MedSAMSegmenter:
    """Lazy-loading wrapper around a MedSAM (SAM-based) inference pipeline."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device: Optional[str] = None

    @staticmethod
    def is_available() -> bool:
        """Returns True if torch + transformers loaded successfully at import."""
        return _TORCH_AVAILABLE

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not _TORCH_AVAILABLE:
            raise MedSAMUnavailable(
                "MedSAM requires `torch` and `transformers`. "
                "These ship with the standard install; if you're running from "
                f"source and they're missing, run: "
                f"pip install torch transformers pillow  "
                f"(import error was: {_TORCH_IMPORT_ERROR})"
            )

        # Device selection: CPU is the safe default. Opt into CUDA only when
        # the user explicitly sets the env var AND torch reports it available.
        cuda_ok = False
        if os.environ.get(_CUDA_OPT_IN_ENV) == "1":
            try:
                cuda_ok = bool(_torch.cuda.is_available())
            except Exception:
                cuda_ok = False
        self._device = "cuda" if cuda_ok else "cpu"

        # Wrap the model + processor load: failures are surfaced as a clean
        # MedSAMUnavailable instead of letting torch native code segfault.
        try:
            model = _SamModel.from_pretrained(_MODEL_ID)
            model = model.to(self._device)
            model.eval()
            self._model = model
            self._processor = _SamProcessor.from_pretrained(_MODEL_ID)
        except Exception as e:
            self._model = None
            self._processor = None
            raise MedSAMUnavailable(
                f"Failed to load MedSAM model (device={self._device!r}): {e}. "
                f"If you opted into CUDA via {_CUDA_OPT_IN_ENV}=1, try without it."
            ) from e

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

        with _torch.no_grad():
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
