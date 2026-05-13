"""Connected-component and morphology operations on segmentation masks."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label

from dicom_viewer.core.segmentation.base import Segmentation


def keep_largest_component(seg: Segmentation) -> Segmentation:
    if seg.is_empty:
        return Segmentation(
            mask=seg.mask.copy(),
            method=f"{seg.method}+largest_component",
            params={"source_method": seg.method, "source_params": dict(seg.params)},
        )
    labeled, n = label(seg.mask)
    if n == 0:
        return Segmentation(
            mask=np.zeros_like(seg.mask),
            method=f"{seg.method}+largest_component",
            params={"source_method": seg.method, "source_params": dict(seg.params)},
        )
    # bincount ignores label 0 (background).
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    winner = int(counts.argmax())
    return Segmentation(
        mask=(labeled == winner),
        method=f"{seg.method}+largest_component",
        params={"source_method": seg.method, "source_params": dict(seg.params)},
    )


def smooth_mask(seg: Segmentation, iterations: int = 1) -> Segmentation:
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    # Closing fills pinholes; opening removes specks.
    closed = binary_closing(seg.mask, iterations=iterations)
    opened = binary_opening(closed, iterations=iterations)
    return Segmentation(
        mask=np.ascontiguousarray(opened),
        method=f"{seg.method}+smooth",
        params={
            "source_method": seg.method,
            "source_params": dict(seg.params),
            "iterations": iterations,
        },
    )
