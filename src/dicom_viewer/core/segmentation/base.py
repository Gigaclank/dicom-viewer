"""Segmentation result type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Segmentation:
    mask: np.ndarray  # bool, shape (z, y, x), same as source Volume
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_:
            raise ValueError(f"mask must be bool, got {self.mask.dtype}")

    @property
    def voxel_count(self) -> int:
        return int(self.mask.sum())

    @property
    def is_empty(self) -> bool:
        return self.voxel_count == 0
