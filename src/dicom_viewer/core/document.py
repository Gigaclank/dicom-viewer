"""Document — the single source of truth for the loaded study and edits.

Observers register a callback via `subscribe(fn)`; `subscribe` returns a
zero-argument unsubscribe handle. Callbacks receive a string event-kind:
"study" | "volume" | "segmentation" | "region" | "windowing".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Volume

Observer = Callable[[str], None]


@dataclass(frozen=True)
class WindowingState:
    center: float
    width: float


_CT_DEFAULT = WindowingState(center=40, width=400)  # soft tissue
_MR_DEFAULT_FALLBACK = WindowingState(center=300, width=600)


class Document:
    def __init__(self) -> None:
        self._study: Study | None = None
        self._segmentation: Segmentation | None = None
        self._region: Region | None = None
        self._windowing: WindowingState = _CT_DEFAULT
        self._observers: list[Observer] = []

    # --- observer plumbing ---
    def subscribe(self, fn: Observer) -> Callable[[], None]:
        self._observers.append(fn)

        def unsubscribe() -> None:
            try:
                self._observers.remove(fn)
            except ValueError:
                pass

        return unsubscribe

    def _emit(self, kind: str) -> None:
        for fn in list(self._observers):
            fn(kind)

    # --- accessors ---
    @property
    def study(self) -> Study | None:
        return self._study

    @property
    def volume(self) -> Volume | None:
        return self._study.volume if self._study else None

    @property
    def segmentation(self) -> Segmentation | None:
        return self._segmentation

    @property
    def region(self) -> Region | None:
        return self._region

    @property
    def windowing(self) -> WindowingState:
        return self._windowing

    # --- mutators ---
    def set_study(self, study: Study) -> None:
        self._study = study
        self._segmentation = None
        self._region = study.volume.bbox()
        self._windowing = self._default_windowing_for(study.volume)
        self._emit("study")
        self._emit("volume")
        self._emit("region")
        self._emit("windowing")

    def set_segmentation(self, seg: Segmentation | None) -> None:
        self._segmentation = seg
        self._emit("segmentation")

    def set_region(self, region: Region) -> None:
        if self.volume is None:
            self._region = region
        else:
            self._region = region.clamp_to(self.volume.bbox())
        self._emit("region")

    def set_windowing(self, w: WindowingState) -> None:
        self._windowing = w
        self._emit("windowing")

    @staticmethod
    def _default_windowing_for(volume: Volume) -> WindowingState:
        if volume.modality == "CT":
            return _CT_DEFAULT
        try:
            lo, hi = volume.intensity_percentiles(1, 99)
            return WindowingState(center=(lo + hi) / 2.0, width=max(hi - lo, 1.0))
        except Exception:
            return _MR_DEFAULT_FALLBACK
