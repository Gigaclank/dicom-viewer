"""Document — the single source of truth for the loaded study and edits.

Observers register a callback via `subscribe(fn)`; `subscribe` returns a
zero-argument unsubscribe handle. Callbacks receive a string event-kind:
"study" | "volume" | "segmentation" | "region" | "windowing" | "mask_library".

Mask library: each Document carries a named-mask catalogue distinct from
the currently-active `segmentation`. The active segmentation IS one of the
library entries (or unsaved scratch). UI consumers use save_mask_as /
activate_mask / delete_mask + `mask_names` to drive the picker UI; project
save/load round-trips the catalogue via companion NIfTI files.
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
        # Named-mask catalogue. The active segmentation may or may not be one
        # of these — masks only enter the catalogue via save_mask_as().
        self._masks: dict[str, Segmentation] = {}
        self._active_mask_name: str = ""

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
        # A new study invalidates any cached masks — they were shaped for
        # the previous volume.
        self._masks.clear()
        self._active_mask_name = ""
        self._emit("study")
        self._emit("volume")
        self._emit("region")
        self._emit("windowing")
        # The segmentation was reset to None above; emit so observers (slice
        # views' overlay actors, the 3D pane) drop their stale mask references
        # before they're ever asked to render against the new volume.
        self._emit("segmentation")
        self._emit("mask_library")

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

    # --- mask library ---
    @property
    def mask_names(self) -> list[str]:
        return list(self._masks.keys())

    @property
    def active_mask_name(self) -> str:
        return self._active_mask_name

    def get_mask(self, name: str) -> Segmentation | None:
        return self._masks.get(name)

    def save_mask_as(self, name: str) -> None:
        """Snapshot the current segmentation under `name`. The mask becomes
        the active library entry. No-op if there's no current segmentation."""
        if self._segmentation is None or not name:
            return
        self._masks[name] = self._segmentation
        self._active_mask_name = name
        self._emit("mask_library")

    def activate_mask(self, name: str) -> None:
        """Switch the active segmentation to a library entry. `name=""`
        clears the active mask without removing it from the library."""
        if name == "":
            self._active_mask_name = ""
            self.set_segmentation(None)
            return
        seg = self._masks.get(name)
        if seg is None:
            return
        self._active_mask_name = name
        # set_segmentation emits 'segmentation'; the library event is a
        # change in WHICH entry is active, which observers may also care
        # about (e.g. the picker keeping selection in sync).
        self.set_segmentation(seg)
        self._emit("mask_library")

    def delete_mask(self, name: str) -> None:
        if name not in self._masks:
            return
        del self._masks[name]
        if self._active_mask_name == name:
            self._active_mask_name = ""
            self.set_segmentation(None)
        self._emit("mask_library")

    def replace_masks(
        self,
        masks: dict[str, Segmentation],
        active_name: str = "",
    ) -> None:
        """Bulk-replace the library — used by project load. Pass `active_name`
        to make one of them the current segmentation."""
        self._masks = dict(masks)
        if active_name in self._masks:
            self._active_mask_name = active_name
            self.set_segmentation(self._masks[active_name])
        else:
            self._active_mask_name = ""
            self.set_segmentation(None)
        self._emit("mask_library")

    @staticmethod
    def _default_windowing_for(volume: Volume) -> WindowingState:
        if volume.modality == "CT":
            return _CT_DEFAULT
        try:
            lo, hi = volume.intensity_percentiles(1, 99)
            return WindowingState(center=(lo + hi) / 2.0, width=max(hi - lo, 1.0))
        except Exception:
            return _MR_DEFAULT_FALLBACK
