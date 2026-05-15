"""SliceView — a QWidget showing one MPR slice with a scrollbar."""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollBar,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.volume import Orientation, Volume
from dicom_viewer.rendering.slice_renderer import SliceRenderer


class SliceView(QWidget):
    slice_changed = pyqtSignal(int)
    # (z, y, x) voxel index of the click, and the brush mode active when it
    # happened. Only emitted when brush_mode != "off"; subscribers translate
    # this into a segmentation grow + merge.
    seed_clicked = pyqtSignal(tuple, str)
    # Emitted on each MouseMove while the left button is held in brush mode.
    # Used by the 2D paint brush to stream stroke positions; click-based
    # brushes ignore this and only act on seed_clicked.
    seed_dragged = pyqtSignal(tuple, str)
    # Emitted on left-button release after a brush press/drag. Lets paint
    # consumers know the stroke ended (useful for committing an undo step).
    seed_drag_ended = pyqtSignal(str)

    BRUSH_MODES = ("off", "add", "remove")

    def __init__(self, orientation: Orientation) -> None:
        super().__init__()
        self.orientation = orientation
        self._volume: Volume | None = None
        self._renderer = SliceRenderer(orientation=orientation)
        self._brush_mode: str = "off"
        # True while the user is dragging in brush mode (left button held).
        # Drives whether MouseMove events emit seed_dragged.
        self._brush_dragging: bool = False

        _headless = (
            os.environ.get("QT_QPA_PLATFORM") == "offscreen"
            or os.environ.get("DICOM_VIEWER_OFFSCREEN") == "1"
        )
        if not _headless:
            try:
                from vtkmodules.qt.QVTKRenderWindowInteractor import (  # type: ignore[import-untyped]
                    QVTKRenderWindowInteractor,
                )
                self._vtk_widget = QVTKRenderWindowInteractor(self)
                self._renderer.attach_render_window(self._vtk_widget.GetRenderWindow())
                self._vtk_widget.installEventFilter(self)
            except Exception:
                # VTK Qt bridge unavailable: fall back to placeholder.
                self._vtk_widget = QLabel("[vtk render area]")  # type: ignore[assignment]
        else:
            # Headless test environment: skip VTK widget instantiation.
            self._vtk_widget = QLabel("[vtk render area]")  # type: ignore[assignment]

        self.scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(0)
        self.scrollbar.valueChanged.connect(self._on_scroll)

        self._label = QLabel(f"{orientation.value} — 0 / 0")
        self.reset_button = QPushButton("Reset view")
        self.reset_button.setToolTip("Reset zoom, pan, and orientation for this pane")
        self.reset_button.clicked.connect(self.reset_view)

        row = QHBoxLayout()
        row.addWidget(self._vtk_widget, stretch=1)
        row.addWidget(self.scrollbar)
        bottom = QHBoxLayout()
        bottom.addWidget(self._label, stretch=1)
        bottom.addWidget(self.reset_button)
        layout = QVBoxLayout(self)
        layout.addLayout(row, stretch=1)
        layout.addLayout(bottom)

    def set_volume(self, volume: Volume) -> None:
        self._volume = volume
        self._renderer.set_volume(volume)
        max_index = self._max_index()
        self.scrollbar.setMaximum(max_index)
        self.scrollbar.setValue(max_index // 2)
        self._update_label()
        if hasattr(self._vtk_widget, "GetRenderWindow"):
            self._renderer.render()

    def set_windowing(self, center: float, width: float) -> None:
        self._renderer.set_windowing(center, width)
        if hasattr(self._vtk_widget, "GetRenderWindow"):
            self._renderer.render()

    def set_overlay_mask(self, mask: np.ndarray | None) -> None:
        self._renderer.set_overlay_mask(mask)
        if hasattr(self._vtk_widget, "GetRenderWindow"):
            self._renderer.render()

    def reset_view(self) -> None:
        """Reset zoom, pan, and orientation for this pane to default."""
        self._renderer.reset_view()

    def set_brush_mode(self, mode: str) -> None:
        """Switch click handling between off / add-seed / remove-seed.

        Off restores VTK's normal interaction (window-level on left-drag,
        pan on middle-drag, zoom on Ctrl+wheel). Add/Remove turn left-click
        into a seed-drop that emits ``seed_clicked``; the cursor changes to
        a crosshair as a visual cue.
        """
        if mode not in self.BRUSH_MODES:
            raise ValueError(f"unknown brush mode: {mode!r}")
        self._brush_mode = mode
        if not hasattr(self._vtk_widget, "GetRenderWindow"):
            return
        if mode == "off":
            self._vtk_widget.unsetCursor()
        else:
            self._vtk_widget.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    @property
    def brush_mode(self) -> str:
        return self._brush_mode

    @property
    def current_index(self) -> int:
        return int(self.scrollbar.value())

    def _on_scroll(self, value: int) -> None:
        self._renderer.set_slice_index(value)
        self._update_label()
        if hasattr(self._vtk_widget, "GetRenderWindow"):
            self._renderer.render()
        self.slice_changed.emit(value)

    def _max_index(self) -> int:
        if self._volume is None:
            return 0
        sz, sy, sx = self._volume.shape
        if self.orientation is Orientation.AXIAL:
            return sz - 1
        if self.orientation is Orientation.CORONAL:
            return sy - 1
        return sx - 1

    def _update_label(self) -> None:
        self._label.setText(
            f"{self.orientation.value} — {self.current_index} / {self._max_index()}"
        )

    def _seed_in_bounds(self, seed: tuple[int, int, int]) -> bool:
        if self._volume is None:
            return False
        sz, sy, sx = self._volume.shape
        z, y, x = seed
        return 0 <= z < sz and 0 <= y < sy and 0 <= x < sx

    # Slices advanced per wheel notch. Chosen empirically: one slice per notch
    # was too slow on typical CT volumes (300+ slices). Shift multiplies for
    # fast traversal of long scans.
    _SCRUB_STEP = 1
    _SCRUB_SHIFT_MULTIPLIER = 5  # 15 slices per notch with Shift held

    def _voxel_at_event_position(self, pos) -> tuple[int, int, int] | None:
        """Convert a Qt mouse position on the VTK widget into a (z, y, x)
        voxel index, accounting for the renderer's display→world map and
        the orientation-specific row/col layout. Returns None if we don't
        have everything we need (no volume, headless mode, etc.)."""
        if self._volume is None or not hasattr(self._vtk_widget, "GetRenderWindow"):
            return None
        vtk_y = max(0, self._vtk_widget.height() - int(pos.y()))
        world = self._renderer._display_to_world(int(pos.x()), vtk_y)
        return self._volume.voxel_at_click(
            self.orientation,
            self.current_index,
            (world[0], world[1]),
        )

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if obj is self._vtk_widget:
            et = event.type()
            # --- brush press: drop a seed, start drag tracking ---
            if et == QEvent.Type.MouseButtonPress:
                me: QMouseEvent = event  # type: ignore[assignment]
                if (
                    me.button() == Qt.MouseButton.LeftButton
                    and self._brush_mode != "off"
                ):
                    seed = self._voxel_at_event_position(me.position())
                    if seed is None:
                        return False
                    if self._seed_in_bounds(seed):
                        self.seed_clicked.emit(seed, self._brush_mode)
                    self._brush_dragging = True
                    return True  # consume so VTK doesn't window-level
            # --- brush drag: stream stroke positions for paint brushes ---
            elif et == QEvent.Type.MouseMove:
                if self._brush_dragging and self._brush_mode != "off":
                    me = event  # type: ignore[assignment]
                    seed = self._voxel_at_event_position(me.position())
                    if seed is not None and self._seed_in_bounds(seed):
                        self.seed_dragged.emit(seed, self._brush_mode)
                    return True
            # --- brush release: stop drag tracking ---
            elif et == QEvent.Type.MouseButtonRelease:
                me = event  # type: ignore[assignment]
                if me.button() == Qt.MouseButton.LeftButton and self._brush_dragging:
                    self._brush_dragging = False
                    self.seed_drag_ended.emit(self._brush_mode)
                    return True
        if obj is self._vtk_widget and event.type() == QEvent.Type.Wheel:
            wheel: QWheelEvent = event  # type: ignore[assignment]
            delta_y = wheel.angleDelta().y()
            if delta_y == 0:
                return False
            if wheel.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Cursor-centered zoom: factor < 1 zooms in (smaller parallel
                # scale), the point under the cursor stays put.
                factor = 1.0 / 1.15 if delta_y > 0 else 1.15
                pos = wheel.position()
                self._renderer.zoom_at(
                    factor,
                    int(pos.x()),
                    int(pos.y()),
                    self._vtk_widget.height(),
                )
                if hasattr(self._vtk_widget, "GetRenderWindow"):
                    self._renderer.render()
                return True
            # Wheel forward (positive delta) = previous slice; backward = next.
            # Convention is arbitrary but feels natural: rolling toward the screen
            # advances "into" the volume.
            step = -self._SCRUB_STEP if delta_y > 0 else self._SCRUB_STEP
            if wheel.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                step *= self._SCRUB_SHIFT_MULTIPLIER
            new_value = max(
                self.scrollbar.minimum(),
                min(self.scrollbar.maximum(), self.scrollbar.value() + step),
            )
            self.scrollbar.setValue(new_value)
            return True
        return super().eventFilter(obj, event)
