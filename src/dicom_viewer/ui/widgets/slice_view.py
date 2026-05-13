"""SliceView — a QWidget showing one MPR slice with a scrollbar."""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
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

    def __init__(self, orientation: Orientation) -> None:
        super().__init__()
        self.orientation = orientation
        self._volume: Volume | None = None
        self._renderer = SliceRenderer(orientation=orientation)

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
