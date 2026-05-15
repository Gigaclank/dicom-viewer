"""Volume3DView — a QWidget hosting the VolumeRenderer for the 4th pane."""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.region import Region
from dicom_viewer.core.volume import Volume
from dicom_viewer.rendering.volume_renderer import VolumeRenderer


class Volume3DView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._volume: Volume | None = None
        self._renderer = VolumeRenderer()

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
                self._vtk_widget.Initialize()
            except Exception:
                self._vtk_widget = QLabel("[3D render area]")  # type: ignore[assignment]
        else:
            self._vtk_widget = QLabel("[3D render area]")  # type: ignore[assignment]

        self._label = QLabel("3D")
        self.reset_button = QPushButton("Reset view")
        self.reset_button.setToolTip("Reset zoom, pan, and orientation of the 3D camera")
        self.reset_button.clicked.connect(self.reset_view)

        bottom = QHBoxLayout()
        bottom.addWidget(self._label, stretch=1)
        bottom.addWidget(self.reset_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self._vtk_widget, stretch=1)
        layout.addLayout(bottom)

    def set_volume(self, volume: Volume) -> None:
        self._volume = volume
        self._renderer.set_volume(volume)
        if volume.shape[0] < 2:
            self._label.setText(
                f"3D — {volume.modality}  {volume.shape[0]}×{volume.shape[1]}×{volume.shape[2]}  "
                f"(2D image — no 3D render)"
            )
        else:
            self._label.setText(
                f"3D — {volume.modality}  {volume.shape[0]}×{volume.shape[1]}×{volume.shape[2]}"
            )
        self._render_if_live()

    def set_overlay_mask(self, mask: np.ndarray | None) -> None:
        self._renderer.set_overlay_mask(mask)
        self._render_if_live()

    def set_windowing(self, center: float, width: float) -> None:
        """Re-iso the 3D view to the new windowing preset. Without this the
        opacity transfer function stays frozen at the bone default and
        Bone/Lung/Soft-tissue presets don't change what's rendered."""
        self._renderer.set_windowing(center, width)
        self._render_if_live()

    def set_region(self, region: Region | None) -> None:
        if region is None or self._volume is None:
            return
        self._renderer.set_region(region)
        self._render_if_live()

    def reset_view(self) -> None:
        self._renderer.reset_view()

    def set_crosshair_position(self, z: int, y: int, x: int) -> None:
        """Move the AX/COR/SAG indicator planes to the given voxel position.
        Called by MainWindow whenever the user scrubs one of the 2D panes."""
        self._renderer.set_crosshair_position(z, y, x)
        self._render_if_live()

    def _render_if_live(self) -> None:
        if hasattr(self._vtk_widget, "GetRenderWindow"):
            self._renderer.render()
