"""MeshPreviewDialog — non-modal window showing the generated STL mesh."""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.mesh_export import Mesh
from dicom_viewer.rendering.mesh_preview import MeshPreview


class MeshPreviewDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("STL Preview")
        self.setWindowFlag(Qt.WindowType.Window)  # independent window with own title bar
        self.resize(640, 480)

        self._preview = MeshPreview()

        _headless = (
            os.environ.get("QT_QPA_PLATFORM") == "offscreen"
            or os.environ.get("DICOM_VIEWER_OFFSCREEN") == "1"
        )
        if not _headless:
            try:
                from vtkmodules.qt.QVTKRenderWindowInteractor import (  # type: ignore[import-untyped]
                    QVTKRenderWindowInteractor,
                )
                self._vtk_widget: QWidget = QVTKRenderWindowInteractor(self)
                self._preview.attach_render_window(self._vtk_widget.GetRenderWindow())
                self._vtk_widget.Initialize()
            except Exception:
                self._vtk_widget = QLabel("[mesh render area]")
        else:
            self._vtk_widget = QLabel("[mesh render area]")

        self._info_label = QLabel("No mesh yet")
        self.reset_button = QPushButton("Reset view")
        self.reset_button.clicked.connect(self._preview.reset_view)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)

        bottom = QHBoxLayout()
        bottom.addWidget(self._info_label, stretch=1)
        bottom.addWidget(self.reset_button)
        bottom.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._vtk_widget, stretch=1)
        layout.addLayout(bottom)

    def set_mesh(self, mesh: Mesh) -> None:
        self._preview.set_mesh(mesh)
        (lo_z, lo_y, lo_x), (hi_z, hi_y, hi_x) = mesh.bounds_mm
        self._info_label.setText(
            f"{mesh.triangle_count:,} triangles • "
            f"{hi_x - lo_x:.1f} × {hi_y - lo_y:.1f} × {hi_z - lo_z:.1f} mm"
        )
        if hasattr(self._vtk_widget, "GetRenderWindow"):
            self._preview.render()
