"""MeshPreviewView — central-area tab that auto-refreshes the STL mesh.

Re-runs generate_mesh in a background thread whenever the segmentation,
region, or export options change. Debounced so dragging a slider doesn't
fire dozens of pipeline runs. Skips work entirely when the tab isn't
currently visible.

When only settings change, the user's current camera orientation is
preserved — only an explicit study swap (or the Reset view button)
re-frames the camera.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.document import Document
from dicom_viewer.core.mesh_export import (
    EmptyMeshError,
    ExportOptions,
    Mesh,
    generate_mesh,
)
from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.volume import Volume
from dicom_viewer.rendering.mesh_preview import MeshPreview


class _PreviewWorker(QThread):
    """Runs generate_mesh off the UI thread and reports a Mesh or an error."""

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        volume: Volume,
        segmentation: Segmentation,
        region: Region,
        options: ExportOptions,
    ) -> None:
        super().__init__()
        self._volume = volume
        self._segmentation = segmentation
        self._region = region
        self._options = options

    def run(self) -> None:
        try:
            mesh = generate_mesh(
                self._volume, self._segmentation, self._region, self._options
            )
            self.finished_ok.emit(mesh)
        except EmptyMeshError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


class MeshPreviewView(QWidget):
    """STL preview tab. Auto-refreshes on relevant changes."""

    DEBOUNCE_MS = 300

    def __init__(self, document: Document, export_panel) -> None:
        super().__init__()
        self._document = document
        self._export_panel = export_panel
        self._worker: _PreviewWorker | None = None
        self._refresh_pending = True
        self._tab_visible = False
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
                self._vtk_widget = QLabel("[STL render area]")
        else:
            self._vtk_widget = QLabel("[STL render area]")

        self._info_label = QLabel("No segmentation yet")
        self._info_label.setWordWrap(True)
        self.reset_button = QPushButton("Reset view")
        self.reset_button.setToolTip("Re-fit the camera to the current mesh")
        self.reset_button.clicked.connect(self._on_reset_clicked)

        bottom = QHBoxLayout()
        bottom.addWidget(self._info_label, stretch=1)
        bottom.addWidget(self.reset_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self._vtk_widget, stretch=1)
        layout.addLayout(bottom)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self.DEBOUNCE_MS)
        self._debounce.timeout.connect(self._maybe_run_worker)

        document.subscribe(self._on_doc_event)
        export_panel.settings_changed.connect(self.schedule_refresh)

    # --- visibility tracking ---
    def set_tab_visible(self, visible: bool) -> None:
        """Called by MainWindow when this widget becomes / stops being the
        current view tab. Defers expensive mesh work until the tab is shown."""
        self._tab_visible = visible
        if visible and self._refresh_pending:
            self._debounce.start()

    # --- refresh plumbing ---
    def schedule_refresh(self) -> None:
        self._refresh_pending = True
        if self._tab_visible:
            self._debounce.start()

    def _on_doc_event(self, kind: str) -> None:
        if kind == "study":
            # New study means the previous camera pose probably doesn't make
            # sense for the new geometry — refit on the next render.
            self._preview.request_fit()
            self.schedule_refresh()
        elif kind in ("segmentation", "region"):
            self.schedule_refresh()

    def _maybe_run_worker(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            # A worker is still running; leave _refresh_pending = True so we
            # re-fire when it finishes.
            return
        volume = self._document.volume
        seg = self._document.segmentation
        if volume is None or seg is None or seg.is_empty:
            self._preview.set_mesh(None)
            self._info_label.setText("No segmentation yet")
            self._refresh_pending = False
            self._render_if_live()
            return
        region = self._document.region or volume.bbox()
        options = self._export_panel.get_export_options()
        self._info_label.setText("Computing mesh…")
        self._refresh_pending = False
        self._worker = _PreviewWorker(volume, seg, region, options)
        self._worker.finished_ok.connect(self._on_mesh_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

    def _on_mesh_ready(self, mesh: Mesh) -> None:
        self._preview.set_mesh(mesh)
        (lo_z, lo_y, lo_x), (hi_z, hi_y, hi_x) = mesh.bounds_mm
        self._info_label.setText(
            f"{mesh.triangle_count:,} triangles • "
            f"{hi_x - lo_x:.1f} × {hi_y - lo_y:.1f} × {hi_z - lo_z:.1f} mm"
        )
        self._render_if_live()

    def _on_failed(self, msg: str) -> None:
        self._info_label.setText(f"Mesh error: {msg}")
        self._preview.set_mesh(None)
        self._render_if_live()

    def _on_worker_done(self) -> None:
        # If the user changed settings while the worker was running, schedule
        # another pass now that we're free.
        if self._refresh_pending and self._tab_visible:
            self._debounce.start()

    # --- camera controls ---
    def _on_reset_clicked(self) -> None:
        self._preview.reset_view()

    def _render_if_live(self) -> None:
        if hasattr(self._vtk_widget, "GetRenderWindow"):
            self._preview.render()
