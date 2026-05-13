"""MainWindow — tabbed views (axial / coronal / sagittal / 3D) with tools dock."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
)

from dicom_viewer.core.document import Document
from dicom_viewer.core.volume import Orientation
from dicom_viewer.io.dicom_loader import (
    LoaderError,
    load_series_from_file,
    load_series_from_folder,
)
from dicom_viewer.ui.panels.export import ExportPanel
from dicom_viewer.ui.panels.segmentation import SegmentationPanel
from dicom_viewer.ui.panels.windowing import WindowingPanel
from dicom_viewer.ui.widgets.mesh_preview_dialog import MeshPreviewDialog
from dicom_viewer.ui.widgets.slice_view import SliceView
from dicom_viewer.ui.widgets.volume3d_view import Volume3DView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DICOM Viewer")
        self.resize(1400, 900)

        self.document = Document()

        self.axial = SliceView(Orientation.AXIAL)
        self.coronal = SliceView(Orientation.CORONAL)
        self.sagittal = SliceView(Orientation.SAGITTAL)
        self.volume3d = Volume3DView()

        self.view_tabs = QTabWidget()
        self.view_tabs.addTab(self.axial, "Axial")
        self.view_tabs.addTab(self.coronal, "Coronal")
        self.view_tabs.addTab(self.sagittal, "Sagittal")
        self.view_tabs.addTab(self.volume3d, "3D")
        self.setCentralWidget(self.view_tabs)

        self.export_panel = ExportPanel(self.document)
        self._preview_dialog: MeshPreviewDialog | None = None
        self.export_panel.mesh_ready.connect(self._on_mesh_ready)

        tabs = QTabWidget()
        tabs.addTab(WindowingPanel(self.document), "Windowing")
        tabs.addTab(SegmentationPanel(self.document), "Segmentation")
        tabs.addTab(self.export_panel, "Export")
        dock = QDockWidget("Tools", self)
        dock.setWidget(tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        open_folder_action = QAction("Open DICOM Folder…", self)
        open_folder_action.setShortcut("Ctrl+O")
        open_folder_action.triggered.connect(self._on_open_folder)
        open_file_action = QAction("Open DICOM File…", self)
        open_file_action.setShortcut("Ctrl+Shift+O")
        open_file_action.triggered.connect(self._on_open_file)
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_folder_action)
        file_menu.addAction(open_file_action)

        reset_views_action = QAction("Reset All Views", self)
        reset_views_action.setShortcut("Ctrl+R")
        reset_views_action.triggered.connect(self._on_reset_views)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(reset_views_action)

        self.document.subscribe(self._on_doc_event)

    def _on_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open DICOM Folder")
        if not folder:
            return
        try:
            result = load_series_from_folder(Path(folder))
        except LoaderError as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        if len(result.studies) == 1:
            chosen = result.studies[0]
        else:
            items = [s.display_name for s in result.studies]
            picked, ok = QInputDialog.getItem(
                self, "Pick a series", "Multiple series found:", items, 0, False
            )
            if not ok:
                return
            chosen = result.studies[items.index(picked)]
        self.document.set_study(chosen)

    def _on_open_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open DICOM File",
            "",
            "DICOM files (*.dcm *.dicom *.DCM *.DICOM);;All files (*)",
        )
        if not path_str:
            return
        try:
            result = load_series_from_file(Path(path_str))
        except LoaderError as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self.document.set_study(result.studies[0])

    def _on_mesh_ready(self, mesh) -> None:
        if self._preview_dialog is None:
            self._preview_dialog = MeshPreviewDialog(parent=self)
        self._preview_dialog.set_mesh(mesh)
        self._preview_dialog.show()
        self._preview_dialog.raise_()
        self._preview_dialog.activateWindow()

    def _on_reset_views(self) -> None:
        self.axial.reset_view()
        self.coronal.reset_view()
        self.sagittal.reset_view()
        self.volume3d.reset_view()

    def _on_doc_event(self, kind: str) -> None:
        volume = self.document.volume
        if volume is None:
            return
        if kind == "study":
            self.axial.set_volume(volume)
            self.coronal.set_volume(volume)
            self.sagittal.set_volume(volume)
            self.volume3d.set_volume(volume)
        if kind in ("study", "windowing"):
            w = self.document.windowing
            self.axial.set_windowing(w.center, w.width)
            self.coronal.set_windowing(w.center, w.width)
            self.sagittal.set_windowing(w.center, w.width)
        if kind == "segmentation":
            mask = self.document.segmentation.mask if self.document.segmentation else None
            self.axial.set_overlay_mask(mask)
            self.coronal.set_overlay_mask(mask)
            self.sagittal.set_overlay_mask(mask)
            self.volume3d.set_overlay_mask(mask)
        if kind in ("study", "region"):
            self.volume3d.set_region(self.document.region)
