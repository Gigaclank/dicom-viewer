"""MainWindow — tabbed views (axial / coronal / sagittal / 3D) with tools dock."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QToolBar,
)

from dicom_viewer.core.document import Document, WindowingState
from dicom_viewer.core.region import Region
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Orientation
from dicom_viewer.io.config import save_last_project
from dicom_viewer.io.dicom_loader import (
    LoaderError,
    load_series_from_file,
    load_series_from_folder,
)
from dicom_viewer.io.project import (
    PROJECT_EXTENSION,
    Project,
    ProjectError,
    RegionSettings,
    SourceSettings,
    WindowingSettings,
    load_project,
    save_project,
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

        self.windowing_panel = WindowingPanel(self.document)
        self.segmentation_panel = SegmentationPanel(self.document)
        self.export_panel = ExportPanel(self.document)
        self._preview_dialog: MeshPreviewDialog | None = None
        self._current_project_path: Path | None = None
        # Last source loaded (folder or file) — saved into projects so they can
        # re-open the same DICOM data without the user picking it again.
        self._current_source: SourceSettings = SourceSettings()
        # All studies pulled from the most recent folder (or [study] for a file).
        # The series picker switches the active one without re-loading the folder.
        self._loaded_studies: list[Study] = []
        self.export_panel.mesh_ready.connect(self._on_mesh_ready)

        tabs = QTabWidget()
        tabs.addTab(self.windowing_panel, "Windowing")
        tabs.addTab(self.segmentation_panel, "Segmentation")
        tabs.addTab(self.export_panel, "Export")
        dock = QDockWidget("Tools", self)
        dock.setWidget(tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # Series picker toolbar — only meaningful for folders that contain
        # more than one series, but kept always-visible so the user knows
        # which series they're viewing.
        self.series_combo = QComboBox()
        self.series_combo.setMinimumWidth(280)
        self.series_combo.currentIndexChanged.connect(self._on_series_combo_changed)
        series_toolbar = QToolBar("Series", self)
        series_toolbar.setMovable(False)
        series_toolbar.addWidget(QLabel("Series: "))
        series_toolbar.addWidget(self.series_combo)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, series_toolbar)
        self._refresh_series_combo()

        new_project_action = QAction("New Project", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self._on_new_project)
        open_project_action = QAction("Open Project…", self)
        open_project_action.setShortcut("Ctrl+P")
        open_project_action.triggered.connect(self._on_open_project)
        save_project_action = QAction("Save Project", self)
        save_project_action.setShortcut("Ctrl+S")
        save_project_action.triggered.connect(self._on_save_project)
        save_project_as_action = QAction("Save Project As…", self)
        save_project_as_action.setShortcut("Ctrl+Shift+S")
        save_project_as_action.triggered.connect(self._on_save_project_as)
        open_folder_action = QAction("Open DICOM Folder…", self)
        open_folder_action.setShortcut("Ctrl+O")
        open_folder_action.triggered.connect(self._on_open_folder)
        open_file_action = QAction("Open DICOM File…", self)
        open_file_action.setShortcut("Ctrl+Shift+F")  # was Ctrl+Shift+O — freed for Save As
        open_file_action.triggered.connect(self._on_open_file)
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(new_project_action)
        file_menu.addAction(open_project_action)
        file_menu.addAction(save_project_action)
        file_menu.addAction(save_project_as_action)
        file_menu.addSeparator()
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
        self.open_folder_path(Path(folder))

    def open_folder_path(self, folder: Path, preferred_series_uid: str = "") -> bool:
        try:
            result = load_series_from_folder(folder)
        except LoaderError as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return False
        # Cache every study so the user can switch series without reloading.
        self._loaded_studies = list(result.studies)
        # Pick the preferred series if it's there, else the first.
        chosen = self._loaded_studies[0]
        if preferred_series_uid:
            for s in self._loaded_studies:
                if s.series_uid == preferred_series_uid:
                    chosen = s
                    break
        self._current_source = SourceSettings(
            kind="folder", path=str(folder), series_uid=chosen.series_uid
        )
        self._refresh_series_combo(active=chosen)
        self.document.set_study(chosen)
        return True

    def _on_open_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open DICOM File",
            "",
            "DICOM files (*.dcm *.dicom *.DCM *.DICOM);;All files (*)",
        )
        if not path_str:
            return
        self.open_file_path(Path(path_str))

    def open_file_path(self, path: Path) -> bool:
        try:
            result = load_series_from_file(path)
        except LoaderError as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return False
        chosen = result.studies[0]
        self._loaded_studies = list(result.studies)
        self._current_source = SourceSettings(
            kind="file", path=str(path), series_uid=chosen.series_uid
        )
        self._refresh_series_combo(active=chosen)
        self.document.set_study(chosen)
        return True

    def _refresh_series_combo(self, active: Study | None = None) -> None:
        """Repopulate the series dropdown for the current _loaded_studies list."""
        self.series_combo.blockSignals(True)
        try:
            self.series_combo.clear()
            if not self._loaded_studies:
                self.series_combo.addItem("(no study loaded)")
                self.series_combo.setEnabled(False)
                return
            self.series_combo.setEnabled(len(self._loaded_studies) > 1)
            chosen_index = 0
            for i, study in enumerate(self._loaded_studies):
                self.series_combo.addItem(study.display_name)
                if active is not None and study.series_uid == active.series_uid:
                    chosen_index = i
            self.series_combo.setCurrentIndex(chosen_index)
        finally:
            self.series_combo.blockSignals(False)

    def _on_series_combo_changed(self, index: int) -> None:
        if not (0 <= index < len(self._loaded_studies)):
            return
        study = self._loaded_studies[index]
        # Don't reload if it's already the active study (avoids resetting the
        # segmentation when the combo refresh runs).
        if self.document.study is study:
            return
        self._current_source = SourceSettings(
            kind=self._current_source.kind,
            path=self._current_source.path,
            series_uid=study.series_uid,
        )
        self.document.set_study(study)

    # --- project file handlers ---
    def _on_new_project(self) -> None:
        self._current_project_path = None
        self._update_window_title()

    def _on_open_project(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            f"DICOM viewer projects (*{PROJECT_EXTENSION});;All files (*)",
        )
        if not path_str:
            return
        self.load_project_from_path(Path(path_str))

    def _on_save_project(self) -> None:
        if self._current_project_path is None:
            self._on_save_project_as()
            return
        self._save_project_to(self._current_project_path)

    def _on_save_project_as(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            f"project{PROJECT_EXTENSION}",
            f"DICOM viewer projects (*{PROJECT_EXTENSION})",
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != PROJECT_EXTENSION:
            path = path.with_suffix(PROJECT_EXTENSION)
        self._save_project_to(path)

    def _save_project_to(self, path: Path) -> None:
        try:
            save_project(path, self.collect_project())
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        self._current_project_path = path
        save_last_project(path)
        self._update_window_title()

    def load_project_from_path(self, path: Path) -> bool:
        """Public entry point: load a project, used by File menu and CLI."""
        try:
            project = load_project(path)
        except ProjectError as e:
            QMessageBox.warning(self, "Open project failed", str(e))
            return False
        if not self.apply_project(project):
            return False
        self._current_project_path = path
        save_last_project(path)
        self._update_window_title()
        return True

    def collect_project(self) -> Project:
        """Snapshot the current UI state into a Project dataclass."""
        win = self.document.windowing
        region = self.document.region
        if region is None and self.document.volume is not None:
            region = self.document.volume.bbox()
        region_settings = (
            RegionSettings(z=region.z, y=region.y, x=region.x)
            if region is not None
            else RegionSettings()
        )
        return Project(
            source=self._current_source,
            windowing=WindowingSettings(center=win.center, width=win.width),
            segmentation=self.segmentation_panel.get_settings(),
            region=region_settings,
            export=self.export_panel.get_settings(),
        )

    def apply_project(self, project: Project) -> bool:
        """Apply settings from `project`, loading its source first."""
        if project.source.path:
            src = Path(project.source.path)
            if project.source.kind == "folder":
                if not self.open_folder_path(src, preferred_series_uid=project.source.series_uid):
                    return False
            elif project.source.kind == "file":
                if not self.open_file_path(src):
                    return False
            else:
                QMessageBox.warning(
                    self, "Open project failed", f"unknown source kind: {project.source.kind!r}"
                )
                return False
        self.document.set_windowing(
            WindowingState(center=project.windowing.center, width=project.windowing.width)
        )
        if project.region.z != (0, 0) or project.region.y != (0, 0) or project.region.x != (0, 0):
            self.document.set_region(
                Region(z=project.region.z, y=project.region.y, x=project.region.x)
            )
        self.segmentation_panel.apply_settings(project.segmentation)
        self.export_panel.apply_settings(project.export)
        return True

    def _update_window_title(self) -> None:
        base = "DICOM Viewer"
        if self._current_project_path is not None:
            self.setWindowTitle(f"{base} — {self._current_project_path.name}")
        else:
            self.setWindowTitle(base)

    def _on_mesh_ready(self, mesh) -> None:
        # Build a fresh dialog each time. Reusing the dialog (and the
        # QVTKRenderWindowInteractor inside it) led to a blank second preview
        # and an unresponsive close button on some platforms.
        prev_geometry = None
        if self._preview_dialog is not None:
            prev_geometry = self._preview_dialog.geometry()
            self._preview_dialog.close()
            self._preview_dialog.deleteLater()
        self._preview_dialog = MeshPreviewDialog(parent=self)
        if prev_geometry is not None:
            self._preview_dialog.setGeometry(prev_geometry)
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
