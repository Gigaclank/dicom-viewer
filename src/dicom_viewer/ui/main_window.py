"""MainWindow — tabbed views (axial / coronal / sagittal / 3D) with tools dock."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QWidget,
)

from dicom_viewer.core.document import Document, WindowingState
from dicom_viewer.core.region import Region
from dicom_viewer.core.study import Study
from dicom_viewer.core.volume import Orientation
from dicom_viewer.io.config import save_last_project
from dicom_viewer.io.dicom_loader import (
    LoaderCancelled,
    LoaderError,
    LoadResult,
    load_series_from_file,
    load_series_from_folder,
)
from dicom_viewer.io.nifti import load_segmentation_from_nifti, save_segmentation_to_nifti
from dicom_viewer.io.project import (
    PROJECT_EXTENSION,
    MaskEntry,
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
from dicom_viewer.ui.status_model import StatusModel
from dicom_viewer.ui.widgets.mesh_preview_view import MeshPreviewView
from dicom_viewer.ui.widgets.slice_view import SliceView
from dicom_viewer.ui.widgets.volume3d_view import Volume3DView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DICOM Viewer")
        self.resize(1400, 900)

        self.document = Document()
        # Single source of truth for 'what's the app currently doing'. Panels
        # and workers register tasks here; the status bar renders the set.
        self.status_model = StatusModel()

        self.axial = SliceView(Orientation.AXIAL)
        self.coronal = SliceView(Orientation.CORONAL)
        self.sagittal = SliceView(Orientation.SAGITTAL)
        self.volume3d = Volume3DView()

        self.windowing_panel = WindowingPanel(self.document)
        self.segmentation_panel = SegmentationPanel(self.document, status=self.status_model)
        self.export_panel = ExportPanel(self.document, status=self.status_model)
        self.stl_preview = MeshPreviewView(self.document, self.export_panel, status=self.status_model)

        # --- quad view: all four anatomical panes visible at once ---
        # Standard radiology layout: AX top-left, COR top-right, SAG bottom-
        # left, 3D bottom-right. Nested QSplitters let the user resize each
        # row and column by dragging the dividers.
        top_row = QSplitter(Qt.Orientation.Horizontal)
        top_row.addWidget(self.axial)
        top_row.addWidget(self.coronal)
        top_row.setStretchFactor(0, 1)
        top_row.setStretchFactor(1, 1)
        bottom_row = QSplitter(Qt.Orientation.Horizontal)
        bottom_row.addWidget(self.sagittal)
        bottom_row.addWidget(self.volume3d)
        bottom_row.setStretchFactor(0, 1)
        bottom_row.setStretchFactor(1, 1)
        self.quad_view = QSplitter(Qt.Orientation.Vertical)
        self.quad_view.addWidget(top_row)
        self.quad_view.addWidget(bottom_row)
        self.quad_view.setStretchFactor(0, 1)
        self.quad_view.setStretchFactor(1, 1)

        # STL preview stays accessible as a separate tab so it doesn't have
        # to compete with the four anatomical panes for screen real estate.
        self.view_tabs = QTabWidget()
        self.view_tabs.addTab(self.quad_view, "Multi-view")
        self.view_tabs.addTab(self.stl_preview, "STL Preview")
        self.view_tabs.currentChanged.connect(self._on_view_tab_changed)
        self.setCentralWidget(self.view_tabs)

        self._current_project_path: Path | None = None
        # Last source loaded (folder or file) — saved into projects so they can
        # re-open the same DICOM data without the user picking it again.
        self._current_source: SourceSettings = SourceSettings()
        # All studies pulled from the most recent folder (or [study] for a file).
        # The series picker switches the active one without re-loading the folder.
        self._loaded_studies: list[Study] = []

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

        # --- click-seed tumor brush wiring ---
        # Panel owns the mode; mainwindow mirrors it to all three slice views
        # so the user can drop seeds in whichever pane shows the tumor best.
        self.segmentation_panel.brush_mode_changed.connect(self._on_brush_mode_changed)
        for sv in (self.axial, self.coronal, self.sagittal):
            sv.seed_clicked.connect(self.segmentation_panel.handle_seed_click)
            # Drag events carry the originating orientation so the 2D paint
            # brush knows which plane it's painting on.
            sv.seed_dragged.connect(
                lambda seed, mode, _o=sv.orientation: self.segmentation_panel.handle_seed_drag(
                    seed, mode, _o
                )
            )

        # --- 3D crosshair wiring ---
        # Each 2D pane drives one axis of the crosshair on the 3D view, so
        # scrubbing through axial slices moves the red plane, etc.
        self.axial.slice_changed.connect(self._on_axial_index_changed)
        self.coronal.slice_changed.connect(self._on_coronal_index_changed)
        self.sagittal.slice_changed.connect(self._on_sagittal_index_changed)

        # --- bottom status bar: "Currently doing: …" ---
        # Always visible at the bottom of the window. Each long operation
        # registers a task on self.status_model; this label re-renders on
        # every change. Helpful for debugging "why does the app feel slow?"
        # — at a glance the user can tell whether a worker is in flight.
        self._status_label = QLabel("Idle")
        self._status_label.setObjectName("currentlyDoingLabel")
        sbar: QStatusBar = self.statusBar()
        sbar.addWidget(self._status_label, stretch=1)
        self.status_model.changed.connect(self._refresh_status_bar)

        # --- 3D overlay rebuild debounce ---
        # set_overlay_mask on the 3D view runs vtkDiscreteMarchingCubes which
        # is expensive on a full-volume mask. Brush bursts (many clicks in a
        # second) would otherwise trigger one mesh rebuild per click and
        # freeze the window. 2D slice overlays still update immediately —
        # they're cheap. We collapse 3D rebuilds into a single run that
        # fires once the user pauses.
        self._pending_3d_mask = None
        self._overlay_3d_debounce = QTimer(self)
        self._overlay_3d_debounce.setSingleShot(True)
        self._overlay_3d_debounce.setInterval(250)
        self._overlay_3d_debounce.timeout.connect(self._apply_pending_3d_overlay)

        self.document.subscribe(self._on_doc_event)

    def _on_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open DICOM Folder")
        if not folder:
            return
        self.open_folder_path(Path(folder))

    def open_folder_path(self, folder: Path, preferred_series_uid: str = "") -> bool:
        # Run the load in a worker thread with a modal progress dialog so the
        # UI stays responsive on big folders (500+ files = many seconds).
        worker = _FolderLoadWorker(Path(folder))
        self.status_model.begin("folder_load", "Loading DICOM folder…")
        dialog = QProgressDialog("Scanning folder…", "Cancel", 0, 100, self)
        dialog.setWindowTitle("Loading DICOM folder")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        # By default QProgressDialog auto-closes / auto-resets when value hits
        # 100, and the reset emits canceled(). That makes the worker's normal
        # 'Done, 1.0' progress event look like a user cancel. Turn both off
        # and close the dialog explicitly from the worker callbacks instead.
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        # Cancel: ask the worker (politely) to stop. The worker's load loop
        # checks isInterruptionRequested between files and between decoded
        # slices, and short-circuits via LoaderCancelled.
        dialog.canceled.connect(worker.requestInterruption)

        state: dict[str, object] = {}

        def on_progress(stage: str, fraction: float) -> None:
            dialog.setLabelText(stage)
            dialog.setValue(int(max(0.0, min(1.0, fraction)) * 100))
            self.status_model.update(
                "folder_load", f"Loading DICOM folder — {stage}"
            )

        def on_finished_ok(result: LoadResult) -> None:
            state["result"] = result
            dialog.close()
            self.status_model.end("folder_load")

        def on_failed(msg: str) -> None:
            state["error"] = msg
            dialog.close()
            self.status_model.end("folder_load")

        def on_cancelled() -> None:
            state["cancelled"] = True
            dialog.close()
            self.status_model.end("folder_load")

        worker.progress.connect(on_progress)
        worker.finished_ok.connect(on_finished_ok)
        worker.failed.connect(on_failed)
        worker.cancelled.connect(on_cancelled)
        worker.start()
        dialog.exec()
        worker.wait()

        # Result wins over cancel — a real completion that races with a
        # spurious cancel signal must still load the data.
        result = state.get("result")
        if isinstance(result, LoadResult):
            pass  # fall through to the success path below
        elif "error" in state:
            QMessageBox.warning(self, "Load failed", str(state["error"]))
            return False
        elif state.get("cancelled"):
            return False
        else:
            return False

        # Cache every study so the user can switch series without reloading.
        self._loaded_studies = list(result.studies)
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
            "Open DICOM or NIfTI File",
            "",
            "Volume files (*.dcm *.dicom *.DCM *.DICOM *.nii *.nii.gz);;"
            "DICOM (*.dcm *.dicom *.DCM *.DICOM);;"
            "NIfTI (*.nii *.nii.gz);;"
            "All files (*)",
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
            # Write companion mask files BEFORE the JSON so the file lookup
            # paths are stable. _write_mask_companions returns the
            # MaskEntry list to embed in the project.
            mask_entries = self._write_mask_companions(path)
            project = self.collect_project()
            project.masks = mask_entries
            project.active_mask = self.document.active_mask_name
            save_project(path, project)
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        self._current_project_path = path
        save_last_project(path)
        self._update_window_title()

    def _write_mask_companions(self, project_path: Path) -> list[MaskEntry]:
        """Write each library mask as <project-stem>.<mask-name>.nii.gz
        alongside `project_path`. Returns MaskEntry list with names + paths
        relative to project_path's parent."""
        volume = self.document.volume
        if volume is None or not self.document.mask_names:
            return []
        stem = project_path.stem
        parent = project_path.parent
        entries: list[MaskEntry] = []
        for name in self.document.mask_names:
            seg = self.document.get_mask(name)
            if seg is None:
                continue
            # Sanitize the mask name for the filesystem; the displayed name
            # stays unchanged in the project JSON.
            safe = name.replace("/", "_").replace("\\", "_")
            mask_file = parent / f"{stem}.{safe}.nii.gz"
            save_segmentation_to_nifti(seg, volume, mask_file)
            entries.append(MaskEntry(name=name, path=mask_file.name))
        return entries

    def load_project_from_path(self, path: Path) -> bool:
        """Public entry point: load a project, used by File menu and CLI."""
        try:
            project = load_project(path)
        except ProjectError as e:
            QMessageBox.warning(self, "Open project failed", str(e))
            return False
        if not self.apply_project(project, project_path=path):
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

    def apply_project(self, project: Project, project_path: Path | None = None) -> bool:
        """Apply settings from `project`, loading its source first.

        `project_path` is the on-disk location of the project file, used to
        resolve relative mask companion paths. CLI / programmatic callers
        can omit it; in that case masks aren't loaded.
        """
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
        # Load companion masks last so the segmentation settings panel's
        # auto-applied result is replaced by whichever named mask the user
        # had active when they saved the project.
        if project_path is not None and project.masks:
            self._load_mask_companions(project_path, project.masks, project.active_mask)
        return True

    def _load_mask_companions(
        self,
        project_path: Path,
        entries: list[MaskEntry],
        active_name: str,
    ) -> None:
        volume = self.document.volume
        if volume is None:
            return
        parent = project_path.parent
        loaded: dict[str, "Segmentation"] = {}  # type: ignore[name-defined]
        skipped: list[str] = []
        for entry in entries:
            if not entry.name or not entry.path:
                continue
            mask_path = parent / entry.path
            if not mask_path.exists():
                skipped.append(f"{entry.name} ({entry.path}: not found)")
                continue
            try:
                seg = load_segmentation_from_nifti(mask_path, volume)
            except Exception as e:  # noqa: BLE001
                skipped.append(f"{entry.name} ({type(e).__name__}: {e})")
                continue
            loaded[entry.name] = seg
        if loaded:
            self.document.replace_masks(loaded, active_name=active_name)
        if skipped:
            QMessageBox.warning(
                self,
                "Some masks couldn't be loaded",
                "The following masks were skipped:\n  • " + "\n  • ".join(skipped),
            )

    def _update_window_title(self) -> None:
        base = "DICOM Viewer"
        if self._current_project_path is not None:
            self.setWindowTitle(f"{base} — {self._current_project_path.name}")
        else:
            self.setWindowTitle(base)

    def _on_view_tab_changed(self, index: int) -> None:
        # Tell the STL preview widget whether its tab is currently visible so
        # it only does the (potentially expensive) mesh pipeline work when the
        # user is actually looking at it.
        self.stl_preview.set_tab_visible(self.view_tabs.widget(index) is self.stl_preview)

    def _on_brush_mode_changed(self, mode: str) -> None:
        for sv in (self.axial, self.coronal, self.sagittal):
            sv.set_brush_mode(mode)

    # --- crosshair handlers ---
    def _push_crosshair_to_3d(self) -> None:
        """Send the current AX/COR/SAG slice indices to the 3D view so the
        translucent indicator planes match what the 2D panes are showing."""
        self.volume3d.set_crosshair_position(
            self.axial.current_index,
            self.coronal.current_index,
            self.sagittal.current_index,
        )

    def _on_axial_index_changed(self, _value: int) -> None:
        self._push_crosshair_to_3d()

    def _on_coronal_index_changed(self, _value: int) -> None:
        self._push_crosshair_to_3d()

    def _on_sagittal_index_changed(self, _value: int) -> None:
        self._push_crosshair_to_3d()

    def _refresh_status_bar(self) -> None:
        self._status_label.setText(self.status_model.render())

    def _apply_pending_3d_overlay(self) -> None:
        """Flush the most recent segmentation mask into the 3D overlay.
        Triggered by the debounce timer to coalesce rapid updates."""
        self.volume3d.set_overlay_mask(self._pending_3d_mask)

    def _on_reset_views(self) -> None:
        self.axial.reset_view()
        self.coronal.reset_view()
        self.sagittal.reset_view()
        self.volume3d.reset_view()
        self.stl_preview._on_reset_clicked()

    def _on_doc_event(self, kind: str) -> None:
        volume = self.document.volume
        if volume is None:
            return
        if kind == "study":
            self.axial.set_volume(volume)
            self.coronal.set_volume(volume)
            self.sagittal.set_volume(volume)
            self.volume3d.set_volume(volume)
            # The 2D panes default to their middle slice; mirror that on the
            # 3D crosshair so all four panes start in agreement.
            self._push_crosshair_to_3d()
        if kind in ("study", "windowing"):
            w = self.document.windowing
            self.axial.set_windowing(w.center, w.width)
            self.coronal.set_windowing(w.center, w.width)
            self.sagittal.set_windowing(w.center, w.width)
            # The 3D view's opacity ramp also tracks windowing — without
            # this, switching to Lung/Soft-tissue presets would leave the
            # 3D pane frozen at the previous iso threshold.
            self.volume3d.set_windowing(w.center, w.width)
        if kind == "segmentation":
            mask = self.document.segmentation.mask if self.document.segmentation else None
            # 2D overlays update on every change — slicing a 3D mask is cheap.
            self.axial.set_overlay_mask(mask)
            self.coronal.set_overlay_mask(mask)
            self.sagittal.set_overlay_mask(mask)
            # 3D overlay rebuild is expensive (marching cubes). Debounce it
            # so rapid brush clicks don't queue up a mesh rebuild per click.
            self._pending_3d_mask = mask
            self._overlay_3d_debounce.start()
        if kind in ("study", "region"):
            self.volume3d.set_region(self.document.region)


class _FolderLoadWorker(QThread):
    """Runs load_series_from_folder off the UI thread, reports progress.

    Cancellation: the UI calls requestInterruption() on the QThread; the
    loader's should_cancel callback is hooked to isInterruptionRequested().
    """

    progress = pyqtSignal(str, float)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, folder: Path) -> None:
        super().__init__()
        self._folder = folder

    def run(self) -> None:
        try:
            result = load_series_from_folder(
                self._folder,
                progress=lambda stage, frac: self.progress.emit(stage, frac),
                should_cancel=self.isInterruptionRequested,
            )
            self.finished_ok.emit(result)
        except LoaderCancelled:
            self.cancelled.emit()
        except LoaderError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
