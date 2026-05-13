"""Export panel — smoothing / decimation options, mesh preview, STL file dialog."""
from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import QCoreApplication, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.document import Document
from dicom_viewer.core.mesh_export import (
    EmptyMeshError,
    ExportOptions,
    Mesh,
    export_stl,
    generate_mesh,
)
from dicom_viewer.io.project import ExportSettings
from dicom_viewer.ui.widgets.labeled_slider import LabeledFloatSlider, LabeledSlider


class _ExportWorker(QThread):
    """Generates a mesh and optionally writes it to disk.

    Emits mesh_ready(Mesh) whenever a mesh is produced (preview OR export).
    Emits finished_ok(path, triangle_count) only when written to disk.
    Emits failed(msg) on any error.
    """

    mesh_ready = pyqtSignal(object)
    finished_ok = pyqtSignal(str, int)
    failed = pyqtSignal(str)

    def __init__(
        self,
        document: Document,
        options: ExportOptions,
        out_path: Path | None,
    ) -> None:
        super().__init__()
        self._document = document
        self._options = options
        self._out_path = out_path  # None => preview only

    def run(self) -> None:
        try:
            volume = self._document.volume
            seg = self._document.segmentation
            region = self._document.region or (volume.bbox() if volume else None)
            if volume is None or seg is None or region is None:
                raise EmptyMeshError("missing volume / segmentation / region")
            mesh = generate_mesh(volume, seg, region, self._options)
            self.mesh_ready.emit(mesh)
            if self._out_path is not None:
                export_stl(mesh, self._out_path)
                self.finished_ok.emit(str(self._out_path), mesh.triangle_count)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ExportPanel(QWidget):
    """STL export controls.

    `settings_changed` fires whenever a slider/checkbox value changes so the
    STL Preview tab can refresh. `mesh_ready` still fires after a successful
    Export STL (some tests rely on it).
    """

    settings_changed = pyqtSignal()
    mesh_ready = pyqtSignal(object)

    def __init__(self, document: Document) -> None:
        super().__init__()
        self._document = document

        # Reasonable practical limits — values above 50 smoothing iterations or
        # above ~0.9 decimation rarely produce better results.
        self.smoothing_slider = LabeledSlider(0, 50, 15)
        self.decimation_slider = LabeledFloatSlider(0.0, 0.95, 0.5, step=0.05, decimals=2)

        self.manifold_checkbox = QCheckBox("Ensure manifold (recommended)")
        self.manifold_checkbox.setChecked(True)

        self.smoothing_slider.valueChanged.connect(lambda _v: self.settings_changed.emit())
        self.decimation_slider.valueChanged.connect(lambda _v: self.settings_changed.emit())
        self.manifold_checkbox.toggled.connect(lambda _on: self.settings_changed.emit())

        self.export_button = QPushButton("Export STL…")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setEnabled(False)

        self._status = QLabel("")
        # Word-wrap so a long path or message doesn't widen the dock column.
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        form = QFormLayout()
        form.addRow("Smoothing iterations", self.smoothing_slider)
        form.addRow("Decimation reduction", self.decimation_slider)
        form.addRow(self.manifold_checkbox)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.export_button)
        layout.addWidget(self._status)

        document.subscribe(self._on_doc_event)
        # Initialise button state for documents that already have a segmentation.
        self._refresh_buttons()

    # --- doc observers ---
    def _on_doc_event(self, kind: str) -> None:
        if kind in ("segmentation", "study"):
            self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        self.export_button.setEnabled(self._document.segmentation is not None)

    # --- actions ---
    def _on_export_clicked(self) -> None:
        out_str, _ = QFileDialog.getSaveFileName(
            self, "Export STL", self._suggested_filename(), "STL files (*.stl)"
        )
        if not out_str:
            return
        self.run_export(Path(out_str))

    def run_export(self, out_path: Path) -> None:
        """Synchronous export entry point — used by the button and by tests."""
        self._run_worker(out_path=out_path)

    def _run_worker(self, out_path: Path | None) -> None:
        options = ExportOptions(
            smoothing_iterations=self.smoothing_slider.value(),
            decimation_target_reduction=self.decimation_slider.float_value(),
            ensure_manifold=self.manifold_checkbox.isChecked(),
        )
        worker = _ExportWorker(self._document, options, out_path)
        worker.mesh_ready.connect(self._on_mesh_ready)
        worker.finished_ok.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.start()
        worker.wait()
        # Cross-thread signals were queued onto the main thread while it was
        # blocked in wait(); drain the queue so slots run before we return.
        QCoreApplication.processEvents()

    # --- worker callbacks ---
    def _on_mesh_ready(self, mesh: Mesh) -> None:
        self._status.setText(f"Generated mesh — {mesh.triangle_count:,} triangles")
        self._status.setToolTip("")
        self.mesh_ready.emit(mesh)

    def _on_done(self, path: str, triangle_count: int) -> None:
        fname = Path(path).name
        self._status.setText(f"Wrote {fname} ({triangle_count:,} triangles)")
        self._status.setToolTip(path)  # full path on hover

    def _on_failed(self, msg: str) -> None:
        self._status.setText(f"Export failed: {msg}")
        self._status.setToolTip("")
        QMessageBox.critical(self, "Export failed", msg)

    # --- live access for consumers (preview tab) ---
    def get_export_options(self) -> ExportOptions:
        """Snapshot of the export controls as the dataclass generate_mesh wants."""
        return ExportOptions(
            smoothing_iterations=self.smoothing_slider.value(),
            decimation_target_reduction=self.decimation_slider.float_value(),
            ensure_manifold=self.manifold_checkbox.isChecked(),
        )

    # --- project file integration ---
    def get_settings(self) -> ExportSettings:
        return ExportSettings(
            smoothing_iterations=self.smoothing_slider.value(),
            decimation_reduction=self.decimation_slider.float_value(),
            ensure_manifold=self.manifold_checkbox.isChecked(),
        )

    def apply_settings(self, s: ExportSettings) -> None:
        self.smoothing_slider.setValue(s.smoothing_iterations)
        self.decimation_slider.setFloatValue(s.decimation_reduction)
        self.manifold_checkbox.setChecked(s.ensure_manifold)

    # --- helpers ---
    def _suggested_filename(self) -> str:
        study = self._document.study
        if study is None:
            return "export.stl"
        method = self._document.segmentation.method if self._document.segmentation else "raw"
        raw = f"{study.patient_id or 'anon'}_{study.series_description or 'series'}_{method}"
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
        return f"{sanitized}.stl"


