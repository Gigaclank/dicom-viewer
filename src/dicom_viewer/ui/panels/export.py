"""Export panel — smoothing / decimation options, mesh preview, STL file dialog."""
from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import QCoreApplication, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
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
    """STL export controls plus a 'Preview mesh' action.

    `mesh_ready` is emitted whenever generate_mesh produces a result; consumers
    (e.g. MeshPreviewDialog) can subscribe to update their display.
    """

    mesh_ready = pyqtSignal(object)  # forwards _ExportWorker.mesh_ready

    def __init__(self, document: Document) -> None:
        super().__init__()
        self._document = document

        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(0, 200)
        self.smoothing_spin.setValue(15)

        self.decimation_spin = QDoubleSpinBox()
        self.decimation_spin.setRange(0.0, 0.95)
        self.decimation_spin.setSingleStep(0.05)
        self.decimation_spin.setValue(0.5)

        self.manifold_checkbox = QCheckBox("Ensure manifold (recommended)")
        self.manifold_checkbox.setChecked(True)

        self.preview_button = QPushButton("Preview mesh")
        self.preview_button.clicked.connect(self._on_preview_clicked)
        self.preview_button.setEnabled(False)

        self.export_button = QPushButton("Export STL…")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setEnabled(False)

        self._status = QLabel("")
        # Word-wrap so a long path or message doesn't widen the dock column.
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        form = QFormLayout()
        form.addRow("Smoothing iterations", self.smoothing_spin)
        form.addRow("Decimation reduction", self.decimation_spin)
        form.addRow(self.manifold_checkbox)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.preview_button)
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
        ok = self._document.segmentation is not None
        self.preview_button.setEnabled(ok)
        self.export_button.setEnabled(ok)

    # --- actions ---
    def _on_preview_clicked(self) -> None:
        self._run_worker(out_path=None)

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
            smoothing_iterations=self.smoothing_spin.value(),
            decimation_target_reduction=float(self.decimation_spin.value()),
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

    # --- helpers ---
    def _suggested_filename(self) -> str:
        study = self._document.study
        if study is None:
            return "export.stl"
        method = self._document.segmentation.method if self._document.segmentation else "raw"
        raw = f"{study.patient_id or 'anon'}_{study.series_description or 'series'}_{method}"
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
        return f"{sanitized}.stl"


