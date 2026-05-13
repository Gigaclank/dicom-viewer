"""Export panel — smoothing / decimation options + STL file dialog."""
from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
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
from dicom_viewer.core.mesh_export import EmptyMeshError, ExportOptions, export_stl, generate_mesh


class _ExportWorker(QThread):
    finished_ok = pyqtSignal(str, int)
    failed = pyqtSignal(str)

    def __init__(self, document: Document, options: ExportOptions, out_path: Path) -> None:
        super().__init__()
        self._document = document
        self._options = options
        self._out_path = out_path

    def run(self) -> None:
        try:
            volume = self._document.volume
            seg = self._document.segmentation
            region = self._document.region or (volume.bbox() if volume else None)
            if volume is None or seg is None or region is None:
                raise EmptyMeshError("missing volume / segmentation / region")
            mesh = generate_mesh(volume, seg, region, self._options)
            export_stl(mesh, self._out_path)
            self.finished_ok.emit(str(self._out_path), mesh.triangle_count)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ExportPanel(QWidget):
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

        self.export_button = QPushButton("Export STL…")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setEnabled(False)

        self._status = QLabel("")

        form = QFormLayout()
        form.addRow("Smoothing iterations", self.smoothing_spin)
        form.addRow("Decimation reduction", self.decimation_spin)
        form.addRow(self.manifold_checkbox)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.export_button)
        layout.addWidget(self._status)

        document.subscribe(self._on_doc_event)
        # Initialise button state for documents that already have a segmentation.
        self.export_button.setEnabled(document.segmentation is not None)

    def _on_doc_event(self, kind: str) -> None:
        if kind in ("segmentation", "study"):
            self.export_button.setEnabled(self._document.segmentation is not None)

    def _on_export_clicked(self) -> None:
        out_str, _ = QFileDialog.getSaveFileName(
            self, "Export STL", self._suggested_filename(), "STL files (*.stl)"
        )
        if not out_str:
            return
        self.run_export(Path(out_str))

    def run_export(self, out_path: Path) -> None:
        """Synchronous export entry point — used by the button and by tests."""
        options = ExportOptions(
            smoothing_iterations=self.smoothing_spin.value(),
            decimation_target_reduction=float(self.decimation_spin.value()),
            ensure_manifold=self.manifold_checkbox.isChecked(),
        )
        worker = _ExportWorker(self._document, options, out_path)
        worker.finished_ok.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.start()
        worker.wait()  # synchronous; UI variant runs it async via QThread normally.

    def _on_done(self, path: str, triangle_count: int) -> None:
        self._status.setText(f"Wrote {path} ({triangle_count} triangles)")

    def _on_failed(self, msg: str) -> None:
        self._status.setText(f"Export failed: {msg}")
        QMessageBox.critical(self, "Export failed", msg)

    def _suggested_filename(self) -> str:
        study = self._document.study
        if study is None:
            return "export.stl"
        method = self._document.segmentation.method if self._document.segmentation else "raw"
        raw = f"{study.patient_id or 'anon'}_{study.series_description or 'series'}_{method}"
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
        return f"{sanitized}.stl"
