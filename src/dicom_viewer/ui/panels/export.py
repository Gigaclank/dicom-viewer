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
    QProgressDialog,
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
    resolve_export_segmentation,
)
from dicom_viewer.io.nifti import save_segmentation_to_nifti
from dicom_viewer.io.project import ExportSettings
from dicom_viewer.ui.widgets.labeled_slider import LabeledFloatSlider, LabeledSlider


class _ExportWorker(QThread):
    """Generates a mesh and optionally writes it to disk.

    Emits progress(stage, fraction) at each pipeline stage.
    Emits mesh_ready(Mesh) whenever a mesh is produced (preview OR export).
    Emits finished_ok(path, triangle_count) only when written to disk.
    Emits failed(msg) on any error.
    """

    progress = pyqtSignal(str, float)
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
            if volume is None:
                raise EmptyMeshError("no volume loaded")
            seg, _label = resolve_export_segmentation(
                volume,
                self._document.segmentation,
                float(self._document.windowing.center),
            )
            region = self._document.region or volume.bbox()
            mesh = generate_mesh(
                volume,
                seg,
                region,
                self._options,
                progress=lambda stage, frac: self.progress.emit(stage, frac),
            )
            self.mesh_ready.emit(mesh)
            if self._out_path is not None:
                self.progress.emit("Writing STL", 0.98)
                export_stl(mesh, self._out_path)
                self.progress.emit("Done", 1.0)
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
        self.export_button.setToolTip(
            "Save the current scene as a 3D-printable STL. If you've applied "
            "a segmentation, that's what's exported. Otherwise the iso-surface "
            "at the active windowing center is meshed — adjust windowing to "
            "control which structure is captured."
        )
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setEnabled(False)

        self.export_nifti_button = QPushButton("Export mask as NIfTI…")
        self.export_nifti_button.setToolTip(
            "Save the current segmentation mask as a NIfTI (.nii.gz) file. "
            "Voxel grid matches the source volume."
        )
        self.export_nifti_button.clicked.connect(self._on_export_nifti_clicked)
        self.export_nifti_button.setEnabled(False)

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
        layout.addWidget(self.export_nifti_button)
        layout.addWidget(self._status)

        document.subscribe(self._on_doc_event)
        # Initialise button state for documents that already have a study.
        self._refresh_buttons()

    # --- doc observers ---
    def _on_doc_event(self, kind: str) -> None:
        if kind in ("segmentation", "study"):
            self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        # STL export works with or without a segmentation — falls back to
        # window-center iso-surface when no mask exists. Enables on volume.
        self.export_button.setEnabled(self._document.volume is not None)
        # NIfTI mask export only makes sense when there's a mask to save.
        self.export_nifti_button.setEnabled(self._document.segmentation is not None)

    def _on_export_nifti_clicked(self) -> None:
        volume = self._document.volume
        seg = self._document.segmentation
        if volume is None or seg is None:
            return
        suggested = self._suggested_filename().rsplit(".", 1)[0] + ".nii.gz"
        out_str, _ = QFileDialog.getSaveFileName(
            self, "Export Mask as NIfTI", suggested, "NIfTI files (*.nii.gz *.nii)"
        )
        if not out_str:
            return
        out = Path(out_str)
        if not (out.name.endswith(".nii") or out.name.endswith(".nii.gz")):
            out = out.with_name(out.name + ".nii.gz")
        try:
            save_segmentation_to_nifti(seg, volume, out)
        except Exception as e:  # noqa: BLE001
            self._status.setText(f"NIfTI export failed: {e}")
            QMessageBox.critical(self, "Export failed", str(e))
            return
        self._status.setText(f"Wrote {out.name}")
        self._status.setToolTip(str(out))

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

        # Show a modal progress dialog while the pipeline runs — marching
        # cubes + smoothing + manifold fix can take half a minute on a big
        # CT and the UI shouldn't look frozen.
        dialog = QProgressDialog("Generating mesh…", None, 0, 100, self)
        dialog.setWindowTitle("Exporting STL" if out_path is not None else "Generating mesh")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        # No cancel — generate_mesh's VTK filters don't expose abort hooks.
        dialog.setCancelButton(None)  # type: ignore[arg-type]

        def on_progress(stage: str, fraction: float) -> None:
            dialog.setLabelText(stage)
            dialog.setValue(int(max(0.0, min(1.0, fraction)) * 100))

        def on_done(_path: str, _triangle_count: int) -> None:
            dialog.close()

        def on_failed(_msg: str) -> None:
            dialog.close()

        worker.progress.connect(on_progress)
        worker.finished_ok.connect(on_done)
        worker.failed.connect(on_failed)
        # Preview path (out_path=None) doesn't fire finished_ok — close on
        # mesh_ready too so the dialog goes away.
        if out_path is None:
            worker.mesh_ready.connect(lambda _mesh: dialog.close())

        worker.start()
        dialog.exec()
        worker.wait()
        # Cross-thread signals were queued onto the main thread while it was
        # blocked in wait(); drain the queue so slots run before we return.
        QCoreApplication.processEvents()

    # --- worker callbacks ---
    def _on_mesh_ready(self, mesh: Mesh) -> None:
        # Tag the status with whether we used a real segmentation or the
        # fallback iso-surface so the user is never surprised by the output.
        is_iso = self._document.segmentation is None
        suffix = "  (iso-surface — no segmentation)" if is_iso else ""
        self._status.setText(
            f"Generated mesh — {mesh.triangle_count:,} triangles{suffix}"
        )
        self._status.setToolTip("")
        self.mesh_ready.emit(mesh)

    def _on_done(self, path: str, triangle_count: int) -> None:
        fname = Path(path).name
        is_iso = self._document.segmentation is None
        suffix = " (iso-surface)" if is_iso else ""
        self._status.setText(f"Wrote {fname} ({triangle_count:,} triangles){suffix}")
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
        seg = self._document.segmentation
        method = seg.method if seg is not None else "iso"
        raw = f"{study.patient_id or 'anon'}_{study.series_description or 'series'}_{method}"
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
        return f"{sanitized}.stl"
