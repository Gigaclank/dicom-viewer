"""Segmentation panel — threshold + region-grow methods with live preview."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.document import Document
from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.morphology import keep_largest_component, smooth_mask
from dicom_viewer.core.segmentation.region_grow import region_grow
from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.volume import Volume
from dicom_viewer.io.nifti import load_segmentation_from_nifti
from dicom_viewer.io.project import SegmentationSettings
from dicom_viewer.ui.widgets.labeled_slider import LabeledSlider


class SegmentationPanel(QWidget):
    def __init__(self, document: Document) -> None:
        super().__init__()
        self._document = document
        self._suppress_live = False

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Threshold", "Region grow"])
        self.method_combo.currentTextChanged.connect(self._on_method_changed)

        self.low_slider = LabeledSlider(-2000, 10000, 300)
        self.high_slider = LabeledSlider(-2000, 10000, 3000)
        self.low_slider.valueChanged.connect(self._on_threshold_changed)
        self.high_slider.valueChanged.connect(self._on_threshold_changed)

        # Region-grow seed coordinates: ranges adapt to the loaded volume's
        # shape; before any volume is loaded they cap at 4096 (well past any
        # practical scan dimension).
        self.seed_z = LabeledSlider(0, 4096, 0)
        self.seed_y = LabeledSlider(0, 4096, 0)
        self.seed_x = LabeledSlider(0, 4096, 0)
        self.tolerance_slider = LabeledSlider(0, 1000, 100)

        self.largest_component_checkbox = QCheckBox("Keep largest connected component")
        self.largest_component_checkbox.setChecked(True)
        self.largest_component_checkbox.toggled.connect(self._on_threshold_changed)
        self.smooth_checkbox = QCheckBox("Smooth mask (close + open)")
        self.smooth_checkbox.setChecked(False)
        self.smooth_checkbox.toggled.connect(self._on_threshold_changed)

        self.live_preview_checkbox = QCheckBox("Live preview")
        self.live_preview_checkbox.setChecked(True)
        self.live_preview_checkbox.setToolTip(
            "Recompute the threshold mask as you drag the sliders. "
            "Turn off if interaction feels slow on large volumes."
        )

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._on_apply)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(lambda: document.set_segmentation(None))
        self.import_button = QPushButton("Import mask…")
        self.import_button.setToolTip(
            "Load a segmentation mask from a NIfTI (.nii / .nii.gz) file. "
            "Shape must match the current volume."
        )
        self.import_button.clicked.connect(self._on_import_clicked)
        self.medsam_button = QPushButton("Segment with MedSAM")
        self.medsam_button.setToolTip(
            "Run MedSAM (SAM-based medical segmenter) over the z-slices of "
            "the active region. The xy bounding box of the region is used as "
            "the prompt. Requires the `medsam` extra: pip install 'dicom-viewer[medsam]'."
        )
        self.medsam_button.clicked.connect(self._on_medsam_clicked)

        self._status = QLabel("No segmentation")
        self._status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Method", self.method_combo)
        form.addRow("Low", self.low_slider)
        form.addRow("High", self.high_slider)

        form.addRow("Seed z", self.seed_z)
        form.addRow("Seed y", self.seed_y)
        form.addRow("Seed x", self.seed_x)
        form.addRow("Tolerance", self.tolerance_slider)
        form.addRow(self.largest_component_checkbox)
        form.addRow(self.smooth_checkbox)
        form.addRow(self.live_preview_checkbox)

        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.clear_button)
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.medsam_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self._status)

        document.subscribe(self._on_doc_event)
        # If a study is already loaded, adapt slider ranges immediately.
        self._refresh_slider_range()

    # --- live preview path ---
    def _on_threshold_changed(self, _value=None) -> None:
        if self._suppress_live:
            return
        if not self.live_preview_checkbox.isChecked():
            return
        if self.method_combo.currentText() != "Threshold":
            return
        self._apply_now()

    def _on_method_changed(self, _text: str) -> None:
        # Method switch is also a moment to refresh the live preview if eligible.
        self._on_threshold_changed()

    def _on_import_clicked(self) -> None:
        volume = self._document.volume
        if volume is None:
            QMessageBox.warning(self, "Import failed", "Load a volume first.")
            return
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Import segmentation mask (NIfTI)",
            "",
            "NIfTI files (*.nii *.nii.gz);;All files (*)",
        )
        if not path_str:
            return
        try:
            seg = load_segmentation_from_nifti(Path(path_str), volume)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Import failed", f"{type(e).__name__}: {e}")
            return
        self._document.set_segmentation(seg)

    def _on_medsam_clicked(self) -> None:
        # Late import so the segmentation panel doesn't pull torch into the
        # base install just by being imported.
        from dicom_viewer.core.segmentation.medsam import MedSAMSegmenter

        if not MedSAMSegmenter.is_available():
            QMessageBox.warning(
                self,
                "MedSAM not available",
                "MedSAM requires extra dependencies.\n\n"
                "Install with:\n    pip install 'dicom-viewer[medsam]'\n\n"
                "First run will download ~360MB of model weights from HuggingFace.",
            )
            return

        volume = self._document.volume
        region = self._document.region
        if volume is None:
            QMessageBox.warning(self, "MedSAM", "Load a volume first.")
            return
        if region is None or region.is_empty:
            QMessageBox.warning(
                self,
                "MedSAM",
                "Set an active region first — its xy bounding box is the prompt "
                "and its z range selects which slices to segment.",
            )
            return

        worker = _MedSAMWorker(volume, region)
        dialog = QProgressDialog("Loading MedSAM model…", "Cancel", 0, 100, self)
        dialog.setWindowTitle("MedSAM segmentation")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.canceled.connect(worker.requestInterruption)

        state: dict[str, object] = {}

        def on_progress(stage: str, fraction: float) -> None:
            dialog.setLabelText(stage)
            dialog.setValue(int(max(0.0, min(1.0, fraction)) * 100))

        def on_done(seg) -> None:
            state["result"] = seg
            dialog.close()

        def on_failed(msg: str) -> None:
            state["error"] = msg
            dialog.close()

        worker.progress.connect(on_progress)
        worker.finished_ok.connect(on_done)
        worker.failed.connect(on_failed)
        worker.start()
        dialog.exec()
        worker.wait()

        if "error" in state:
            QMessageBox.warning(self, "MedSAM failed", str(state["error"]))
            return
        seg = state.get("result")
        if seg is not None:
            self._document.set_segmentation(seg)

    def _on_apply(self) -> None:
        self._apply_now()

    def _apply_now(self) -> None:
        volume = self._document.volume
        if volume is None:
            return
        method = self.method_combo.currentText()
        if method == "Threshold":
            lo, hi = self.low_slider.value(), self.high_slider.value()
            if hi < lo:
                return  # ignore invalid range; user is mid-drag
            seg = threshold(volume, lo, hi)
        else:
            seg = region_grow(
                volume,
                seed=(self.seed_z.value(), self.seed_y.value(), self.seed_x.value()),
                tolerance=self.tolerance_slider.value(),
            )
        if self.largest_component_checkbox.isChecked():
            seg = keep_largest_component(seg)
        if self.smooth_checkbox.isChecked():
            seg = smooth_mask(seg, iterations=1)
        self._document.set_segmentation(seg)

    # --- doc observer ---
    def _on_doc_event(self, kind: str) -> None:
        if kind == "study":
            self._refresh_slider_range()
        if kind == "segmentation":
            seg = self._document.segmentation
            if seg is None:
                self._status.setText("No segmentation")
            else:
                self._status.setText(f"{seg.method} — {seg.voxel_count:,} voxels")

    # --- project file integration ---
    def get_settings(self) -> SegmentationSettings:
        return SegmentationSettings(
            method=self.method_combo.currentText(),
            low=self.low_slider.value(),
            high=self.high_slider.value(),
            seed_z=self.seed_z.value(),
            seed_y=self.seed_y.value(),
            seed_x=self.seed_x.value(),
            tolerance=self.tolerance_slider.value(),
            keep_largest_component=self.largest_component_checkbox.isChecked(),
            smooth=self.smooth_checkbox.isChecked(),
            live_preview=self.live_preview_checkbox.isChecked(),
        )

    def apply_settings(self, s: SegmentationSettings) -> None:
        self._suppress_live = True
        try:
            idx = self.method_combo.findText(s.method)
            if idx >= 0:
                self.method_combo.setCurrentIndex(idx)
            # The slider range was sized to the volume's intensity range, but
            # the user may have saved a value outside that range. Widen the
            # range as needed so setValue doesn't silently clamp.
            self._widen_to_fit(self.low_slider, s.low)
            self._widen_to_fit(self.high_slider, s.high)
            self.low_slider.setValue(s.low)
            self.high_slider.setValue(s.high)
            self.seed_z.setValue(s.seed_z)
            self.seed_y.setValue(s.seed_y)
            self.seed_x.setValue(s.seed_x)
            self.tolerance_slider.setValue(s.tolerance)
            self.largest_component_checkbox.setChecked(s.keep_largest_component)
            self.smooth_checkbox.setChecked(s.smooth)
            self.live_preview_checkbox.setChecked(s.live_preview)
        finally:
            self._suppress_live = False
        # Compute once with the loaded settings so the segmentation appears.
        self._apply_now()

    @staticmethod
    def _widen_to_fit(slider: LabeledSlider, target: int) -> None:
        lo = min(slider.slider.minimum(), target)
        hi = max(slider.slider.maximum(), target)
        slider.setRange(lo, hi)

    def _refresh_slider_range(self) -> None:
        volume = self._document.volume
        if volume is None:
            return
        lo, hi = volume.intensity_range()
        pad = max(int((hi - lo) * 0.05), 1)
        lo_i, hi_i = int(lo - pad), int(hi + pad)
        sz, sy, sx = volume.shape
        # Suppress live preview while reconfiguring; defaults shouldn't auto-segment.
        self._suppress_live = True
        try:
            self.low_slider.setRange(lo_i, hi_i)
            self.high_slider.setRange(lo_i, hi_i)
            mid = int(lo + (hi - lo) * 0.5)
            self.low_slider.setValue(mid)
            self.high_slider.setValue(int(hi))
            # Region-grow seed sliders track volume shape so the slider
            # range never lets you pick a seed outside the data.
            self.seed_z.setRange(0, max(sz - 1, 0))
            self.seed_y.setRange(0, max(sy - 1, 0))
            self.seed_x.setRange(0, max(sx - 1, 0))
            self.seed_z.setValue(sz // 2)
            self.seed_y.setValue(sy // 2)
            self.seed_x.setValue(sx // 2)
        finally:
            self._suppress_live = False


class _MedSAMWorker(QThread):
    """Runs MedSAM segmentation off the UI thread with progress + cancel."""

    progress = pyqtSignal(str, float)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, volume: Volume, region: Region) -> None:
        super().__init__()
        self._volume = volume
        self._region = region

    def run(self) -> None:
        try:
            from dicom_viewer.core.segmentation.medsam import MedSAMSegmenter

            segmenter = MedSAMSegmenter()
            seg = segmenter.segment_volume_z(
                self._volume,
                self._region,
                progress=lambda s, f: self.progress.emit(s, f),
                should_cancel=self.isInterruptionRequested,
            )
            self.finished_ok.emit(seg)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
