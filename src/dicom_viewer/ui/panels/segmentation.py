"""Segmentation panel — threshold + region-grow methods with live preview."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import QCoreApplication, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.document import Document
from dicom_viewer.core.region import Region
from dicom_viewer.core.segmentation.base import Segmentation
from dicom_viewer.core.segmentation.click_seed import (
    apply_brush_stroke,
    box_from_seed,
    confidence_grow_from_seed,
    grow_from_seed,
    paint_disc_2d,
    sphere_from_seed,
    threshold_from_seed,
)
from dicom_viewer.core.segmentation.morphology import keep_largest_component, smooth_mask
from dicom_viewer.core.segmentation.region_grow import region_grow
from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.volume import Orientation, Volume
from dicom_viewer.io.nifti import load_segmentation_from_nifti
from dicom_viewer.io.project import SegmentationSettings
from dicom_viewer.ui.status_model import StatusModel
from dicom_viewer.ui.widgets.labeled_slider import LabeledSlider


class SegmentationPanel(QWidget):
    # 300 ms after the last slider/checkbox change — long enough that dragging
    # doesn't spawn workers per pixel, short enough to feel responsive.
    LIVE_DEBOUNCE_MS = 300

    # Mirrors SliceView's brush modes. Emitted whenever the user picks a
    # different mode; MainWindow routes this to all three slice views.
    brush_mode_changed = pyqtSignal(str)

    def __init__(self, document: Document, status: StatusModel | None = None) -> None:
        super().__init__()
        self._document = document
        # Optional global "Currently doing" surface — Nothing fails if it's
        # absent (tests construct the panel directly without a MainWindow).
        self._status_model = status
        self._suppress_live = False
        self._seg_worker: Optional["_SegmentationWorker"] = None
        self._brush_worker: Optional["_BrushWorker"] = None
        # All brush workers we've started that haven't reported `finished`
        # yet. SITK ConnectedThreshold can't be interrupted mid-call, so a
        # cancelled worker keeps running for the full SITK duration; if we
        # only held a reference to the latest worker, Python would GC the
        # older ones while their QThreads were still running, which aborts
        # the process. This list keeps them alive until `finished` fires.
        self._brush_workers_in_flight: list["_BrushWorker"] = []
        self._brush_mode: str = "off"

        # Debounce live-preview triggers — slider value-changed fires per-tick,
        # but we only want to start a worker when the user pauses.
        self._live_debounce = QTimer(self)
        self._live_debounce.setSingleShot(True)
        self._live_debounce.setInterval(self.LIVE_DEBOUNCE_MS)
        self._live_debounce.timeout.connect(self._apply_now)

        # --- mask library row ---
        self.mask_combo = QComboBox()
        self.mask_combo.setToolTip(
            "Switch between named masks saved in this project. The active "
            "segmentation comes from the selected entry, or 'unsaved' if "
            "you've created a new mask but not named it yet."
        )
        self.mask_combo.currentIndexChanged.connect(self._on_mask_combo_changed)
        self.save_mask_button = QPushButton("Save as…")
        self.save_mask_button.setToolTip(
            "Save the current segmentation under a name so you can switch "
            "back to it. Saved masks travel with the project file."
        )
        self.save_mask_button.clicked.connect(self._on_save_mask_clicked)
        self.delete_mask_button = QPushButton("Delete")
        self.delete_mask_button.setToolTip("Remove the selected named mask from the project.")
        self.delete_mask_button.clicked.connect(self._on_delete_mask_clicked)
        self.delete_mask_button.setEnabled(False)

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
            "the prompt. First click downloads ~360MB of model weights."
        )
        self.medsam_button.clicked.connect(self._on_medsam_clicked)

        # --- Tumor brush ---
        # Click-driven interactive segmentation: click inside the tumor to add
        # a flood-filled region; click on a leak to subtract it. Tolerance
        # below controls the intensity window around the seed voxel.
        self.brush_off_radio = QRadioButton("Off")
        self.brush_add_radio = QRadioButton("Add")
        self.brush_remove_radio = QRadioButton("Remove")
        self.brush_off_radio.setChecked(True)
        self._brush_group = QButtonGroup(self)
        self._brush_group.addButton(self.brush_off_radio)
        self._brush_group.addButton(self.brush_add_radio)
        self._brush_group.addButton(self.brush_remove_radio)
        self.brush_off_radio.toggled.connect(
            lambda checked: checked and self._set_brush_mode("off")
        )
        self.brush_add_radio.toggled.connect(
            lambda checked: checked and self._set_brush_mode("add")
        )
        self.brush_remove_radio.toggled.connect(
            lambda checked: checked and self._set_brush_mode("remove")
        )
        for btn, tip in (
            (self.brush_off_radio, "Normal slice interaction (drag = window/level, wheel = scrub)."),
            (self.brush_add_radio, "Click inside the tumor on any slice to grow a region and add it to the current mask."),
            (self.brush_remove_radio, "Click on tissue that leaked into the mask to subtract it."),
        ):
            btn.setToolTip(tip)

        # --- brush kind selector ---
        # Different shapes/algorithms for the same click+mode workflow:
        # Region grow is the click-anywhere flood-fill; Threshold catches
        # internal voids; Sphere/Box drop a 3D primitive; 2D paint streams
        # disc strokes as the user drags; Smart grow uses local image
        # statistics for diffuse boundaries.
        self.brush_kind_combo = QComboBox()
        self.brush_kind_combo.addItem("Region grow", userData="region_grow")
        self.brush_kind_combo.addItem("Threshold (no connectivity)", userData="threshold")
        self.brush_kind_combo.addItem("Sphere (drops a 3D ball)", userData="sphere")
        self.brush_kind_combo.addItem("Box (drops a 3D box)", userData="box")
        self.brush_kind_combo.addItem("2D paint (drag to draw)", userData="paint_2d")
        self.brush_kind_combo.addItem("Smart grow (statistical)", userData="confidence")
        self.brush_kind_combo.setToolTip(
            "Which math the brush click runs. Region grow is the default; "
            "switch to others for tumors with internal voids, diffuse "
            "boundaries, or when you want a primitive shape instead."
        )

        # Spatial brush size — used by sphere/box (mm radius / half-extent)
        # and 2D paint (pixel radius). Reinterpreted per kind; the label
        # in the panel updates so the unit stays clear.
        self.brush_radius_slider = LabeledSlider(1, 100, 8)
        self.brush_radius_slider.setToolTip(
            "Spatial size of the brush. Units depend on the brush kind: "
            "mm for sphere/box (world-space), pixels for 2D paint."
        )

        self._status = QLabel("No segmentation")
        self._status.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumHeight(14)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)

        form = QFormLayout()
        # Mask library row at the top — most-used control when juggling
        # multiple targets (bone, tumor, vessels, ...).
        mask_row = QHBoxLayout()
        mask_row.addWidget(self.mask_combo, stretch=1)
        mask_row.addWidget(self.save_mask_button)
        mask_row.addWidget(self.delete_mask_button)
        form.addRow("Saved mask", mask_row)
        # Tumor brush: primary tumor-segmentation workflow, kept above the
        # threshold/region-grow controls because click-drive is the fast
        # path for soft-tissue lesions where threshold can't find an edge.
        brush_row = QHBoxLayout()
        brush_row.addWidget(self.brush_off_radio)
        brush_row.addWidget(self.brush_add_radio)
        brush_row.addWidget(self.brush_remove_radio)
        brush_row.addStretch(1)
        form.addRow("Tumor brush", brush_row)
        form.addRow("Brush kind", self.brush_kind_combo)
        form.addRow("Brush size", self.brush_radius_slider)
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
        layout.addWidget(self.progress_bar)
        layout.addWidget(self._status)

        document.subscribe(self._on_doc_event)
        # If a study is already loaded, adapt slider ranges immediately.
        self._refresh_slider_range()
        self._refresh_mask_combo()

    # --- brush mode ---
    def _set_brush_mode(self, mode: str) -> None:
        """Switch the click-seed brush. Emits brush_mode_changed so MainWindow
        can mirror the mode to all three slice views."""
        if mode == self._brush_mode:
            return
        self._brush_mode = mode
        self.brush_mode_changed.emit(mode)

    @property
    def brush_mode(self) -> str:
        return self._brush_mode

    def handle_seed_click(self, seed: tuple[int, int, int], mode: str) -> None:
        """Slot for SliceView.seed_clicked. Branches on the selected brush
        kind: the heavy (SITK-driven) kinds dispatch to a worker thread so
        the UI stays responsive; the cheap kinds (sphere/box) also use the
        worker for code uniformity; 2D paint runs synchronously per-click
        because strokes need single-frame latency."""
        volume = self._document.volume
        if volume is None:
            return
        kind = self.brush_kind_combo.currentData() or "region_grow"
        if kind == "paint_2d":
            self._apply_paint_disc(seed, mode)
            return

        # Cancel any in-flight brush worker. SITK ConnectedThreshold can't be
        # interrupted mid-call, so the old run will complete; its result is
        # rejected by the identity check on finished_ok. Guard against the
        # ref pointing at a C++ object Qt has already deleted (happens when
        # the previous worker finished and retired before this click).
        if self._brush_worker is not None:
            try:
                running = self._brush_worker.isRunning()
            except RuntimeError:
                # C++ object gone — Qt already deleted it after finished().
                running = False
                self._brush_worker = None
            if running:
                self._brush_worker.requestInterruption()

        tolerance = float(self.tolerance_slider.value())
        radius = float(self.brush_radius_slider.value())
        region = self._document.region
        base_mask = (
            self._document.segmentation.mask if self._document.segmentation else None
        )
        prev_method = (
            self._document.segmentation.method if self._document.segmentation else ""
        )

        worker = _BrushWorker(
            volume=volume,
            seed=seed,
            tolerance=tolerance,
            radius=radius,
            region=region,
            mode=mode,
            kind=kind,
            base_mask=base_mask,
            prev_method=prev_method,
        )
        worker.progress.connect(self._on_seg_progress)
        worker.finished_ok.connect(self._on_brush_ready)
        worker.failed.connect(self._on_brush_failed)
        # Keep alive until the QThread actually exits, even if a newer click
        # has replaced the `_brush_worker` reference.
        worker.finished.connect(lambda w=worker: self._retire_brush_worker(w))
        self._brush_workers_in_flight.append(worker)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Brush — %p%")
        self.progress_bar.setVisible(True)
        self._report_status("brush", f"Brush {mode} ({kind})")

        self._brush_worker = worker
        worker.start()

    def handle_seed_drag(
        self,
        seed: tuple[int, int, int],
        mode: str,
        orientation: Orientation,
    ) -> None:
        """Slot for SliceView.seed_dragged. Paint brushes turn drag streams
        into per-position disc stamps; other kinds ignore drags (they only
        act on the initial press)."""
        kind = self.brush_kind_combo.currentData() or "region_grow"
        if kind != "paint_2d":
            return
        self._apply_paint_disc(seed, mode, orientation=orientation)

    def _apply_paint_disc(
        self,
        seed: tuple[int, int, int],
        mode: str,
        orientation: Orientation | None = None,
    ) -> None:
        """Mutate the current segmentation mask with a 2D disc at the seed
        position. Runs synchronously on the UI thread because strokes need
        single-frame latency; the disc paint is a cheap numpy slice."""
        volume = self._document.volume
        if volume is None:
            return
        radius_px = max(1, int(self.brush_radius_slider.value()))
        # Reuse the existing mask in place when possible — paint strokes
        # generate many events per second and allocating a fresh array
        # each time would thrash GC on large volumes.
        seg = self._document.segmentation
        if seg is None:
            mask = np.zeros(volume.shape, dtype=bool)
        else:
            mask = seg.mask
            if not mask.flags.writeable:
                mask = mask.copy()

        z, y, x = seed
        # If the caller didn't tell us the orientation (e.g. press event
        # that arrived via handle_seed_click), infer it from which axis
        # equals the current slice index. Default to AXIAL.
        if orientation is None:
            orientation = Orientation.AXIAL
        if orientation is Orientation.AXIAL:
            slice_idx, center = z, (y, x)
        elif orientation is Orientation.CORONAL:
            slice_idx, center = y, ((volume.shape[0] - 1) - z, x)
        else:  # SAGITTAL
            slice_idx, center = x, ((volume.shape[0] - 1) - z, y)
        paint_disc_2d(
            mask,
            orientation,
            slice_idx,
            center,
            radius_px,
            set_value=(mode == "add"),
        )
        prev_method = seg.method if seg is not None else ""
        method = (
            "paint_2d"
            if not prev_method
            else (
                prev_method if prev_method.endswith("paint_2d") else f"{prev_method}+paint_2d"
            )
        )
        new_seg = Segmentation(
            mask=mask,
            method=method,
            params={"radius_px": radius_px, "mode": mode},
        )
        self._document.set_segmentation(new_seg)

    def _retire_brush_worker(self, worker: "_BrushWorker") -> None:
        """Called when a brush worker's QThread emits `finished`. Drops it
        from the keep-alive list AND clears `self._brush_worker` if it was
        the latest reference, so the next click doesn't try to inspect a
        worker whose C++ object Qt has already collected."""
        try:
            self._brush_workers_in_flight.remove(worker)
        except ValueError:
            pass
        if self._brush_worker is worker:
            self._brush_worker = None

    def _on_brush_ready(
        self,
        worker: "_BrushWorker",
        mask,
        method: str,
        params: dict,
    ) -> None:
        if worker is not self._brush_worker:
            return
        seg = Segmentation(mask=mask, method=method, params=params)
        self._document.set_segmentation(seg)
        self.progress_bar.setVisible(False)
        self._end_status("brush")

    def _on_brush_failed(self, worker: "_BrushWorker", msg: str) -> None:
        if worker is not self._brush_worker:
            return
        self._status.setText(f"Brush failed: {msg}")
        self.progress_bar.setVisible(False)
        self._end_status("brush")

    def run_handle_seed_click_blocking(
        self, seed: tuple[int, int, int], mode: str, timeout_ms: int = 5000
    ) -> None:
        """Dispatch a brush click and wait for its worker. Used by tests
        that need a deterministic post-click state."""
        self.handle_seed_click(seed, mode)
        if self._brush_worker is None:
            return
        self._brush_worker.wait(timeout_ms)
        QCoreApplication.processEvents()

    # --- live preview path ---
    def _on_threshold_changed(self, _value=None) -> None:
        if self._suppress_live:
            return
        # The brush owns the segmentation while it's active; threshold drags
        # would clobber accumulated brush strokes.
        if self._brush_mode != "off":
            return
        if not self.live_preview_checkbox.isChecked():
            return
        if self.method_combo.currentText() != "Threshold":
            return
        # Restart the debounce on every change so we only fire after the user
        # has been still for LIVE_DEBOUNCE_MS — no per-tick worker spawning.
        self._live_debounce.start()

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
                "torch / transformers couldn't be imported from this "
                "environment. If you're running from source, run:\n\n"
                "    pip install torch transformers pillow",
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
        self._report_status("medsam", "MedSAM — loading model…")

        def on_progress(stage: str, fraction: float) -> None:
            dialog.setLabelText(stage)
            dialog.setValue(int(max(0.0, min(1.0, fraction)) * 100))
            self._update_status("medsam", f"MedSAM — {stage}")

        def on_done(seg) -> None:
            state["result"] = seg
            dialog.close()
            self._end_status("medsam")

        def on_failed(msg: str) -> None:
            state["error"] = msg
            dialog.close()
            self._end_status("medsam")

        worker.progress.connect(on_progress)
        worker.finished_ok.connect(on_done)
        worker.failed.connect(on_failed)
        worker.start()
        dialog.exec()
        worker.wait()
        # Safety net for pure-cancel paths where neither finished_ok nor
        # failed fired (worker bailed via interruption without emitting).
        self._end_status("medsam")

        if "error" in state:
            QMessageBox.warning(self, "MedSAM failed", str(state["error"]))
            return
        seg = state.get("result")
        if seg is not None:
            self._document.set_segmentation(seg)

    def _on_apply(self) -> None:
        self._apply_now()

    def _apply_now(self) -> None:
        """Start a worker-thread segmentation run. If a worker is already in
        flight, request its interruption and start a fresh one — the old
        worker's result (if it finishes anyway) is discarded by an identity
        check in _on_seg_ready."""
        volume = self._document.volume
        if volume is None:
            return
        method = self.method_combo.currentText()
        lo, hi = self.low_slider.value(), self.high_slider.value()
        if method == "Threshold" and hi < lo:
            return  # invalid range; user is mid-drag

        # Cancel any in-flight worker. We don't .wait() because the current
        # stage (smooth/largest_component) can't be interrupted mid-call;
        # the worker checks isInterruptionRequested between stages and bails.
        # Either way we replace the reference so the stale result gets
        # rejected on arrival.
        if self._seg_worker is not None and self._seg_worker.isRunning():
            self._seg_worker.requestInterruption()

        worker = _SegmentationWorker(
            volume=volume,
            method=method,
            threshold_lo=lo,
            threshold_hi=hi,
            seed=(self.seed_z.value(), self.seed_y.value(), self.seed_x.value()),
            tolerance=self.tolerance_slider.value(),
            keep_largest=self.largest_component_checkbox.isChecked(),
            smooth=self.smooth_checkbox.isChecked(),
        )
        worker.progress.connect(self._on_seg_progress)
        worker.finished_ok.connect(self._on_seg_ready)
        worker.failed.connect(self._on_seg_failed)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting… %p%")
        self.progress_bar.setVisible(True)
        self._report_status("segmentation", f"Segmenting ({method})")

        self._seg_worker = worker
        worker.start()

    def run_apply_blocking(self, timeout_ms: int = 120_000) -> None:
        """Run _apply_now and wait for completion. Used by apply_settings
        (project load) and tests that need a deterministic state."""
        self._apply_now()
        if self._seg_worker is None:
            return
        self._seg_worker.wait(timeout_ms)
        # Drain queued signals so finished_ok / failed slots run before we
        # return to the caller.
        QCoreApplication.processEvents()

    # --- worker callbacks ---
    def _on_seg_progress(self, stage: str, fraction: float) -> None:
        self.progress_bar.setFormat(f"{stage} — %p%")
        self.progress_bar.setValue(int(max(0.0, min(1.0, fraction)) * 100))
        self._update_status("segmentation", f"Segmenting — {stage}")

    def _on_seg_ready(self, worker: "_SegmentationWorker", seg) -> None:
        # Only honour the latest worker's result — stale results from
        # workers that were interrupted by a newer request are discarded.
        if worker is not self._seg_worker:
            return
        self._document.set_segmentation(seg)
        self.progress_bar.setVisible(False)
        self._end_status("segmentation")

    def _on_seg_failed(self, worker: "_SegmentationWorker", msg: str) -> None:
        if worker is not self._seg_worker:
            return
        self._status.setText(f"Apply failed: {msg}")
        self.progress_bar.setVisible(False)
        self._end_status("segmentation")

    # --- status helpers ---
    def _report_status(self, task_id: str, label: str) -> None:
        if self._status_model is not None:
            self._status_model.begin(task_id, label)

    def _update_status(self, task_id: str, label: str) -> None:
        if self._status_model is not None:
            self._status_model.update(task_id, label)

    def _end_status(self, task_id: str) -> None:
        if self._status_model is not None:
            self._status_model.end(task_id)

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
        if kind in ("mask_library", "study"):
            self._refresh_mask_combo()

    # --- mask library UI ---
    _UNSAVED_LABEL = "(unsaved)"

    def _refresh_mask_combo(self) -> None:
        """Sync the combo to the document's library + active mask."""
        self.mask_combo.blockSignals(True)
        try:
            self.mask_combo.clear()
            self.mask_combo.addItem(self._UNSAVED_LABEL)
            for name in self._document.mask_names:
                self.mask_combo.addItem(name)
            active = self._document.active_mask_name
            if active:
                idx = self.mask_combo.findText(active)
                if idx >= 0:
                    self.mask_combo.setCurrentIndex(idx)
            else:
                self.mask_combo.setCurrentIndex(0)
        finally:
            self.mask_combo.blockSignals(False)
        # Delete is only valid for a saved entry.
        self.delete_mask_button.setEnabled(bool(self._document.active_mask_name))

    def _on_mask_combo_changed(self, _index: int) -> None:
        name = self.mask_combo.currentText()
        if name == self._UNSAVED_LABEL:
            # The user clicked back to 'unsaved' — clear the active library
            # entry but don't wipe the current segmentation, in case they
            # want to keep editing without losing it.
            # (Wiping would surprise users who picked unsaved by accident.)
            self._document._active_mask_name = ""  # noqa: SLF001 — internal protocol
            self._document._emit("mask_library")  # noqa: SLF001
            return
        if name == self._document.active_mask_name:
            return
        self._document.activate_mask(name)

    def _on_save_mask_clicked(self) -> None:
        if self._document.segmentation is None:
            QMessageBox.information(
                self,
                "No segmentation",
                "Apply a segmentation first, then save it under a name.",
            )
            return
        existing = self._document.active_mask_name
        suggested = existing or self._suggested_mask_name()
        name, ok = QInputDialog.getText(
            self, "Save mask", "Mask name:", text=suggested
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        # Sanitize: avoid characters that confuse the companion-file lookup
        # ('/', '\\', '..'). Names show up verbatim in the picker.
        bad = ("/", "\\", "..")
        if any(b in name for b in bad):
            QMessageBox.warning(
                self, "Invalid name",
                "Mask names cannot contain '/', '\\' or '..'.",
            )
            return
        if name in self._document.mask_names and existing != name:
            yes = QMessageBox.question(
                self, "Overwrite mask?",
                f"A mask named '{name}' already exists. Overwrite?",
            )
            if yes != QMessageBox.StandardButton.Yes:
                return
        self._document.save_mask_as(name)

    def _on_delete_mask_clicked(self) -> None:
        name = self._document.active_mask_name
        if not name:
            return
        yes = QMessageBox.question(
            self, "Delete mask?", f"Delete saved mask '{name}'?",
        )
        if yes != QMessageBox.StandardButton.Yes:
            return
        self._document.delete_mask(name)

    def _suggested_mask_name(self) -> str:
        seg = self._document.segmentation
        if seg is None:
            return "mask"
        # Use the method as a starting hint, drop the chained suffixes.
        base = seg.method.split("+", 1)[0]
        # If a mask already has this name, suffix with -2, -3, ...
        existing = set(self._document.mask_names)
        candidate = base
        n = 2
        while candidate in existing:
            candidate = f"{base}-{n}"
            n += 1
        return candidate

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
        """Restore the panel's controls from a saved project. We deliberately
        do NOT re-run the segmentation here — the user explicitly asked that
        loading a project leave the mask unset until they hit Apply. This
        avoids spending CPU cycles on a threshold run the user may not even
        want (they often just want to scrub the slices first). Project file
        load still restores any saved companion masks via the mask library,
        so the on-disk segmentation isn't lost.
        """
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


class _SegmentationCancelled(Exception):
    """Internal signal used by _SegmentationWorker to short-circuit between
    pipeline stages when the panel has requested a fresh run."""


class _SegmentationWorker(QThread):
    """Runs threshold / region-grow plus chained refinements off the UI
    thread, reporting progress per stage.

    `finished_ok` carries `(self, segmentation)` — the panel's slot checks
    that the worker is still the latest one before applying the result.
    """

    progress = pyqtSignal(str, float)
    finished_ok = pyqtSignal(object, object)
    failed = pyqtSignal(object, str)

    def __init__(
        self,
        *,
        volume: Volume,
        method: str,
        threshold_lo: int,
        threshold_hi: int,
        seed: tuple[int, int, int],
        tolerance: int,
        keep_largest: bool,
        smooth: bool,
    ) -> None:
        super().__init__()
        self._volume = volume
        self._method = method
        self._threshold_lo = threshold_lo
        self._threshold_hi = threshold_hi
        self._seed = seed
        self._tolerance = tolerance
        self._keep_largest = keep_largest
        self._smooth = smooth

    def _check(self) -> None:
        if self.isInterruptionRequested():
            raise _SegmentationCancelled()

    def run(self) -> None:
        try:
            self._check()
            self.progress.emit("Computing base mask", 0.05)
            if self._method == "Threshold":
                seg = threshold(self._volume, self._threshold_lo, self._threshold_hi)
            else:
                seg = region_grow(
                    self._volume, seed=self._seed, tolerance=self._tolerance
                )
            self.progress.emit("Computing base mask", 0.40)

            self._check()
            if self._keep_largest:
                self.progress.emit("Keeping largest component", 0.45)
                seg = keep_largest_component(seg)
                self.progress.emit("Keeping largest component", 0.70)

            self._check()
            if self._smooth:
                self.progress.emit("Smoothing", 0.75)
                seg = smooth_mask(seg, iterations=1)
                self.progress.emit("Smoothing", 0.95)

            self._check()
            self.progress.emit("Done", 1.0)
            self.finished_ok.emit(self, seg)
        except _SegmentationCancelled:
            # A newer request took over; silently exit. The panel discards any
            # stale finished_ok via the identity check, but we shouldn't even
            # emit if we caught the cancellation early.
            return
        except Exception as e:  # noqa: BLE001
            self.failed.emit(self, f"{type(e).__name__}: {e}")


class _BrushWorker(QThread):
    """Runs a single click-seed brush evaluation off the UI thread.

    Dispatches by ``kind`` to the appropriate core function. Cancellation:
    SITK callers can't interrupt mid-call, but requestInterruption + the
    panel's identity check on finished_ok mean a stale brush click whose
    result arrives after a newer click is simply discarded.
    """

    progress = pyqtSignal(str, float)
    # (worker, mask, method, params)
    finished_ok = pyqtSignal(object, object, str, dict)
    failed = pyqtSignal(object, str)

    def __init__(
        self,
        *,
        volume: Volume,
        seed: tuple[int, int, int],
        tolerance: float,
        radius: float,
        region: Optional[Region],
        mode: str,
        kind: str,
        base_mask,
        prev_method: str,
    ) -> None:
        super().__init__()
        self._volume = volume
        self._seed = seed
        self._tolerance = tolerance
        self._radius = radius
        self._region = region
        self._mode = mode
        self._kind = kind
        self._base_mask = base_mask
        self._prev_method = prev_method

    def _compute_addition(self):
        """Run the kind-specific math. Returns the 3D bool array to merge."""
        if self._kind == "region_grow":
            return grow_from_seed(self._volume, self._seed, self._tolerance, self._region)
        if self._kind == "threshold":
            return threshold_from_seed(
                self._volume, self._seed, self._tolerance, self._region
            )
        if self._kind == "sphere":
            return sphere_from_seed(
                self._volume, self._seed, self._radius, self._region
            )
        if self._kind == "box":
            return box_from_seed(
                self._volume, self._seed, self._radius, self._region
            )
        if self._kind == "confidence":
            # The radius slider doubles as the confidence multiplier *10
            # so the same widget covers both tolerance and multiplier
            # without adding yet another control. /10 to scale into the
            # 0.1–10.0 range ConfidenceConnected expects.
            multiplier = max(0.1, self._tolerance / 100.0) if self._tolerance else 2.5
            return confidence_grow_from_seed(
                self._volume,
                self._seed,
                multiplier=multiplier,
                region=self._region,
            )
        raise ValueError(f"unknown brush kind: {self._kind!r}")

    def run(self) -> None:
        try:
            self.progress.emit(f"Computing {self._kind}", 0.10)
            addition = self._compute_addition()
            if self.isInterruptionRequested():
                return
            self.progress.emit("Merging", 0.80)
            new_mask = apply_brush_stroke(
                self._base_mask, addition, self._mode, self._volume.shape
            )
            method_tag = f"brush_{self._kind}"
            method = (
                method_tag
                if not self._prev_method
                else (
                    self._prev_method
                    if self._prev_method.endswith(method_tag)
                    else f"{self._prev_method}+{method_tag}"
                )
            )
            params = {
                "seed": tuple(int(v) for v in self._seed),
                "tolerance": float(self._tolerance),
                "radius": float(self._radius),
                "mode": self._mode,
                "kind": self._kind,
            }
            self.progress.emit("Done", 1.0)
            self.finished_ok.emit(self, new_mask, method, params)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(self, f"{type(e).__name__}: {e}")
