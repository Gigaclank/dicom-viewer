"""Segmentation panel — threshold + region-grow methods with live preview."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.document import Document
from dicom_viewer.core.segmentation.morphology import keep_largest_component, smooth_mask
from dicom_viewer.core.segmentation.region_grow import region_grow
from dicom_viewer.core.segmentation.threshold import threshold


class _LabeledSlider(QWidget):
    """A horizontal slider with a numeric readout label."""

    valueChanged = pyqtSignal(int)

    def __init__(self, lo: int, hi: int, initial: int) -> None:
        super().__init__()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(initial)
        self.value_label = QLabel(str(initial))
        self.value_label.setMinimumWidth(60)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._on_changed)

    def _on_changed(self, v: int) -> None:
        self.value_label.setText(str(v))
        self.valueChanged.emit(v)

    def value(self) -> int:
        return int(self.slider.value())

    def setValue(self, v: int) -> None:  # noqa: N802 — Qt-style API
        self.slider.setValue(int(v))

    def setRange(self, lo: int, hi: int) -> None:  # noqa: N802 — Qt-style API
        self.slider.setRange(int(lo), int(hi))


class SegmentationPanel(QWidget):
    def __init__(self, document: Document) -> None:
        super().__init__()
        self._document = document
        self._suppress_live = False

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Threshold", "Region grow"])
        self.method_combo.currentTextChanged.connect(self._on_method_changed)

        self.low_slider = _LabeledSlider(-2000, 10000, 300)
        self.high_slider = _LabeledSlider(-2000, 10000, 3000)
        self.low_slider.valueChanged.connect(self._on_threshold_changed)
        self.high_slider.valueChanged.connect(self._on_threshold_changed)

        self.seed_z = QSpinBox(); self.seed_z.setRange(0, 100000)
        self.seed_y = QSpinBox(); self.seed_y.setRange(0, 100000)
        self.seed_x = QSpinBox(); self.seed_x.setRange(0, 100000)
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(0, 10000)
        self.tolerance_spin.setValue(100)

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

        self._status = QLabel("No segmentation")
        self._status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Method", self.method_combo)
        form.addRow("Low", self.low_slider)
        form.addRow("High", self.high_slider)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("seed z/y/x:"))
        seed_row.addWidget(self.seed_z)
        seed_row.addWidget(self.seed_y)
        seed_row.addWidget(self.seed_x)
        form.addRow(seed_row)
        form.addRow("Tolerance", self.tolerance_spin)
        form.addRow(self.largest_component_checkbox)
        form.addRow(self.smooth_checkbox)
        form.addRow(self.live_preview_checkbox)

        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.clear_button)

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
                tolerance=self.tolerance_spin.value(),
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

    def _refresh_slider_range(self) -> None:
        volume = self._document.volume
        if volume is None:
            return
        lo, hi = volume.intensity_range()
        pad = max(int((hi - lo) * 0.05), 1)
        lo_i, hi_i = int(lo - pad), int(hi + pad)
        # Suppress live preview while reconfiguring; defaults shouldn't auto-segment.
        self._suppress_live = True
        try:
            self.low_slider.setRange(lo_i, hi_i)
            self.high_slider.setRange(lo_i, hi_i)
            # Sensible defaults: low = midpoint, high = max — user usually
            # narrows the low edge to isolate bone / a tissue class.
            mid = int(lo + (hi - lo) * 0.5)
            self.low_slider.setValue(mid)
            self.high_slider.setValue(int(hi))
        finally:
            self._suppress_live = False
