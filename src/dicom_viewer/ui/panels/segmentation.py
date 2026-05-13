"""Segmentation panel — threshold + region-grow methods with chained refinements."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.document import Document
from dicom_viewer.core.segmentation.morphology import keep_largest_component, smooth_mask
from dicom_viewer.core.segmentation.region_grow import region_grow
from dicom_viewer.core.segmentation.threshold import threshold


class SegmentationPanel(QWidget):
    def __init__(self, document: Document) -> None:
        super().__init__()
        self._document = document

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Threshold", "Region grow"])

        self.low_spin = QSpinBox()
        self.low_spin.setRange(-2000, 10000)
        self.low_spin.setValue(300)

        self.high_spin = QSpinBox()
        self.high_spin.setRange(-2000, 10000)
        self.high_spin.setValue(3000)

        self.seed_z = QSpinBox(); self.seed_z.setRange(0, 100000)
        self.seed_y = QSpinBox(); self.seed_y.setRange(0, 100000)
        self.seed_x = QSpinBox(); self.seed_x.setRange(0, 100000)
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(0, 10000)
        self.tolerance_spin.setValue(100)

        self.largest_component_checkbox = QCheckBox("Keep largest connected component")
        self.largest_component_checkbox.setChecked(True)
        self.smooth_checkbox = QCheckBox("Smooth mask (close + open)")
        self.smooth_checkbox.setChecked(False)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._on_apply)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(lambda: document.set_segmentation(None))

        self._status = QLabel("No segmentation")

        form = QFormLayout()
        form.addRow("Method", self.method_combo)
        form.addRow("Low", self.low_spin)
        form.addRow("High", self.high_spin)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("seed z/y/x:"))
        seed_row.addWidget(self.seed_z)
        seed_row.addWidget(self.seed_y)
        seed_row.addWidget(self.seed_x)
        form.addRow(seed_row)
        form.addRow("Tolerance", self.tolerance_spin)
        form.addRow(self.largest_component_checkbox)
        form.addRow(self.smooth_checkbox)

        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.clear_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self._status)

        document.subscribe(self._on_doc_event)

    def _on_apply(self) -> None:
        volume = self._document.volume
        if volume is None:
            return
        method = self.method_combo.currentText()
        if method == "Threshold":
            seg = threshold(volume, self.low_spin.value(), self.high_spin.value())
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

    def _on_doc_event(self, kind: str) -> None:
        if kind == "segmentation":
            seg = self._document.segmentation
            if seg is None:
                self._status.setText("No segmentation")
            else:
                self._status.setText(f"{seg.method} — {seg.voxel_count} voxels")
