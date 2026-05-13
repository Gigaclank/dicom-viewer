"""Window/Level panel — sliders + modality-aware presets."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from dicom_viewer.core.document import Document, WindowingState

_CT_PRESETS: dict[str, tuple[int, int]] = {
    "Bone": (400, 1500),
    "Soft Tissue": (40, 400),
    "Lung": (-600, 1500),
    "Brain": (40, 80),
}


class WindowingPanel(QWidget):
    def __init__(self, document: Document) -> None:
        super().__init__()
        self._document = document
        self._building = False

        self.preset_combo = QComboBox()
        self._refresh_presets()
        self.preset_combo.activated.connect(
            lambda _i: self.apply_preset(self.preset_combo.currentText())
        )

        self.center_slider = QSlider(Qt.Orientation.Horizontal)
        self.center_slider.setRange(-1024, 4096)
        self.center_slider.valueChanged.connect(self._on_slider_changed)

        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider.setRange(1, 8192)
        self.width_slider.valueChanged.connect(self._on_slider_changed)

        self._readout = QLabel()

        form = QFormLayout()
        form.addRow("Preset", self.preset_combo)
        form.addRow("Center", self.center_slider)
        form.addRow("Width", self.width_slider)
        form.addRow(self._readout)
        layout = QVBoxLayout(self)
        layout.addLayout(form)

        document.subscribe(self._on_doc_event)
        self._sync_from_document()

    def apply_preset(self, name: str) -> None:
        if name not in _CT_PRESETS:
            return
        c, w = _CT_PRESETS[name]
        self._document.set_windowing(WindowingState(center=float(c), width=float(w)))

    def _on_slider_changed(self, _value: int) -> None:
        if self._building:
            return
        self._document.set_windowing(
            WindowingState(
                center=float(self.center_slider.value()),
                width=float(self.width_slider.value()),
            )
        )

    def _on_doc_event(self, kind: str) -> None:
        if kind in ("study", "windowing"):
            self._sync_from_document()
        if kind == "study":
            self._refresh_presets()

    def _refresh_presets(self) -> None:
        self.preset_combo.clear()
        if self._document.volume and self._document.volume.modality == "CT":
            for name in _CT_PRESETS:
                self.preset_combo.addItem(name)
        else:
            self.preset_combo.addItem("Auto (MRI)")

    def _sync_from_document(self) -> None:
        self._building = True
        try:
            w = self._document.windowing
            self.center_slider.setValue(int(round(w.center)))
            self.width_slider.setValue(int(round(w.width)))
            self._readout.setText(f"C={w.center:.0f} W={w.width:.0f}")
        finally:
            self._building = False
