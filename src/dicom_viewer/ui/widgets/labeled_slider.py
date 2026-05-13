"""LabeledSlider — a horizontal slider with a value-readout label.

Used wherever a numeric input was previously a QSpinBox or QDoubleSpinBox.
The base class handles integer values; LabeledFloatSlider exposes a float
view backed by an integer slider with a configurable step.
"""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class LabeledSlider(QWidget):
    """Horizontal slider + readout label."""

    valueChanged = pyqtSignal(int)

    def __init__(
        self,
        lo: int,
        hi: int,
        initial: int,
        *,
        format_value: Callable[[int], str] = str,
        suffix: str = "",
    ) -> None:
        super().__init__()
        self._format_value = format_value
        self._suffix = suffix
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(initial)
        self.value_label = QLabel(self._render(initial))
        self.value_label.setMinimumWidth(60)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._on_changed)

    def _render(self, v: int) -> str:
        return f"{self._format_value(v)}{self._suffix}"

    def _on_changed(self, v: int) -> None:
        self.value_label.setText(self._render(v))
        self.valueChanged.emit(v)

    def value(self) -> int:
        return int(self.slider.value())

    def setValue(self, v: int) -> None:  # noqa: N802 — Qt-style API
        self.slider.setValue(int(v))

    def setRange(self, lo: int, hi: int) -> None:  # noqa: N802 — Qt-style API
        self.slider.setRange(int(lo), int(hi))


class LabeledFloatSlider(LabeledSlider):
    """Slider for a float range, backed by an integer slider with a fixed step.

    Example: LabeledFloatSlider(0.0, 0.95, 0.5, step=0.05) -> slider [0..19].
    Use float_value() / setFloatValue() instead of value()/setValue() to work
    in the user-facing range.
    """

    floatValueChanged = pyqtSignal(float)

    def __init__(
        self,
        lo: float,
        hi: float,
        initial: float,
        *,
        step: float = 0.01,
        decimals: int = 2,
    ) -> None:
        if step <= 0:
            raise ValueError("step must be > 0")
        self._step = step
        self._decimals = decimals
        n_lo = int(round(lo / step))
        n_hi = int(round(hi / step))
        n_init = int(round(initial / step))
        super().__init__(
            n_lo,
            n_hi,
            n_init,
            format_value=lambda v: f"{v * step:.{decimals}f}",
        )
        self.valueChanged.connect(lambda _v: self.floatValueChanged.emit(self.float_value()))

    def float_value(self) -> float:
        return float(self.value() * self._step)

    def setFloatValue(self, v: float) -> None:  # noqa: N802 — Qt-style API
        self.setValue(int(round(v / self._step)))

    def setFloatRange(self, lo: float, hi: float) -> None:  # noqa: N802 — Qt-style API
        self.setRange(int(round(lo / self._step)), int(round(hi / self._step)))
