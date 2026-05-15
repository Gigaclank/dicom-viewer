"""Single-source-of-truth for 'what is the app currently doing'.

Panels and workers report tasks to this model via ``begin(id, label)`` /
``update(id, label)`` / ``end(id)``. The MainWindow's status bar subscribes
and renders the active set. Idle when no tasks are registered.

Each task has a string id (stable across update calls) and a human-readable
label (may change as a worker progresses through stages). Reporting from a
QThread is safe IF the worker connects its progress signal to a slot that
calls into this model — Qt queues those calls onto the main thread.
"""
from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class StatusModel(QObject):
    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        # Ordered so the status bar shows the first-started task first when
        # multiple are in flight (e.g. brush + STL preview).
        self._tasks: dict[str, str] = {}

    def begin(self, task_id: str, label: str) -> None:
        self._tasks[task_id] = label
        self.changed.emit()

    def update(self, task_id: str, label: str) -> None:
        # Only update if the task is still active — late progress signals
        # arriving after end() shouldn't resurrect a finished task.
        if task_id in self._tasks:
            self._tasks[task_id] = label
            self.changed.emit()

    def end(self, task_id: str) -> None:
        if task_id in self._tasks:
            del self._tasks[task_id]
            self.changed.emit()

    @property
    def is_idle(self) -> bool:
        return not self._tasks

    @property
    def active_labels(self) -> list[str]:
        return list(self._tasks.values())

    def render(self) -> str:
        """One-line status text suitable for the bottom bar."""
        labels = self.active_labels
        if not labels:
            return "Idle"
        if len(labels) == 1:
            return f"Currently doing: {labels[0]}"
        return f"Currently doing: {len(labels)} tasks — {labels[0]}"
