"""Study — a single DICOM series ready to view."""
from __future__ import annotations

from dataclasses import dataclass

from dicom_viewer.core.volume import Volume


@dataclass(frozen=True)
class Study:
    volume: Volume
    patient_id: str
    patient_name: str
    study_uid: str
    series_uid: str
    series_description: str
    orientation_cosines: tuple[float, float, float, float, float, float]

    @property
    def modality(self) -> str:
        return self.volume.modality

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        return self.volume.spacing_mm

    @property
    def display_name(self) -> str:
        patient = self.patient_id or "<anonymized>"
        return f"{patient} / {self.series_description or '<no description>'}"
