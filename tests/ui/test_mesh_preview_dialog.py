import numpy as np

from dicom_viewer.core.mesh_export import ExportOptions, generate_mesh
from dicom_viewer.core.segmentation.threshold import threshold
from dicom_viewer.core.volume import Volume
from dicom_viewer.ui.widgets.mesh_preview_dialog import MeshPreviewDialog


def _cube_mesh():
    arr = np.zeros((16, 16, 16), dtype=np.int16)
    arr[4:12, 4:12, 4:12] = 1000
    v = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")
    seg = threshold(v, low=500, high=2000)
    return generate_mesh(v, seg, region=v.bbox(), options=ExportOptions(smoothing_iterations=0))


def test_mesh_preview_dialog_shows_triangle_count(qtbot):
    dlg = MeshPreviewDialog()
    qtbot.addWidget(dlg)
    mesh = _cube_mesh()
    dlg.set_mesh(mesh)
    text = dlg._info_label.text()
    assert "triangles" in text
    assert "mm" in text
    assert str(mesh.triangle_count) in text or f"{mesh.triangle_count:,}" in text


def test_mesh_preview_dialog_reset_button_callable(qtbot):
    dlg = MeshPreviewDialog()
    qtbot.addWidget(dlg)
    dlg.set_mesh(_cube_mesh())
    dlg.reset_button.click()  # must not raise
