import os

import numpy as np
import pytest

from dicom_viewer.core.region import Region
from dicom_viewer.core.volume import Volume
from dicom_viewer.rendering.volume_renderer import VolumeRenderer


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    monkeypatch.setenv("DICOM_VIEWER_OFFSCREEN", "1")


def _vol() -> Volume:
    arr = np.zeros((8, 8, 8), dtype=np.int16)
    arr[2:6, 2:6, 2:6] = 500
    return Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="CT")


def test_volume_renderer_smoke():
    r = VolumeRenderer()
    r.set_volume(_vol())
    r.render()


def test_volume_renderer_region_box():
    r = VolumeRenderer()
    r.set_volume(_vol())
    r.set_region(Region(z=(1, 5), y=(1, 5), x=(1, 5)))
    r.render()


def test_volume_renderer_handles_no_volume():
    r = VolumeRenderer()
    r.render()  # must not raise


def test_volume_renderer_reset_view_restores_home_camera():
    """Regression: reset_view must actually undo a user rotation.

    Plain vtkRenderer.ResetCamera() preserves the existing view direction and
    view-up vector, so it only re-distances the camera. We need a true reset
    that snaps position + focal point + view-up back to the state captured
    when the volume was first loaded.
    """
    r = VolumeRenderer()
    r.set_volume(_vol())
    cam = r._renderer.GetActiveCamera()
    home_position = cam.GetPosition()
    home_view_up = cam.GetViewUp()

    # Move the camera around — simulating user interaction.
    cam.SetPosition(home_position[0] + 100, home_position[1] + 50, home_position[2] - 30)
    cam.SetViewUp(1.0, 0.0, 0.0)  # nonsense up-vector
    assert cam.GetPosition() != home_position
    assert cam.GetViewUp() != home_view_up

    r.reset_view()
    assert cam.GetPosition() == home_position
    assert cam.GetViewUp() == home_view_up


def test_volume_renderer_reset_view_without_volume_does_not_crash():
    VolumeRenderer().reset_view()  # no scene yet — must not raise


def test_set_volume_skips_3d_render_for_2d_image():
    """Mammograms and X-rays have z=1 and would blow VTK's 3D-texture limit;
    the volume renderer must skip adding the volume actor and leave the
    scene empty."""
    arr = np.zeros((1, 8, 8), dtype=np.int16)
    vol = Volume(array=arr, spacing_mm=(1.0, 1.0, 1.0), modality="MG")
    r = VolumeRenderer()
    r.set_volume(vol)
    assert r._volume_actor is None


def test_loading_a_new_volume_uses_default_orientation_not_inherited_rotation():
    """Each new volume must start from the same anatomical-anterior pose; the
    rotation the user left on the previous volume must not bleed into the
    home camera captured for the next volume."""
    r = VolumeRenderer()
    r.set_volume(_vol())

    cam = r._renderer.GetActiveCamera()
    # Simulate the user rotating the 3D camera arbitrarily.
    cam.SetViewUp(1.0, 1.0, 0.0)
    cam.Azimuth(45)
    rotated_view_up = cam.GetViewUp()
    assert rotated_view_up != (0.0, 0.0, 1.0)

    # Loading another volume must snap the camera back to (Z-up, anterior).
    r.set_volume(_vol())
    assert r._renderer.GetActiveCamera().GetViewUp() == (0.0, 0.0, 1.0)
    # And the captured home reflects that, not the rotated state.
    assert r._home_view_up == (0.0, 0.0, 1.0)


def test_camera_focal_point_is_volume_center_so_rotation_pivots_correctly():
    """Trackball rotation orbits around the camera's focal point. If the focal
    point sits at world origin (a corner of the volume), rotation feels like
    swinging the scene through space instead of turning it in place. After
    ResetCamera, the focal point should land at the rendered volume's
    bounding-box center — i.e. inside the volume, not at the corner."""
    arr = np.zeros((10, 20, 30), dtype=np.int16)
    arr[2:8, 2:18, 2:28] = 500
    vol = Volume(array=arr, spacing_mm=(2.0, 1.5, 1.0), modality="CT")
    r = VolumeRenderer()
    r.set_volume(vol)
    cam = r._renderer.GetActiveCamera()
    fp = cam.GetFocalPoint()
    # VTK image points sit at voxel centers from 0 to (n-1)*spacing, so the
    # bounding-box center is ((x-1)*sx/2, (y-1)*sy/2, (z-1)*sz/2).
    expected = ((30 - 1) * 1.0 / 2, (20 - 1) * 1.5 / 2, (10 - 1) * 2.0 / 2)
    assert fp == pytest.approx(expected, abs=1e-6)
    # And critically: NOT at the origin / corner of the volume.
    assert fp != (0.0, 0.0, 0.0)


def test_overlay_mask_hides_volume_and_shows_surface():
    r = VolumeRenderer()
    r.set_volume(_vol())
    assert r._volume_actor is not None
    assert r._volume_actor.GetVisibility() == 1

    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[3:5, 3:5, 3:5] = True
    r.set_overlay_mask(mask)
    # Volume render hidden, surface actor present.
    assert r._volume_actor.GetVisibility() == 0
    assert r._overlay_actor is not None

    # Clearing the mask restores the volume render.
    r.set_overlay_mask(None)
    assert r._volume_actor.GetVisibility() == 1
    assert r._overlay_actor is None


def test_set_windowing_rebuilds_transfer_function_with_new_iso():
    """Switching windowing presets must rebuild the 3D opacity transfer
    function so the iso threshold tracks the new window center. Without
    this the 3D pane stays frozen at the modality-default iso (Bone for
    CT), and Lung/Soft-tissue presets do nothing."""
    r = VolumeRenderer()
    r.set_volume(_vol())
    # Capture the opacity at intensity 0 BEFORE switching to a window
    # whose center is at 0 — voxels at 0 should become opaque under the
    # new preset but be invisible under the bone-ish default.
    opacity_before = r._volume_property.GetScalarOpacity().GetValue(0.0)
    r.set_windowing(center=0.0, width=400.0)
    opacity_after = r._volume_property.GetScalarOpacity().GetValue(0.0)
    assert opacity_before != pytest.approx(opacity_after)


def test_set_windowing_persists_across_volume_load():
    """The user's preset choice must survive opening a new volume — without
    this the 3D pane would snap back to the modality default each time the
    series picker switches study."""
    r = VolumeRenderer()
    r.set_volume(_vol())
    r.set_windowing(center=-500.0, width=1500.0)  # lung-ish
    saved = r._windowing
    r.set_volume(_vol())
    assert r._windowing is saved
    # And the opacity TF has been rebuilt around the new iso, not the bone default.
    iso = float(saved.center)
    # At the iso the opacity must be nonzero (it's the lower edge of the ramp).
    assert r._volume_property.GetScalarOpacity().GetValue(iso) > 0.0


def test_crosshair_planes_appear_after_set_volume():
    """Three translucent indicator planes (AX/COR/SAG) are added to the
    scene as part of set_volume. They're invisible-by-default-no — they
    are always present once a 3D-capable volume is loaded."""
    r = VolumeRenderer()
    r.set_volume(_vol())
    assert r._crosshair_actors is not None
    assert len(r._crosshair_actors) == 3
    # Each plane lives in the renderer's actor list.
    actor_list = r._renderer.GetActors()
    actor_list.InitTraversal()
    in_scene = []
    while True:
        a = actor_list.GetNextActor()
        if a is None:
            break
        in_scene.append(a)
    for ch_actor in r._crosshair_actors:
        assert ch_actor in in_scene


def test_crosshair_planes_translucent_and_lighting_off():
    """The planes are indicators, not lit geometry — they must stay legible
    over the volume render regardless of camera angle."""
    r = VolumeRenderer()
    r.set_volume(_vol())
    for a in r._crosshair_actors:
        prop = a.GetProperty()
        # ~18% opacity per the design comment.
        assert 0.05 < prop.GetOpacity() < 0.5
        # Lighting must be disabled (flat color regardless of view angle).
        assert prop.GetLighting() == 0


def test_set_crosshair_position_moves_plane_origins():
    """Changing the crosshair voxel must move the three plane sources to
    world coords matching (z+0.5)*sz, (y+0.5)*sy, (x+0.5)*sx for the AX,
    COR, and SAG planes respectively."""
    arr = np.zeros((10, 20, 30), dtype=np.int16)
    arr[2:8, 2:18, 2:28] = 500
    vol = Volume(array=arr, spacing_mm=(2.0, 1.5, 1.0), modality="CT")
    r = VolumeRenderer()
    r.set_volume(vol)

    r.set_crosshair_position(z=4, y=8, x=15)
    ax_src, co_src, sa_src = r._crosshair_sources
    # Axial plane normal to Z, positioned at world Z = (4+0.5)*2.0 = 9.0.
    assert ax_src.GetOrigin()[2] == pytest.approx(9.0, abs=1e-6)
    # Coronal plane normal to Y, world Y = (8+0.5)*1.5 = 12.75.
    assert co_src.GetOrigin()[1] == pytest.approx(12.75, abs=1e-6)
    # Sagittal plane normal to X, world X = (15+0.5)*1.0 = 15.5.
    assert sa_src.GetOrigin()[0] == pytest.approx(15.5, abs=1e-6)
    assert r.crosshair_voxel == (4, 8, 15)


def test_set_crosshair_position_clamps_out_of_bounds():
    """Callers may pass out-of-range indices (e.g. negative scrub). Clamp
    to the volume bounds rather than crashing or rendering empty planes."""
    r = VolumeRenderer()
    r.set_volume(_vol())
    r.set_crosshair_position(z=-50, y=999, x=4)
    z, y, x = r.crosshair_voxel
    assert z == 0
    assert y == r._volume.shape[1] - 1
    assert x == 4


def test_loading_a_new_volume_recreates_crosshair_actors():
    """When the user opens a different DICOM, the previous volume's
    crosshair actors must be removed and fresh ones built for the new
    geometry — otherwise they'd be sized for the wrong volume."""
    r = VolumeRenderer()
    r.set_volume(_vol())
    first_actors = r._crosshair_actors
    r.set_volume(_vol())
    second_actors = r._crosshair_actors
    assert first_actors is not None and second_actors is not None
    # New actor instances.
    assert second_actors[0] is not first_actors[0]


def test_overlay_re_renders_when_region_changes():
    r = VolumeRenderer()
    r.set_volume(_vol())
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    r.set_overlay_mask(mask)
    first_actor = r._overlay_actor
    assert first_actor is not None

    # Tightening the region should produce a fresh surface actor (different
    # geometry under the hood).
    r.set_region(Region(z=(2, 4), y=(2, 6), x=(2, 6)))
    assert r._overlay_actor is not None
    assert r._overlay_actor is not first_actor
