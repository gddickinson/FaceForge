"""The heatmap sets the flags the renderer reads, and honours external levels."""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.muscle_activation import (
    ACTIVATION_BANDS, PALETTES, MuscleActivationSystem, level_band,
)
from faceforge.core.material import Material
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.state import BodyState


def _mesh(name, n=4, colors=None):
    geom = BufferGeometry(positions=np.zeros(n * 3, np.float32), normals=np.zeros(n * 3, np.float32),
                          indices=np.arange(n, dtype=np.uint32), vertex_colors=colors)
    return MeshInstance(name=name, geometry=geom, material=Material())


@pytest.fixture
def system():
    return MuscleActivationSystem(dof_map={"elbow_r_flex": ["Biceps Long R"]})


def test_bands_follow_digiovine():
    assert level_band(0.1) == "low" and level_band(0.3) == "moderate"
    assert level_band(0.5) == "high" and level_band(0.9) == "very high"
    assert [b for _, b in ACTIVATION_BANDS] == ["low", "moderate", "high", "very high"]


def test_update_sets_the_material_flag_and_marks_colours_dirty(system):
    mesh = _mesh("Biceps Long R")
    system.register_muscle(mesh, "Biceps Long R")
    system.set_enabled(True)
    state = BodyState()
    state.elbow_r_flex = 1.0
    system.update(state)
    assert mesh.material.vertex_colors_active is True
    assert mesh.geometry.colors_dirty is True
    assert mesh.geometry.vertex_colors.shape == (4, 3)
    assert np.allclose(mesh.geometry.vertex_colors[0], PALETTES["classic"][255])
    assert system.current_levels == {"Biceps Long R": 1.0}


def test_disabling_restores_the_original_colours(system):
    original = np.full((4, 3), 0.5, np.float32)
    mesh = _mesh("Biceps Long R", colors=original.copy())
    system.register_muscle(mesh, "Biceps Long R")
    system.set_enabled(True)
    system.update(BodyState())
    system.set_enabled(False)
    assert mesh.material.vertex_colors_active is False
    assert np.array_equal(mesh.geometry.vertex_colors, original)
    assert mesh.geometry.colors_dirty is True


def test_external_levels_override_the_pose_estimate_and_zero_the_rest(system):
    biceps, triceps = _mesh("Biceps Long R"), _mesh("Triceps Long R")
    system.register_muscle(biceps, "Biceps Long R")
    system.register_muscle(triceps, "Triceps Long R")
    system.set_enabled(True)
    state = BodyState()
    state.elbow_r_flex = 1.0
    system.set_levels({"Triceps Long R": 0.6})
    system.update(state)
    assert system.current_levels == {"Biceps Long R": 0.0, "Triceps Long R": 0.6}
    assert system.levels_are_external
    system.set_levels(None)
    system.update(state)
    assert system.current_levels["Biceps Long R"] == 1.0


def test_palettes_are_selectable_and_thermal_is_luminance_ordered(system):
    system.set_palette("thermal")
    lut = PALETTES["thermal"]
    lum = lut @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(np.diff(lum) >= -1e-6)
    with pytest.raises(KeyError):
        system.set_palette("rainbow")
