"""Ambient, directional, and point light setup for Phong shading."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from faceforge.core.math_utils import Vec3, Mat4, normalize, vec3
from faceforge.rendering.shader_program import ShaderProgram


@dataclass
class PointLight:
    """A positional light source with distance attenuation.

    Attributes
    ----------
    position : np.ndarray
        World-space [x, y, z] position.
    color : tuple
        RGB colour (0..1).
    intensity : float
        Brightness multiplier.
    range : float
        Effective range for attenuation denominator.
    enabled : bool
        Whether the light contributes to the scene.
    """
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    color: tuple[float, float, float] = (1.0, 0.95, 0.85)  # warm white
    intensity: float = 1.5
    range: float = 400.0
    enabled: bool = False


class LightSetup:
    """Holds ambient, directional, and optional point light for the scene.

    Attributes
    ----------
    ambient_color : Vec3
        RGB ambient light contribution (default warm grey).
    light_dir : Vec3
        Direction *toward* the light in view/world space (normalised).
    light_color : Vec3
        RGB intensity of the directional light.
    point_light : Optional[PointLight]
        Optional point light for scene mode.
    """

    def __init__(self) -> None:
        self.ambient_color: Vec3 = vec3(0.4, 0.4, 0.45)
        self.light_dir: Vec3 = normalize(vec3(1.0, 1.0, 1.0))
        self.light_color: Vec3 = vec3(0.8, 0.8, 0.75)
        self.point_light: Optional[PointLight] = None

    def apply(self, shader: ShaderProgram) -> None:
        """Upload light uniforms to the given shader program.

        Must be called after ``shader.use()``.
        """
        shader.set_uniform_vec3("uAmbientColor", self.ambient_color)
        shader.set_uniform_vec3("uLightDir", self.light_dir)
        shader.set_uniform_vec3("uLightColor", self.light_color)

    def upload_point_light(self, shader: ShaderProgram, view: Mat4) -> None:
        """Upload point light uniforms.  *view* transforms world → view space.

        Must be called after ``shader.use()``.
        """
        pl = self.point_light
        if pl is None or not pl.enabled:
            shader.set_uniform_int("uHasPointLight", 0)
            return

        shader.set_uniform_int("uHasPointLight", 1)

        # Transform world-space position into view space
        pos_w = np.array([pl.position[0], pl.position[1], pl.position[2], 1.0],
                         dtype=np.float64)
        pos_v = view @ pos_w
        shader.set_uniform_vec3("uPointLightPos", pos_v[:3])
        shader.set_uniform_vec3("uPointLightColor", pl.color)
        shader.set_uniform_float("uPointLightIntensity", pl.intensity)
        shader.set_uniform_float("uPointLightRange", pl.range)
