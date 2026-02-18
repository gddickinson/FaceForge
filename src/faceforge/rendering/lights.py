"""Ambient and directional light setup for Phong shading."""

import numpy as np

from faceforge.core.math_utils import Vec3, normalize, vec3
from faceforge.rendering.shader_program import ShaderProgram


class LightSetup:
    """Holds ambient and a single directional light for the scene.

    Attributes
    ----------
    ambient_color : Vec3
        RGB ambient light contribution (default warm grey).
    light_dir : Vec3
        Direction *toward* the light in view/world space (normalised).
    light_color : Vec3
        RGB intensity of the directional light.
    """

    def __init__(self) -> None:
        self.ambient_color: Vec3 = vec3(0.4, 0.4, 0.45)
        self.light_dir: Vec3 = normalize(vec3(1.0, 1.0, 1.0))
        self.light_color: Vec3 = vec3(0.8, 0.8, 0.75)

    def apply(self, shader: ShaderProgram) -> None:
        """Upload light uniforms to the given shader program.

        Must be called after ``shader.use()``.
        """
        shader.set_uniform_vec3("uAmbientColor", self.ambient_color)
        shader.set_uniform_vec3("uLightDir", self.light_dir)
        shader.set_uniform_vec3("uLightColor", self.light_color)
