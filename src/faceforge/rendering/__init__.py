"""Rendering subsystem -- OpenGL 3.3 core profile with PySide6 integration."""

from faceforge.rendering.camera import Camera
from faceforge.rendering.gl_material import apply_material, restore_material_defaults
from faceforge.rendering.gl_mesh import GLMesh
from faceforge.rendering.gl_widget import GLViewport
from faceforge.rendering.lights import LightSetup
from faceforge.rendering.orbit_controls import OrbitControls
from faceforge.rendering.renderer import GLRenderer
from faceforge.rendering.shader_program import ShaderProgram

__all__ = [
    "Camera",
    "GLMesh",
    "GLRenderer",
    "GLViewport",
    "LightSetup",
    "OrbitControls",
    "ShaderProgram",
    "apply_material",
    "restore_material_defaults",
]
