"""Apply Material properties to a shader program and configure GL state."""

from OpenGL.GL import (
    GL_BACK,
    GL_BLEND,
    GL_CULL_FACE,
    GL_FILL,
    GL_FRONT_AND_BACK,
    GL_LINE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINT,
    GL_SRC_ALPHA,
    glBlendFunc,
    glCullFace,
    glDisable,
    glEnable,
    glDepthMask,
    glPolygonMode,
)

from faceforge.core.material import Material, RenderMode
from faceforge.rendering.shader_program import ShaderProgram


def apply_material(shader: ShaderProgram, material: Material) -> None:
    """Set shader uniforms and GL state to match *material*.

    Must be called after ``shader.use()`` and before the draw call.
    """
    # --- Uniforms --------------------------------------------------------
    shader.set_uniform_vec3("uColor", material.color)
    shader.set_uniform_float("uOpacity", material.opacity)
    shader.set_uniform_float("uShininess", material.shininess)
    shader.set_uniform_int("uUseVertexColor", 1 if material.vertex_colors_active else 0)

    # --- Transparency / blending -----------------------------------------
    if material.render_mode == RenderMode.OPAQUE:
        # Opaque mode forces full opacity, no blending, depth write on
        shader.set_uniform_float("uOpacity", 1.0)
        glDisable(GL_BLEND)
        glDepthMask(True)
    else:
        is_transparent = material.transparent or material.opacity < 1.0
        if is_transparent:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(False)  # Transparent objects typically don't write depth
        else:
            glDisable(GL_BLEND)
            glDepthMask(True)

    # --- Face culling ----------------------------------------------------
    if material.double_sided:
        glDisable(GL_CULL_FACE)
    else:
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

    # --- Polygon mode for wireframe --------------------------------------
    if material.render_mode == RenderMode.WIREFRAME:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    elif material.render_mode == RenderMode.POINTS:
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


def restore_material_defaults() -> None:
    """Reset GL state changed by :func:`apply_material` to safe defaults."""
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glDisable(GL_BLEND)
    glDepthMask(True)
    glEnable(GL_CULL_FACE)
