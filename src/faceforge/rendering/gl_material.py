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

# ----------------------------------------------------------------------
# Blending policy
# ----------------------------------------------------------------------

# Render modes whose fragment shader computes an alpha of its own, i.e. whose
# entire visual identity lives in the alpha channel:
#
#   xray.frag      alpha = uOpacity * (fresnel * 0.8  + 0.15)
#   hologram.frag  alpha = uOpacity * (glow + interior)
#   blueprint.frag alpha = uOpacity * (wireframe * 0.8 + 0.15 + gridLine)
#   ethereal.frag  alpha = uOpacity * (facing * 0.5 + glow + 0.15)
#   points.frag    alpha = uOpacity * smoothstep(1.0, 0.8, dist)
#
# Each factor is < 1, so with GL_BLEND off the pipeline throws the result away
# and the mode renders as a dark solid.  Until now these modes only blended by
# accident: stl_batch_loader defaulted every structure to opacity 0.7 /
# transparent=True, so the *material* test below happened to be true.  With
# that default corrected to 1.0 (opaque anatomy, so early-Z works) the mode
# must ask for blending on its own behalf -- hence this table.
_MODE_NEEDS_BLENDING: frozenset = frozenset({
    RenderMode.XRAY,
    RenderMode.HOLOGRAM,
    RenderMode.BLUEPRINT,
    RenderMode.ETHEREAL,
    RenderMode.POINTS,
})


def mode_needs_blending(mode: RenderMode) -> bool:
    """True if *mode*'s shader produces a fractional alpha of its own."""
    return mode in _MODE_NEEDS_BLENDING


def needs_blending(material: Material) -> bool:
    """Whether *material* must be drawn with alpha blending enabled.

    The material flag OR the render mode -- either is sufficient.  OPAQUE mode
    overrides both: it exists precisely to force everything solid.

    This is the single source of truth for the blend decision.  The renderer
    uses it both to set the GL state and to decide which meshes go in the
    depth-sorted transparent pass, so the two can never disagree.
    """
    if material.render_mode == RenderMode.OPAQUE:
        return False
    return (material.transparent
            or material.opacity < 1.0
            or material.render_mode in _MODE_NEEDS_BLENDING)


def apply_material(shader: ShaderProgram, material: Material) -> None:
    """Set shader uniforms and GL state to match *material*.

    Must be called after ``shader.use()`` and before the draw call.
    """
    # --- Uniforms --------------------------------------------------------
    is_opaque_mode = material.render_mode == RenderMode.OPAQUE
    shader.set_uniform_vec3("uColor", material.color)
    # Opaque mode forces full opacity.  Uploaded once, not twice as before.
    shader.set_uniform_float(
        "uOpacity", 1.0 if is_opaque_mode else material.opacity)
    shader.set_uniform_float("uShininess", material.shininess)
    shader.set_uniform_int("uUseVertexColor", 1 if material.vertex_colors_active else 0)

    # --- Transparency / blending -----------------------------------------
    if needs_blending(material):
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
