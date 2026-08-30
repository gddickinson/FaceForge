"""Check the numpy transliteration against the real GLSL compiler and rasteriser.

``test_shader_semantics.py`` asserts invariants over ``tools.glsl_cpu``.  Those
assertions only say something about the *shaders* if the transliteration really
is the shader.  This module establishes that: for each of the 16 modes it links
the real ``.frag`` file against a pass-through vertex shader, renders fragments
with exactly known varyings, reads the pixels back, and asserts the numpy
function reproduces them.

Together the two modules give what neither gives alone -- glslang proves the
GLSL compiles, this proves the CPU model matches the driver, and the semantics
tests then constrain behaviour over a swept domain that no single render covers.

Skipping
--------
This needs a real GL context.  On macOS, ``tools.glcontext`` reaches Apple's
software rasteriser without a window server (see that module for the measured
details), so this usually runs even headless.  Where no context exists at all,
the module skips rather than fails: the semantics tests still run, they just
rest on an unverified transliteration, and the skip reason says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.glsl_cpu import MODE_FRAGMENTS, Uniforms, Varyings, vec3

pytest.importorskip("OpenGL", reason="PyOpenGL not installed")

from tools.glcontext import GLContextError, acquire_offscreen_gl  # noqa: E402

# The framebuffer is RGBA8, so a channel carries 1/255 = 0.0039 of range.  The
# GPU computes in float32 and numpy in float64, and the shaders contain pow(),
# sin() and cos() whose last-bit results differ between implementations.  2/255
# absorbs that; a real transliteration error (a wrong constant, a missing term,
# an inverted sign) moves pixels by tens of levels, as the THERMAL 0.6->0.62
# perturbation recorded in gpu_validation.md shows at max_abs=11.
TOLERANCE_8BIT = 2

FBO_SIZE = 16          # small: one flat quad, we sample a handful of pixels

# A pass-through vertex shader.  It emits the four varyings default.vert emits,
# taken straight from vertex attributes, so a test can dictate the exact
# fragment inputs instead of solving for a camera that produces them.
PASSTHROUGH_VERT = """#version 330 core
layout(location = 0) in vec2 aClipPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aViewPos;
layout(location = 3) in vec3 aWorldPos;
layout(location = 4) in vec3 aVertexColor;

out vec3 vNormal;
out vec3 vViewPos;
out vec3 vVertexColor;
out vec3 vWorldPos;

void main() {
    vNormal = aNormal;
    vViewPos = aViewPos;
    vWorldPos = aWorldPos;
    vVertexColor = aVertexColor;
    gl_Position = vec4(aClipPos, 0.0, 1.0);
}
"""

# points.frag declares no vVertexColor input (it pairs with points.vert), so it
# needs a vertex shader that does not emit one -- an unmatched fragment `in` is
# a link error on strict drivers, which is exactly why points.frag is separate.
PASSTHROUGH_VERT_POINTS = """#version 330 core
layout(location = 0) in vec2 aClipPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aViewPos;
layout(location = 3) in vec3 aWorldPos;

out vec3 vNormal;
out vec3 vViewPos;
out vec3 vWorldPos;

void main() {
    vNormal = aNormal;
    vViewPos = aViewPos;
    vWorldPos = aWorldPos;
    gl_Position = vec4(aClipPos, 0.0, 1.0);
}
"""


@pytest.fixture(scope="module")
def gl_context():
    try:
        info = acquire_offscreen_gl("auto")
    except GLContextError as exc:
        pytest.skip(
            "no OpenGL context available, so the numpy transliteration cannot be "
            f"checked against a driver here: {exc}"
        )
    return info


@pytest.fixture(scope="module")
def fbo(gl_context):
    from OpenGL.GL import (
        GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24,
        GL_FRAMEBUFFER, GL_FRAMEBUFFER_COMPLETE, GL_RENDERBUFFER, GL_RGBA,
        GL_RGBA8, GL_TEXTURE_2D, GL_UNSIGNED_BYTE, glBindFramebuffer,
        glBindRenderbuffer, glBindTexture, glCheckFramebufferStatus,
        glFramebufferRenderbuffer, glFramebufferTexture2D, glGenFramebuffers,
        glGenRenderbuffers, glGenTextures, glRenderbufferStorage, glTexImage2D,
    )

    handle = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, handle)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, FBO_SIZE, FBO_SIZE, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, FBO_SIZE, FBO_SIZE)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        pytest.skip(f"framebuffer incomplete (status 0x{int(status):04X})")
    return handle


def _compile_program(vert_src: str, frag_src: str):
    from OpenGL.GL import (
        GL_COMPILE_STATUS, GL_FRAGMENT_SHADER, GL_LINK_STATUS, GL_VERTEX_SHADER,
        glAttachShader, glCompileShader, glCreateProgram, glCreateShader,
        glGetProgramInfoLog, glGetProgramiv, glGetShaderInfoLog, glGetShaderiv,
        glLinkProgram, glShaderSource,
    )

    def one(src: str, kind: int) -> int:
        s = glCreateShader(kind)
        glShaderSource(s, src)
        glCompileShader(s)
        if not glGetShaderiv(s, GL_COMPILE_STATUS):
            log = glGetShaderInfoLog(s)
            raise AssertionError(
                f"shader compile failed: {log.decode() if isinstance(log, bytes) else log}"
            )
        return s

    prog = glCreateProgram()
    glAttachShader(prog, one(vert_src, GL_VERTEX_SHADER))
    glAttachShader(prog, one(frag_src, GL_FRAGMENT_SHADER))
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        log = glGetProgramInfoLog(prog)
        raise AssertionError(
            f"link failed: {log.decode() if isinstance(log, bytes) else log}"
        )
    return prog


def _set_uniforms(prog: int, u: Uniforms) -> None:
    """Upload every uniform any mode might read.

    ``glGetUniformLocation`` returns -1 for a name the linked program does not
    use, and the setters below skip those, so one function serves all 16 modes.
    """
    from OpenGL.GL import (
        glGetUniformLocation, glUniform1f, glUniform1i, glUniform3f,
    )

    def f(name: str, val: float) -> None:
        loc = glGetUniformLocation(prog, name)
        if loc != -1:
            glUniform1f(loc, float(val))

    def i(name: str, val: int) -> None:
        loc = glGetUniformLocation(prog, name)
        if loc != -1:
            glUniform1i(loc, int(val))

    def v3(name: str, val) -> None:
        loc = glGetUniformLocation(prog, name)
        if loc != -1:
            a = np.asarray(val, np.float64)
            glUniform3f(loc, float(a[0]), float(a[1]), float(a[2]))

    v3("uColor", u.color)
    f("uOpacity", u.opacity)
    i("uUseVertexColor", u.use_vertex_color)
    f("uShininess", u.shininess)
    v3("uAmbientColor", u.ambient_color)
    v3("uLightDir", u.light_dir)
    v3("uLightColor", u.light_color)
    i("uHasPointLight", u.has_point_light)
    v3("uPointLightPos", u.point_light_pos)
    v3("uPointLightColor", u.point_light_color)
    f("uPointLightIntensity", u.point_light_intensity)
    f("uPointLightRange", u.point_light_range)
    i("uClipEnabled", 0)


def _render_flat_quad(prog: int, normal, view_pos, world_pos, vertex_color) -> np.ndarray:
    """Draw a full-viewport quad whose four vertices share one set of varyings.

    Because all four vertices carry identical values, interpolation is the
    identity and every fragment receives exactly ``normal`` / ``view_pos`` /
    ``world_pos`` / ``vertex_color``.  Only ``gl_FragCoord`` varies, which is
    what the screen-space modes need in order to be interesting.
    """
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST, GL_FLOAT, GL_RGBA, GL_STATIC_DRAW, GL_TRIANGLE_STRIP,
        GL_UNSIGNED_BYTE, glBindBuffer, glBindVertexArray, glBufferData, glClear,
        glClearColor, glDisable, glDrawArrays, glEnableVertexAttribArray,
        glFinish, glGenBuffers, glGenVertexArrays, glReadPixels, glUseProgram,
        glVertexAttribPointer, glViewport,
    )

    glViewport(0, 0, FBO_SIZE, FBO_SIZE)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_BLEND)          # we want the shader's raw output, not a composite
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(prog)

    n = np.asarray(normal, np.float32)
    vp = np.asarray(view_pos, np.float32)
    wp = np.asarray(world_pos, np.float32)
    vc = np.asarray(vertex_color, np.float32)
    corners = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], np.float32)
    rows = [np.concatenate([c, n, vp, wp, vc]) for c in corners]
    data = np.asarray(rows, np.float32).ravel()

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
    stride = 14 * 4
    for loc, size, offset in ((0, 2, 0), (1, 3, 2), (2, 3, 5), (3, 3, 8), (4, 3, 11)):
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, size, GL_FLOAT, False, stride,
                              ctypes_offset(offset * 4))
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    glFinish()
    raw = glReadPixels(0, 0, FBO_SIZE, FBO_SIZE, GL_RGBA, GL_UNSIGNED_BYTE)
    return np.frombuffer(raw, np.uint8).reshape(FBO_SIZE, FBO_SIZE, 4)


def ctypes_offset(byte_offset: int):
    import ctypes
    return ctypes.c_void_p(byte_offset)


# Fragment inputs to sweep.  Chosen to hit facing surfaces, silhouettes,
# back-faces, lit and unlit orientations, non-unit normals (which the shaders
# must renormalise) and non-zero world positions (for the grid and interference
# terms).
CASES = [
    ("facing_lit", (0.0, 0.0, 1.0), (0.0, 0.0, -100.0), (0.0, 0.0, 0.0)),
    ("facing_unlit", (0.0, 0.0, 1.0), (0.0, 0.0, -100.0), (3.3, -7.1, 2.5)),
    ("silhouette", (1.0, 0.0, 0.0), (0.0, 0.0, -100.0), (1.0, 2.0, 3.0)),
    ("oblique", (0.35, -0.62, 0.70), (12.0, -8.0, -140.0), (-4.5, 9.25, 1.75)),
    ("backfacing", (0.0, 0.0, -1.0), (0.0, 0.0, -100.0), (2.0, 2.0, 2.0)),
    ("non_unit_normal", (0.0, 0.0, 4.7), (5.0, 5.0, -60.0), (0.4, 0.6, 0.8)),
    ("away_from_light", (-0.3, 0.6, -0.7), (0.0, 0.0, -100.0), (6.1, -2.2, 8.4)),
    ("near_grazing", (0.98, 0.10, 0.17), (-3.0, 4.0, -200.0), (10.0, 10.0, 10.0)),
]

# Pixels to compare.  Several, because the screen-space modes vary with
# gl_FragCoord and a single sample could agree by luck.
SAMPLE_PIXELS = [(3, 3), (4, 9), (8, 8), (11, 5), (13, 12)]

MODE_NAMES = sorted(MODE_FRAGMENTS)


@pytest.fixture(scope="module")
def programs(gl_context, fbo):
    """Link every real fragment shader against the pass-through vertex shader."""
    from faceforge.rendering.shader_program import load_shader_source

    out = {}
    for mode, (_, filename) in MODE_FRAGMENTS.items():
        frag = load_shader_source(filename)
        vert = PASSTHROUGH_VERT_POINTS if filename == "points.frag" else PASSTHROUGH_VERT
        if filename == "points.frag":
            # points.frag reads gl_PointCoord, which is undefined for a
            # triangle primitive.  Its maths is covered on the CPU side by
            # test_shader_semantics; there is nothing meaningful to compare
            # here, so it is linked (proving it links against a minimal vertex
            # stage) but not rendered.
            out[mode] = (_compile_program(vert, frag), False)
            continue
        out[mode] = (_compile_program(vert, frag), True)
    return out


def test_every_fragment_shader_links_on_a_real_driver(programs):
    """glslang links these too; this is the driver's own front end agreeing."""
    assert len(programs) == 16
    for mode, (prog, _) in programs.items():
        assert prog > 0, f"{mode}: program id {prog}"


@pytest.mark.parametrize("mode", MODE_NAMES)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_numpy_transliteration_matches_the_driver(programs, mode, case):
    """The core cross-check: numpy must reproduce the driver's pixels."""
    name, normal, view_pos, world_pos = case
    prog, renderable = programs[mode]
    if not renderable:
        pytest.skip("points.frag needs gl_PointCoord, undefined for a triangle")

    u = Uniforms(use_vertex_color=0)
    vertex_color = vec3(0.0, 0.0, 0.0)

    _set_uniforms_bound(prog, u)
    img = _render_flat_quad(prog, normal, view_pos, world_pos, vertex_color)

    for (px, py) in SAMPLE_PIXELS:
        # gl_FragCoord is (x + 0.5, y + 0.5) with the origin at the lower left,
        # which is the row order glReadPixels returns.
        frag_coord = np.array([[px + 0.5, py + 0.5]])
        v = Varyings(
            normal=np.array([normal], np.float64),
            view_pos=np.array([view_pos], np.float64),
            world_pos=np.array([world_pos], np.float64),
            vertex_color=np.array([vertex_color], np.float64),
            frag_coord=frag_coord,
        )
        fn, _ = MODE_FRAGMENTS[mode]
        expected = np.clip(fn(v, u)[0], 0.0, 1.0) * 255.0
        got = img[py, px].astype(np.float64)
        delta = np.abs(expected - got)
        assert delta.max() <= TOLERANCE_8BIT, (
            f"{mode} / {name} / pixel ({px},{py}): numpy and the driver disagree.\n"
            f"  numpy  RGBA = {np.round(expected, 2).tolist()}\n"
            f"  driver RGBA = {got.tolist()}\n"
            f"  |delta|     = {np.round(delta, 2).tolist()} (tolerance {TOLERANCE_8BIT})\n"
            "  A disagreement here means tools/glsl_cpu.py no longer transliterates "
            "this shader, so the invariants in test_shader_semantics.py are being "
            "asserted against the wrong maths."
        )


def _set_uniforms_bound(prog: int, u: Uniforms) -> None:
    from OpenGL.GL import glUseProgram
    glUseProgram(prog)
    _set_uniforms(prog, u)


@pytest.mark.parametrize("mode", ["SOLID", "MEDICAL", "CARTOON", "HOLOGRAM", "BLUEPRINT"])
def test_vertex_colour_path_also_agrees(programs, mode):
    """The uUseVertexColor branch is a second code path and needs its own check."""
    prog, renderable = programs[mode]
    assert renderable
    u = Uniforms(use_vertex_color=1)
    vertex_color = vec3(0.85, 0.20, 0.35)
    normal, view_pos, world_pos = (0.3, -0.5, 0.8), (4.0, -2.0, -120.0), (1.5, 2.5, 3.5)

    _set_uniforms_bound(prog, u)
    img = _render_flat_quad(prog, normal, view_pos, world_pos, vertex_color)

    px, py = 8, 8
    v = Varyings(
        normal=np.array([normal], np.float64),
        view_pos=np.array([view_pos], np.float64),
        world_pos=np.array([world_pos], np.float64),
        vertex_color=np.array([vertex_color], np.float64),
        frag_coord=np.array([[px + 0.5, py + 0.5]]),
    )
    fn, _ = MODE_FRAGMENTS[mode]
    expected = np.clip(fn(v, u)[0], 0.0, 1.0) * 255.0
    got = img[py, px].astype(np.float64)
    assert np.abs(expected - got).max() <= TOLERANCE_8BIT, (
        f"{mode} vertex-colour path: numpy {np.round(expected, 2).tolist()} vs "
        f"driver {got.tolist()}"
    )


@pytest.mark.parametrize("opacity", [1.0, 0.5, 0.25])
def test_opacity_reaches_the_alpha_channel_on_the_driver(programs, opacity):
    """Guards the blending contract at the driver, not just in the CPU model."""
    prog, _ = programs["SOLID"]
    u = Uniforms(opacity=opacity)
    _set_uniforms_bound(prog, u)
    img = _render_flat_quad(prog, (0.0, 0.0, 1.0), (0.0, 0.0, -100.0),
                            (0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0))
    got_alpha = img[8, 8, 3] / 255.0
    assert abs(got_alpha - opacity) <= TOLERANCE_8BIT / 255.0, (
        f"uOpacity={opacity} produced framebuffer alpha {got_alpha}"
    )


def test_alpha_modulating_modes_really_drop_alpha_on_the_driver(programs):
    """The five blending modes must emit alpha < uOpacity at a facing fragment.

    This is the property that made the opacity-default change safe: with
    blending off these modes composite their own fractional alpha into a dark
    solid, which is why gl_material._MODE_NEEDS_BLENDING exists.
    """
    from tools.glsl_cpu import ALPHA_MODULATING_MODES

    checked = []
    for mode in sorted(ALPHA_MODULATING_MODES):
        prog, renderable = programs[mode]
        if not renderable:
            continue
        _set_uniforms_bound(prog, Uniforms(opacity=1.0))
        img = _render_flat_quad(prog, (0.0, 0.0, 1.0), (0.0, 0.0, -100.0),
                                (0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0))
        alpha = img[8, 8, 3] / 255.0
        assert alpha < 0.95, f"{mode}: facing alpha {alpha} is not fractional"
        checked.append(mode)
    assert len(checked) == 4, f"expected 4 renderable blending modes, checked {checked}"


def test_context_kind_is_reported(gl_context):
    """Record which driver these results came from — it bounds their scope."""
    assert gl_context.gl_version, "no GL_VERSION string"
    assert gl_context.max_clip_distances >= 1, (
        f"driver reports {gl_context.max_clip_distances} clip distances; "
        "default.vert writes gl_ClipDistance[0] and needs at least 1"
    )
