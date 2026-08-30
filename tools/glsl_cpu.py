"""A numpy transliteration of FaceForge's fragment shaders.

Why
---
``tests/rendering/test_shader_compile.py`` compiles all 18 shaders and links all
16 programs with ``glslangValidator``.  That proves the GLSL is *syntactically*
valid and that every identifier resolves.  It cannot prove the shader computes
the right thing: a shader that returns pure black, inverts its lighting, or
emits alpha 0 for every fragment compiles perfectly.

This module closes that gap by re-implementing each fragment shader's colour
maths in numpy, so invariants can be asserted over a swept input domain on the
CPU with no GPU and no context:

  * output range -- every channel and the alpha in [0, 1] after the shader's own
    clamping, for every input in the domain;
  * alpha semantics -- exactly the five modes in ``_MODE_NEEDS_BLENDING`` may
    return an alpha below ``uOpacity``; the other eleven must return exactly
    ``uOpacity``;
  * lighting monotonicity -- a surface turned toward the light must not get
    darker;
  * clip-plane behaviour -- the sign convention of ``gl_ClipDistance[0]``.

A transliteration only carries weight if it actually matches the GLSL, so it is
not trusted on inspection: ``tests/rendering/test_shader_gpu_agreement.py``
links each real ``.frag`` against a pass-through vertex shader, renders a grid
of known varying values, and asserts these functions reproduce the driver's
pixels.  The invariant tests are meaningful because that agreement test passes.

Conventions
-----------
Every function is vectorised over leading axes, with vectors in the trailing
axis (``(..., 3)``).  Scalars are ``(...)``.  Colours are linear, unclamped
until the shader clamps them, matching GLSL.  Inputs mirror the varyings from
``default.vert`` -- in particular ``normal`` is the *raw, un-normalised*
``vNormal``, because normalising before interpolation is exactly the mistake
``default.vert`` documents not making.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# GLSL built-ins.  Named after the GLSL function so a reader can diff the
# transliteration against the shader line by line.
# ---------------------------------------------------------------------------

EPS = 1e-20


def glsl_normalize(v: np.ndarray) -> np.ndarray:
    """``normalize(v)``.  Zero-length input is undefined in GLSL; kept finite here."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, EPS)


def glsl_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(a, np.float64) * np.asarray(b, np.float64), axis=-1)


def glsl_length(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(v, np.float64), axis=-1)


def glsl_mix(a, b, t):
    """``mix(a, b, t) = a*(1-t) + b*t``.

    Plain numpy broadcasting, with no shape guessing.  An earlier version tried
    to be helpful by expanding a scalar ``t`` to ``t[..., None]`` when ``a``
    looked like a vec3 -- which silently produced ``(..., 1, 3)`` results
    wherever a call site had already expanded ``t`` itself, and would also have
    misfired on any batch of exactly 3 fragments.  Call sites that mix vec3s by
    a per-fragment scalar pass ``t[..., None]`` explicitly; that is the whole
    convention.
    """
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a * (1.0 - np.asarray(t, np.float64)) + b * np.asarray(t, np.float64)


def glsl_clamp(x, lo, hi):
    return np.clip(np.asarray(x, np.float64), lo, hi)


def glsl_step(edge, x):
    """``step(edge, x)`` -- 0.0 when x < edge, else 1.0."""
    return np.where(np.asarray(x, np.float64) < np.asarray(edge, np.float64), 0.0, 1.0)


def glsl_smoothstep(e0, e1, x):
    """``smoothstep(e0, e1, x)``.  Correct for e0 > e1 (points.frag relies on it)."""
    e0, e1 = np.asarray(e0, np.float64), np.asarray(e1, np.float64)
    x = np.asarray(x, np.float64)
    denom = e1 - e0
    t = np.clip((x - e0) / np.where(np.abs(denom) < EPS, EPS, denom), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def glsl_fract(x):
    """``fract(x) = x - floor(x)``."""
    x = np.asarray(x, np.float64)
    return x - np.floor(x)


def glsl_mod(x, y):
    """``mod(x, y) = x - y*floor(x/y)``."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return x - y * np.floor(x / y)


def glsl_reflect(i: np.ndarray, n: np.ndarray) -> np.ndarray:
    """``reflect(I, N) = I - 2*dot(N, I)*N``."""
    i, n = np.asarray(i, np.float64), np.asarray(n, np.float64)
    return i - 2.0 * glsl_dot(n, i)[..., None] * n


def glsl_pow(x, y):
    """``pow(x, y)``.  Undefined in GLSL for x < 0; clamped at 0 to stay finite."""
    return np.power(np.maximum(np.asarray(x, np.float64), 0.0), y)


def vec3(x, y=None, z=None) -> np.ndarray:
    if y is None:
        return np.array([x, x, x], dtype=np.float64)
    return np.array([x, y, z], dtype=np.float64)


# ---------------------------------------------------------------------------
# Shader inputs
# ---------------------------------------------------------------------------

@dataclass
class Varyings:
    """The interpolated per-fragment inputs from default.vert / points.vert."""

    normal: np.ndarray                 # vNormal, raw (NOT unit length)
    view_pos: np.ndarray               # vViewPos
    world_pos: np.ndarray              # vWorldPos
    vertex_color: np.ndarray           # vVertexColor
    frag_coord: np.ndarray             # gl_FragCoord.xy
    point_coord: np.ndarray | None = None   # gl_PointCoord, points.frag only

    def broadcast_shape(self) -> tuple:
        return np.asarray(self.normal, np.float64).shape[:-1]


@dataclass
class Uniforms:
    """Uniform values.  Defaults match LightSetup() and a typical material."""

    color: np.ndarray = field(default_factory=lambda: vec3(0.82, 0.76, 0.68))
    opacity: float = 1.0
    use_vertex_color: int = 0
    shininess: float = 30.0
    ambient_color: np.ndarray = field(default_factory=lambda: vec3(0.25, 0.25, 0.28))
    light_dir: np.ndarray = field(default_factory=lambda: glsl_normalize(vec3(0.3, -0.6, 0.7)))
    light_color: np.ndarray = field(default_factory=lambda: vec3(1.0, 0.97, 0.92))
    # phong_pointlight only
    has_point_light: int = 0
    point_light_pos: np.ndarray = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    point_light_color: np.ndarray = field(default_factory=lambda: vec3(1.0, 0.95, 0.85))
    point_light_intensity: float = 1.5
    point_light_range: float = 400.0


# ---------------------------------------------------------------------------
# _common.glsl / _lighting.glsl helpers
# ---------------------------------------------------------------------------

def ff_normal(v: Varyings) -> np.ndarray:
    return glsl_normalize(v.normal)


def ff_view_dir(v: Varyings) -> np.ndarray:
    return glsl_normalize(-np.asarray(v.view_pos, np.float64))


def ff_base_color(v: Varyings, u: Uniforms) -> np.ndarray:
    """``uUseVertexColor != 0 ? vVertexColor : uColor``."""
    if u.use_vertex_color != 0:
        return np.asarray(v.vertex_color, np.float64)
    return np.broadcast_to(
        np.asarray(u.color, np.float64), (*v.broadcast_shape(), 3)
    ).astype(np.float64)


def ff_edge(n: np.ndarray, view: np.ndarray) -> np.ndarray:
    """``1.0 - abs(dot(N, V))`` -- 0 facing the camera, 1 at the silhouette."""
    return 1.0 - np.abs(glsl_dot(n, view))


def ff_luma(c: np.ndarray) -> np.ndarray:
    """Rec.601 luma."""
    return glsl_dot(c, vec3(0.299, 0.587, 0.114))


def ff_specular(n, view, light, shininess) -> np.ndarray:
    """``pow(max(dot(V, reflect(-L, N)), 0.0), shininess)``."""
    r = glsl_reflect(-np.asarray(light, np.float64), n)
    return glsl_pow(np.maximum(glsl_dot(view, r), 0.0), shininess)


def ff_light_dir(u: Uniforms) -> np.ndarray:
    return glsl_normalize(u.light_dir)


def clip_distance(world_pos, clip_enabled: int, clip_plane) -> np.ndarray:
    """``gl_ClipDistance[0]`` as written by default.vert and points.vert.

    Negative clips the vertex.  Returns 1.0 when the cutaway is off, so a
    context that leaves GL_CLIP_DISTANCE0 enabled still shows everything.
    """
    world_pos = np.asarray(world_pos, np.float64)
    if clip_enabled == 0:
        return np.ones(world_pos.shape[:-1], np.float64)
    p = np.asarray(clip_plane, np.float64)
    return glsl_dot(world_pos, p[..., :3]) + p[..., 3]


def _rgba(rgb: np.ndarray, alpha) -> np.ndarray:
    rgb = np.asarray(rgb, np.float64)
    alpha = np.broadcast_to(np.asarray(alpha, np.float64), rgb.shape[:-1])
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


# ---------------------------------------------------------------------------
# The 16 modes.  Each mirrors its .frag line by line.
# ---------------------------------------------------------------------------

def frag_phong_pointlight(v: Varyings, u: Uniforms) -> np.ndarray:
    """phong_pointlight.frag -- used by both SOLID and OPAQUE."""
    base = ff_base_color(v, u)
    n, view = ff_normal(v), ff_view_dir(v)
    light = ff_light_dir(u)
    diff = np.maximum(glsl_dot(n, light), 0.0)
    spec = ff_specular(n, view, light, u.shininess)
    color = (
        np.asarray(u.ambient_color, np.float64) * base
        + diff[..., None] * np.asarray(u.light_color, np.float64) * base
        + spec[..., None] * np.asarray(u.light_color, np.float64) * 0.3
    )
    if u.has_point_light != 0:
        to_light = np.asarray(u.point_light_pos, np.float64) - np.asarray(
            v.view_pos, np.float64
        )
        dist = glsl_length(to_light)
        lp = to_light / np.maximum(dist, 0.001)[..., None]
        rng = u.point_light_range
        atten = 1.0 / (1.0 + dist * dist / (rng * rng))
        atten = atten * u.point_light_intensity
        diff_p = np.maximum(glsl_dot(n, lp), 0.0)
        rp = glsl_reflect(-lp, n)
        spec_p = glsl_pow(np.maximum(glsl_dot(view, rp), 0.0), u.shininess)
        plc = np.asarray(u.point_light_color, np.float64)
        color = color + atten[..., None] * (
            diff_p[..., None] * plc * base + spec_p[..., None] * plc * 0.3
        )
    return _rgba(color, u.opacity)


def frag_wireframe(v: Varyings, u: Uniforms) -> np.ndarray:
    return _rgba(ff_base_color(v, u), u.opacity)


def frag_xray(v: Varyings, u: Uniforms) -> np.ndarray:
    n, view = ff_normal(v), ff_view_dir(v)
    edge = ff_edge(n, view)
    fresnel = glsl_pow(edge, 1.5)
    light = ff_light_dir(u)
    diff = np.maximum(glsl_dot(n, light), 0.0) * 0.3
    base = ff_base_color(v, u)
    color = base * (fresnel * 0.8 + diff + 0.1)[..., None]
    alpha = u.opacity * (fresnel * 0.8 + 0.15)
    return _rgba(color, alpha)


def frag_points(v: Varyings, u: Uniforms) -> np.ndarray:
    """points.frag.  Discarded fragments are returned as NaN, not as a colour."""
    if v.point_coord is None:
        raise ValueError("points.frag needs gl_PointCoord; set Varyings.point_coord")
    coord = np.asarray(v.point_coord, np.float64) * 2.0 - 1.0
    dist = glsl_dot(coord, coord)
    alpha = u.opacity * glsl_smoothstep(1.0, 0.8, dist)
    rgb = np.broadcast_to(
        np.asarray(u.color, np.float64), (*np.shape(dist), 3)
    ).astype(np.float64)
    out = _rgba(rgb, alpha)
    out[dist > 1.0] = np.nan       # discard
    return out


def _hatch_common(v: Varyings, diff: np.ndarray, period: float, thresh: float):
    sc = np.asarray(v.frag_coord, np.float64)
    h1 = glsl_step(thresh, glsl_mod(sc[..., 0] - sc[..., 1], period))
    h2 = glsl_step(thresh, glsl_mod(sc[..., 0] + sc[..., 1], period))
    return h1, h2, 1.0 - diff


def frag_illustration(v: Varyings, u: Uniforms) -> np.ndarray:
    n, view, light = ff_normal(v), ff_view_dir(v), ff_light_dir(u)
    ndotl = glsl_dot(n, light)
    diff = glsl_clamp(ndotl, 0.0, 1.0)
    t = (ndotl + 1.0) * 0.5
    gooch = glsl_mix(0.25, 0.92, t)
    edge = ff_edge(n, view)
    contour = glsl_smoothstep(0.0, 0.45, edge)
    edge_darken = 1.0 - contour * 0.85
    h1, h2, shadow = _hatch_common(v, diff, 6.0, 1.5)
    hatch_mask = np.ones_like(shadow)
    hatch_mask = np.where(
        shadow > 0.35,
        hatch_mask * glsl_mix(1.0, h1, glsl_smoothstep(0.35, 0.65, shadow) * 0.45),
        hatch_mask,
    )
    hatch_mask = np.where(
        shadow > 0.65,
        hatch_mask * glsl_mix(1.0, h2, glsl_smoothstep(0.65, 0.9, shadow) * 0.35),
        hatch_mask,
    )
    spec = ff_specular(n, view, light, max(u.shininess, 20.0))
    highlight = spec * 0.15
    base_lum = ff_luma(ff_base_color(v, u))
    material_tint = glsl_mix(0.85, 1.0, base_lum)
    grey = glsl_clamp(gooch * edge_darken * hatch_mask * material_tint + highlight, 0.0, 1.0)
    final = glsl_mix(vec3(0.06, 0.06, 0.08), vec3(0.98, 0.96, 0.93), grey[..., None])
    return _rgba(final, u.opacity)


def frag_sepia(v: Varyings, u: Uniforms) -> np.ndarray:
    n, view, light = ff_normal(v), ff_view_dir(v), ff_light_dir(u)
    ndotl = glsl_dot(n, light)
    diff = glsl_clamp(ndotl, 0.0, 1.0)
    t = (ndotl + 1.0) * 0.5
    tone = glsl_mix(0.20, 0.88, t)
    edge = ff_edge(n, view)
    contour = glsl_smoothstep(0.0, 0.40, edge)
    edge_darken = 1.0 - contour * 0.80
    h1, h2, shadow = _hatch_common(v, diff, 5.0, 1.2)
    hm = np.ones_like(shadow)
    hm = np.where(shadow > 0.30,
                  hm * glsl_mix(1.0, h1, glsl_smoothstep(0.30, 0.60, shadow) * 0.40), hm)
    hm = np.where(shadow > 0.60,
                  hm * glsl_mix(1.0, h2, glsl_smoothstep(0.60, 0.85, shadow) * 0.30), hm)
    mt = glsl_mix(0.85, 1.0, ff_luma(ff_base_color(v, u)))
    grey = glsl_clamp(tone * edge_darken * hm * mt, 0.0, 1.0)
    final = glsl_mix(vec3(0.20, 0.12, 0.06), vec3(0.95, 0.88, 0.75), grey[..., None])
    return _rgba(final, u.opacity)


def frag_color_atlas(v: Varyings, u: Uniforms) -> np.ndarray:
    n, view, light = ff_normal(v), ff_view_dir(v), ff_light_dir(u)
    base = ff_base_color(v, u)
    ndotl = glsl_dot(n, light)
    diff = glsl_clamp(ndotl, 0.0, 1.0)
    t = (ndotl + 1.0) * 0.5
    cool = base * 0.55 + vec3(0.0, 0.0, 0.06)
    warm = base * 1.05 + vec3(0.04, 0.02, 0.0)
    gooch = glsl_mix(cool, warm, t[..., None])
    contour = glsl_smoothstep(0.0, 0.42, ff_edge(n, view))
    edge_darken = 1.0 - contour * 0.75
    sc = np.asarray(v.frag_coord, np.float64)
    hatch = glsl_step(1.5, glsl_mod(sc[..., 0] - sc[..., 1], 7.0))
    shadow = 1.0 - diff
    hm = np.ones_like(shadow)
    hm = np.where(shadow > 0.40,
                  hm * glsl_mix(1.0, hatch, glsl_smoothstep(0.40, 0.70, shadow) * 0.30), hm)
    spec = ff_specular(n, view, light, max(u.shininess, 25.0)) * 0.12
    final = gooch * (edge_darken * hm)[..., None] + spec[..., None]
    return _rgba(np.clip(final, 0.0, 1.0), u.opacity)


def frag_pen_ink(v: Varyings, u: Uniforms) -> np.ndarray:
    n, view, light = ff_normal(v), ff_view_dir(v), ff_light_dir(u)
    ndotl = glsl_dot(n, light)
    diff = glsl_clamp(ndotl, 0.0, 1.0)
    outline = glsl_smoothstep(0.0, 0.55, ff_edge(n, view))
    ink = 1.0 - outline * 0.95
    sc = np.asarray(v.frag_coord, np.float64)
    shadow = 1.0 - diff
    spacing = 4.0
    cell = glsl_mod(sc, spacing)
    dot_dist = glsl_length(cell - spacing * 0.5)
    dot_radius = shadow * 1.8
    stipple = glsl_smoothstep(dot_radius, dot_radius + 0.5, dot_dist)
    hatch = glsl_step(0.8, glsl_mod(sc[..., 0] - sc[..., 1], 3.0))
    deep = np.where(
        shadow > 0.60,
        glsl_mix(1.0, hatch, glsl_smoothstep(0.60, 0.90, shadow) * 0.50),
        1.0,
    )
    brightness = glsl_clamp(stipple * deep * ink, 0.0, 1.0)
    return _rgba(np.repeat(brightness[..., None], 3, axis=-1), u.opacity)


def frag_medical(v: Varyings, u: Uniforms) -> np.ndarray:
    n, view, light = ff_normal(v), ff_view_dir(v), ff_light_dir(u)
    base = ff_base_color(v, u)
    lum = ff_luma(base)
    saturated = np.clip(glsl_mix(np.repeat(lum[..., None], 3, -1), base, 1.25), 0.0, 1.0)
    ndotl = glsl_dot(n, light)
    diff = glsl_clamp(ndotl, 0.0, 1.0)
    # NOTE: uses the RAW uLightDir, not the normalised ffLightDir().
    ld = np.asarray(u.light_dir, np.float64)
    fill_dir = glsl_normalize(vec3(-ld[0], -ld[1], ld[2]))
    fill_diff = glsl_clamp(glsl_dot(n, fill_dir), 0.0, 1.0) * 0.35
    ao = 0.45 + 0.55 * diff
    spec = ff_specular(n, view, light, max(u.shininess, 40.0))
    rim = glsl_pow(1.0 - np.maximum(glsl_dot(n, view), 0.0), 3.0) * 0.15
    color = (
        saturated * (
            np.asarray(u.ambient_color, np.float64) * ao[..., None]
            + diff[..., None] * np.asarray(u.light_color, np.float64)
            + fill_diff[..., None] * vec3(0.6, 0.65, 0.7)
        )
        + spec[..., None] * np.asarray(u.light_color, np.float64) * 0.25
        + rim[..., None]
    )
    edge_fade = glsl_smoothstep(0.0, 0.20, 1.0 - ff_edge(n, view))
    color = color * (edge_fade * 0.3 + 0.7)[..., None]
    return _rgba(np.clip(color, 0.0, 1.0), u.opacity)


def frag_hologram(v: Varyings, u: Uniforms) -> np.ndarray:
    n, view = ff_normal(v), ff_view_dir(v)
    edge = ff_edge(n, view)
    facing = 1.0 - edge
    fresnel = glsl_pow(edge, 2.0)
    fc = np.asarray(v.frag_coord, np.float64)
    scanline = 0.85 + 0.15 * glsl_step(0.5, glsl_fract(fc[..., 1] * 0.25))
    wp = np.asarray(v.world_pos, np.float64)
    interference = 0.90 + 0.10 * np.sin(wp[..., 2] * 3.0 + wp[..., 1] * 1.5)
    if u.use_vertex_color != 0:
        vc = np.asarray(v.vertex_color, np.float64)
        core = vc * 0.9
        edge_col = glsl_mix(vc, vec3(1.0), 0.4)
    else:
        core = np.broadcast_to(vec3(0.0, 0.85, 0.95), (*edge.shape, 3))
        edge_col = np.broadcast_to(vec3(0.4, 0.9, 1.0), (*edge.shape, 3))
    holo = glsl_mix(core, edge_col, fresnel[..., None])
    glow = (fresnel * 0.85 + 0.08) * scanline * interference
    interior = facing * 0.06
    alpha = glsl_clamp(u.opacity * (glow + interior), 0.0, 1.0)
    bloom = holo * (glow * 1.3)[..., None]
    final = np.clip(bloom + holo * interior[..., None], 0.0, 1.0)
    return _rgba(final, alpha)


def frag_cartoon(v: Varyings, u: Uniforms) -> np.ndarray:
    n, view, light = ff_normal(v), ff_view_dir(v), ff_light_dir(u)
    base = ff_base_color(v, u)
    ndotl = glsl_dot(n, light)
    diff = glsl_clamp(ndotl, 0.0, 1.0)
    band = np.select(
        [diff > 0.85, diff > 0.55, diff > 0.25],
        [1.0, 0.75, 0.50],
        default=0.30,
    )
    lum = ff_luma(base)
    saturated = np.clip(glsl_mix(np.repeat(lum[..., None], 3, -1), base, 1.40), 0.0, 1.0)
    color = saturated * band[..., None]
    spec = ff_specular(n, view, light, 80.0)
    spec_band = glsl_step(0.60, spec)
    color = color + (spec_band * 0.45)[..., None]
    outline = glsl_smoothstep(0.0, 0.30, 1.0 - ff_edge(n, view))
    color = color * outline[..., None]
    return _rgba(np.clip(color, 0.0, 1.0), u.opacity)


def frag_porcelain(v: Varyings, u: Uniforms) -> np.ndarray:
    n, view, light = ff_normal(v), ff_view_dir(v), ff_light_dir(u)
    base = ff_base_color(v, u)
    lum = ff_luma(base)
    porc = glsl_mix(np.repeat(lum[..., None], 3, -1), base, 0.35)
    porc = glsl_mix(porc, vec3(0.92, 0.90, 0.88), 0.55)
    wrap = 0.45
    ndotl = glsl_dot(n, light)
    wrap_diff = glsl_clamp((ndotl + wrap) / (1.0 + wrap), 0.0, 1.0)
    scatter = glsl_clamp(1.0 - wrap_diff, 0.0, 1.0)
    sss = vec3(0.90, 0.55, 0.45) * (scatter * 0.18)[..., None]
    half_vec = glsl_normalize(light + view)
    spec_angle = np.maximum(glsl_dot(n, half_vec), 0.0)
    spec = glsl_pow(spec_angle, 120.0) * 0.55
    rim = glsl_pow(1.0 - np.maximum(glsl_dot(n, view), 0.0), 2.5) * 0.12
    color = (
        porc * 0.50
        + porc * (wrap_diff * 0.65)[..., None]
        + sss
        + spec[..., None]
        + rim[..., None]
    )
    return _rgba(np.clip(color, 0.0, 1.0), u.opacity)


def frag_blueprint(v: Varyings, u: Uniforms) -> np.ndarray:
    n, view, light = ff_normal(v), ff_view_dir(v), ff_light_dir(u)
    edge = ff_edge(n, view)
    wireframe = glsl_smoothstep(0.0, 0.50, edge)
    wp = np.asarray(v.world_pos, np.float64)
    grid_uv = glsl_fract(wp[..., [0, 2]] / 5.0)
    grid_line = 1.0 - glsl_step(
        0.04,
        np.minimum(
            np.minimum(grid_uv[..., 0], 1.0 - grid_uv[..., 0]),
            np.minimum(grid_uv[..., 1], 1.0 - grid_uv[..., 1]),
        ),
    )
    grid_line = grid_line * 0.12
    ndotl = glsl_dot(n, light)
    depth_shade = glsl_clamp((ndotl + 1.0) * 0.5, 0.0, 1.0) * 0.08
    line_intensity = glsl_clamp(wireframe * 0.90 + grid_line + depth_shade, 0.0, 1.0)
    bg_blue = vec3(0.05, 0.12, 0.28)
    if u.use_vertex_color != 0:
        line_white = np.asarray(v.vertex_color, np.float64)
    else:
        line_white = np.broadcast_to(vec3(0.85, 0.90, 1.0), (*edge.shape, 3))
    final = glsl_mix(bg_blue, line_white, line_intensity[..., None])
    alpha = glsl_clamp(u.opacity * (wireframe * 0.80 + 0.15 + grid_line), 0.0, 1.0)
    return _rgba(final, alpha)


def frag_thermal(v: Varyings, u: Uniforms) -> np.ndarray:
    n, view, light = ff_normal(v), ff_view_dir(v), ff_light_dir(u)
    facing = 1.0 - ff_edge(n, view)
    ndotl = glsl_dot(n, light)
    diff = glsl_clamp(ndotl, 0.0, 1.0)
    heat = facing * 0.6 + diff * 0.4
    lum = ff_luma(ff_base_color(v, u))
    heat = glsl_clamp(heat * 0.7 + lum * 0.3, 0.0, 1.0)
    h = heat[..., None]
    col = np.where(
        h < 0.25,
        glsl_mix(vec3(0.0, 0.0, 0.08), vec3(0.0, 0.0, 0.8), h / 0.25),
        np.where(
            h < 0.50,
            glsl_mix(vec3(0.0, 0.0, 0.8), vec3(0.85, 0.0, 0.65), (h - 0.25) / 0.25),
            np.where(
                h < 0.75,
                glsl_mix(vec3(0.85, 0.0, 0.65), vec3(1.0, 0.9, 0.0), (h - 0.50) / 0.25),
                glsl_mix(vec3(1.0, 0.9, 0.0), vec3(1.0, 1.0, 1.0), (h - 0.75) / 0.25),
            ),
        ),
    )
    fc = np.asarray(v.frag_coord, np.float64)
    scanline = 0.92 + 0.08 * glsl_step(0.5, glsl_fract(fc[..., 1] * 0.5))
    return _rgba(col * scanline[..., None], u.opacity)


def frag_ethereal(v: Varyings, u: Uniforms) -> np.ndarray:
    n, view, light = ff_normal(v), ff_view_dir(v), ff_light_dir(u)
    base = ff_base_color(v, u)
    edge = ff_edge(n, view)
    facing = 1.0 - edge
    hue = edge * 2.5 + glsl_dot(n, vec3(0.3, 0.6, 0.7)) * 1.5
    iridescent = np.stack(
        [
            0.5 + 0.5 * np.cos(hue),
            0.5 + 0.5 * np.cos(hue + 2.094),
            0.5 + 0.5 * np.cos(hue + 4.189),
        ],
        axis=-1,
    )
    color = glsl_mix(base * 0.6, iridescent, (edge * 0.7)[..., None])
    ndotl = glsl_dot(n, light)
    diff = glsl_clamp((ndotl + 0.5) / 1.5, 0.0, 1.0)
    glow = glsl_pow(edge, 1.8) * 0.65
    r = glsl_reflect(-light, n)
    spec = glsl_pow(np.maximum(glsl_dot(view, r), 0.0), 60.0) * 0.35
    backlight = glsl_clamp(-ndotl * 0.3, 0.0, 1.0)
    final = (
        color * (diff * 0.7)[..., None]
        + color * glow[..., None]
        + iridescent * spec[..., None]
        + base * (backlight * 0.15)[..., None]
        + vec3(0.05, 0.03, 0.08)
    )
    alpha = glsl_clamp(u.opacity * (facing * 0.5 + glow + 0.15), 0.0, 1.0)
    return _rgba(np.clip(final, 0.0, 1.0), alpha)


# RenderMode name -> (numpy fragment function, fragment shader filename).
# SOLID and OPAQUE deliberately share phong_pointlight.frag: OPAQUE differs
# only in GL state (blending off), not in shader code.
MODE_FRAGMENTS: dict[str, tuple] = {
    "SOLID": (frag_phong_pointlight, "phong_pointlight.frag"),
    "WIREFRAME": (frag_wireframe, "wireframe.frag"),
    "XRAY": (frag_xray, "xray.frag"),
    "POINTS": (frag_points, "points.frag"),
    "OPAQUE": (frag_phong_pointlight, "phong_pointlight.frag"),
    "ILLUSTRATION": (frag_illustration, "illustration.frag"),
    "SEPIA": (frag_sepia, "sepia.frag"),
    "COLOR_ATLAS": (frag_color_atlas, "color_atlas.frag"),
    "PEN_INK": (frag_pen_ink, "pen_ink.frag"),
    "MEDICAL": (frag_medical, "medical.frag"),
    "HOLOGRAM": (frag_hologram, "hologram.frag"),
    "CARTOON": (frag_cartoon, "cartoon.frag"),
    "PORCELAIN": (frag_porcelain, "porcelain.frag"),
    "BLUEPRINT": (frag_blueprint, "blueprint.frag"),
    "THERMAL": (frag_thermal, "thermal.frag"),
    "ETHEREAL": (frag_ethereal, "ethereal.frag"),
}

# The five modes whose shader computes a fractional alpha of its own.  Must
# equal gl_material._MODE_NEEDS_BLENDING; a test asserts that.
ALPHA_MODULATING_MODES = frozenset({"XRAY", "HOLOGRAM", "BLUEPRINT", "ETHEREAL", "POINTS"})


def sample_domain(n_side: int = 9, seed: int = 20260829) -> Varyings:
    """A swept, reproducible input domain covering the interesting geometry.

    Normals sweep the full sphere (so ``dot(N, L)`` covers [-1, 1] and both
    facing and silhouette orientations occur), view positions span near and far,
    world positions span several grid periods, and fragment coordinates span
    several hatch periods.  Seeded, so a failure is reproducible.
    """
    rng = np.random.default_rng(seed)
    # Fibonacci sphere: even coverage without clustering at the poles.
    k = np.arange(n_side * n_side, dtype=np.float64)
    total = float(n_side * n_side)
    phi = np.arccos(1.0 - 2.0 * (k + 0.5) / total)
    theta = np.pi * (1.0 + 5.0**0.5) * (k + 0.5)
    normals = np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1
    )
    # Deliberately NOT unit length: vNormal arrives interpolated, and every
    # shader is expected to renormalise it.
    normals = normals * rng.uniform(0.4, 2.5, size=(normals.shape[0], 1))
    m = normals.shape[0]
    view_pos = np.stack(
        [
            rng.uniform(-60, 60, m),
            rng.uniform(-60, 60, m),
            rng.uniform(-400, -20, m),   # in front of the camera
        ],
        axis=-1,
    )
    world_pos = rng.uniform(-30, 30, size=(m, 3))
    vertex_color = rng.uniform(0.0, 1.0, size=(m, 3))
    frag_coord = rng.uniform(0.5, 64.5, size=(m, 2))
    point_coord = rng.uniform(0.0, 1.0, size=(m, 2))
    return Varyings(
        normal=normals, view_pos=view_pos, world_pos=world_pos,
        vertex_color=vertex_color, frag_coord=frag_coord, point_coord=point_coord,
    )
