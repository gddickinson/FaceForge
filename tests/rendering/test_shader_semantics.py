"""Shader *semantics* on the CPU: does the GLSL compute the right thing?

`test_shader_static.py` inspects shader text.  `test_shader_compile.py` compiles
all 18 shaders and links all 16 programs with glslangValidator.  Between them
they prove the GLSL is well-formed and that every identifier resolves.

None of that constrains behaviour.  A shader that returns pure black, inverts
its lighting, emits alpha 0 everywhere, or overflows to NaN compiles and links
without complaint.  This module asserts behaviour instead, over a swept input
domain, using the numpy transliteration in ``tools.glsl_cpu``.

The transliteration is not taken on trust: ``test_shader_gpu_agreement.py``
checks it against the real driver's pixels.  Read these two modules together --
this one says what the maths must satisfy, that one says the maths is really the
shader's.

No GPU, no GL context, no window server.  Runs anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.glsl_cpu import (
    ALPHA_MODULATING_MODES,
    MODE_FRAGMENTS,
    Uniforms,
    Varyings,
    clip_distance,
    ff_edge,
    ff_luma,
    ff_normal,
    ff_specular,
    ff_view_dir,
    glsl_normalize,
    glsl_smoothstep,
    sample_domain,
    vec3,
)

MODE_NAMES = sorted(MODE_FRAGMENTS)


@pytest.fixture(scope="module")
def domain() -> Varyings:
    return sample_domain(n_side=11)


@pytest.fixture(scope="module")
def uniforms() -> Uniforms:
    return Uniforms()


def evaluate(mode: str, v: Varyings, u: Uniforms) -> np.ndarray:
    fn, _ = MODE_FRAGMENTS[mode]
    return fn(v, u)


def finite_rows(out: np.ndarray) -> np.ndarray:
    """Drop discarded fragments (NaN rows), which only points.frag produces."""
    return out[~np.isnan(out).any(axis=-1)]


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------

def test_every_render_mode_has_a_transliteration():
    from faceforge.core.material import RenderMode

    assert set(MODE_FRAGMENTS) == {m.name for m in RenderMode}, (
        "MODE_FRAGMENTS must cover exactly the RenderMode enum"
    )


def test_transliterated_shader_files_exist_and_are_the_ones_the_renderer_uses():
    from faceforge.rendering.shader_program import _SHADER_DIR

    for mode, (_, filename) in MODE_FRAGMENTS.items():
        assert (_SHADER_DIR / filename).is_file(), f"{mode}: {filename} missing"


def test_alpha_modulating_set_matches_the_renderer_blending_table():
    """The CPU model and the GL state machine must agree on which modes blend."""
    from faceforge.core.material import RenderMode
    from faceforge.rendering.gl_material import _MODE_NEEDS_BLENDING

    assert ALPHA_MODULATING_MODES == {m.name for m in _MODE_NEEDS_BLENDING}, (
        "the shaders that compute their own alpha must be exactly the modes "
        "gl_material enables blending for, or those modes render as dark solids"
    )
    assert RenderMode.OPAQUE.name not in ALPHA_MODULATING_MODES


# ---------------------------------------------------------------------------
# Output range: the invariant that catches overflow, NaN and sign errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODE_NAMES)
def test_output_is_finite_over_the_whole_domain(mode, domain, uniforms):
    out = evaluate(mode, domain, uniforms)
    rows = finite_rows(out)
    assert len(rows) > 0, f"{mode}: every fragment was discarded"
    assert np.isfinite(rows).all(), f"{mode}: produced non-finite values"


@pytest.mark.parametrize("mode", MODE_NAMES)
def test_alpha_is_in_unit_range(mode, domain, uniforms):
    rows = finite_rows(evaluate(mode, domain, uniforms))
    a = rows[:, 3]
    assert a.min() >= 0.0 - 1e-12, f"{mode}: alpha min {a.min()}"
    assert a.max() <= 1.0 + 1e-12, f"{mode}: alpha max {a.max()}"


@pytest.mark.parametrize("mode", MODE_NAMES)
def test_colour_channels_are_never_negative(mode, domain, uniforms):
    """Negative light is unphysical and indicates a sign error."""
    rows = finite_rows(evaluate(mode, domain, uniforms))
    rgb = rows[:, :3]
    assert rgb.min() >= 0.0 - 1e-12, f"{mode}: negative channel {rgb.min()}"


# The 11 modes that clamp their own colour must never exceed 1.0.  SOLID,
# OPAQUE and WIREFRAME deliberately do not clamp: phong_pointlight.frag can
# exceed 1.0 with a bright point light and relies on the fixed-function
# framebuffer clamp, and wireframe.frag passes uColor through unchanged.
SELF_CLAMPING_MODES = [
    m for m in MODE_NAMES if m not in ("SOLID", "OPAQUE", "WIREFRAME")
]


@pytest.mark.parametrize("mode", SELF_CLAMPING_MODES)
def test_self_clamping_modes_stay_within_unit_range(mode, domain, uniforms):
    rows = finite_rows(evaluate(mode, domain, uniforms))
    rgb = rows[:, :3]
    assert rgb.max() <= 1.0 + 1e-9, (
        f"{mode}: channel {rgb.max()} exceeds 1.0 despite an explicit clamp in the shader"
    )


def test_unclamped_modes_are_documented_as_such(domain):
    """SOLID/OPAQUE really can exceed 1.0 — so the claim above is not vacuous."""
    u = Uniforms(has_point_light=1, point_light_intensity=8.0,
                 point_light_pos=vec3(0.0, 0.0, -30.0), point_light_range=200.0)
    out = evaluate("SOLID", domain, u)
    assert out[:, :3].max() > 1.0, (
        "phong_pointlight.frag was expected to overflow with a bright point light; "
        "if it now clamps, move SOLID/OPAQUE into SELF_CLAMPING_MODES"
    )


@pytest.mark.parametrize("mode", MODE_NAMES)
def test_no_mode_is_uniformly_black_or_uniformly_flat(mode, domain, uniforms):
    """A shader returning a constant has lost its inputs — the classic silent bug."""
    rows = finite_rows(evaluate(mode, domain, uniforms))
    rgb = rows[:, :3]
    assert rgb.max() > 0.01, f"{mode}: output is effectively black everywhere"
    # Two modes are flat in RGB by design and must vary elsewhere or not at all:
    #   wireframe.frag passes uColor straight through, with no varying input;
    #   points.frag emits uColor unchanged and puts all its variation in alpha.
    # 1e-12 rather than exact zero: the constant colour is broadcast through
    # float64 arithmetic, so std() lands at ~2e-15 rather than 0.
    if mode == "WIREFRAME":
        assert rgb.std(axis=0).max() < 1e-12, (
            "wireframe.frag is expected to be flat; if it now shades, this test "
            "should assert variation instead"
        )
        return
    if mode == "POINTS":
        assert rgb.std(axis=0).max() < 1e-12, "points.frag should emit uColor unchanged"
        assert rows[:, 3].std() > 1e-6, "points.frag must vary its alpha across the sprite"
        return
    assert rgb.std(axis=0).max() > 1e-6, (
        f"{mode}: output does not vary across the input domain, so the shader is "
        "ignoring its varyings"
    )


# ---------------------------------------------------------------------------
# Alpha semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [m for m in MODE_NAMES if m not in ALPHA_MODULATING_MODES])
def test_non_blending_modes_pass_opacity_through_exactly(mode, domain):
    for opacity in (1.0, 0.5, 0.25):
        u = Uniforms(opacity=opacity)
        rows = finite_rows(evaluate(mode, domain, u))
        assert np.allclose(rows[:, 3], opacity), (
            f"{mode}: alpha is not exactly uOpacity={opacity}; it would then need "
            "to be in _MODE_NEEDS_BLENDING"
        )


@pytest.mark.parametrize("mode", sorted(ALPHA_MODULATING_MODES))
def test_blending_modes_produce_fractional_alpha(mode, domain):
    """These modes' identity lives in alpha: with blending off they go dark."""
    u = Uniforms(opacity=1.0)
    rows = finite_rows(evaluate(mode, domain, u))
    a = rows[:, 3]
    assert a.min() < 0.95, (
        f"{mode}: alpha never drops below 0.95, so it does not modulate alpha and "
        "does not belong in _MODE_NEEDS_BLENDING"
    )
    assert (a > 0.0).any(), f"{mode}: alpha is zero everywhere — nothing would draw"


@pytest.mark.parametrize("mode", sorted(ALPHA_MODULATING_MODES))
def test_blending_modes_scale_alpha_with_opacity(mode, domain):
    """uOpacity must still be a global multiplier for the alpha-modulating modes."""
    full = finite_rows(evaluate(mode, domain, Uniforms(opacity=1.0)))[:, 3]
    half = finite_rows(evaluate(mode, domain, Uniforms(opacity=0.5)))[:, 3]
    # Clamping can break exact proportionality at the top; compare where it cannot.
    safe = full < 0.9
    assert safe.sum() > 10, f"{mode}: too few unclamped samples to test scaling"
    assert np.allclose(half[safe], full[safe] * 0.5, atol=1e-9), (
        f"{mode}: halving uOpacity did not halve alpha"
    )


def test_zero_opacity_makes_every_mode_fully_transparent(domain):
    for mode in MODE_NAMES:
        rows = finite_rows(evaluate(mode, domain, Uniforms(opacity=0.0)))
        assert np.allclose(rows[:, 3], 0.0), (
            f"{mode}: uOpacity=0 must give alpha 0, got max {rows[:, 3].max()}"
        )


# ---------------------------------------------------------------------------
# Lighting monotonicity
# ---------------------------------------------------------------------------

# Modes whose brightness is a monotonic non-decreasing function of dot(N, L)
# at a fixed view angle.  XRAY, HOLOGRAM, BLUEPRINT and ETHEREAL are excluded
# because their appearance is driven by the Fresnel edge term, not by the light;
# PEN_INK, ILLUSTRATION, SEPIA and COLOR_ATLAS are excluded because their
# screen-space hatching is deliberately non-monotonic in shadow depth.
LIGHT_MONOTONIC_MODES = ["SOLID", "OPAQUE", "MEDICAL", "PORCELAIN", "THERMAL"]


@pytest.mark.parametrize("mode", LIGHT_MONOTONIC_MODES)
def test_brightness_is_non_decreasing_in_dot_n_l(mode):
    """Turning a surface toward the light must not make it darker.

    Constructed so only dot(N, L) varies: the view direction is held fixed by
    rotating N in the plane perpendicular to V, which keeps ffEdge(N, V)
    constant and isolates the light term from the silhouette term.
    """
    u = Uniforms()
    light = glsl_normalize(u.light_dir)
    # An orthonormal frame with `light` as one axis.
    tmp = vec3(0.0, 0.0, 1.0)
    if abs(np.dot(tmp, light)) > 0.9:
        tmp = vec3(1.0, 0.0, 0.0)
    b1 = glsl_normalize(np.cross(light, tmp))
    # View direction perpendicular to the rotation plane keeps |dot(N, V)| fixed.
    view_dir = glsl_normalize(np.cross(light, b1))

    angles = np.linspace(np.pi, 0.0, 64)          # dot(N,L) from -1 to +1
    normals = np.cos(angles)[:, None] * light + np.sin(angles)[:, None] * b1
    m = len(angles)
    v = Varyings(
        normal=normals,
        view_pos=np.repeat((-view_dir * 100.0)[None, :], m, axis=0),
        world_pos=np.zeros((m, 3)),
        vertex_color=np.zeros((m, 3)),
        frag_coord=np.repeat(np.array([[8.5, 8.5]]), m, axis=0),
        point_coord=np.repeat(np.array([[0.5, 0.5]]), m, axis=0),
    )
    ndotl = np.cos(angles)
    assert ndotl[0] < -0.99 and ndotl[-1] > 0.99, "fixture does not sweep dot(N,L)"
    edge = ff_edge(ff_normal(v), ff_view_dir(v))
    assert edge.std() < 1e-9, f"view term is not constant across the sweep ({edge.std()})"

    lum = ff_luma(evaluate(mode, v, u)[:, :3])
    d = np.diff(lum)
    assert (d >= -1e-9).all(), (
        f"{mode}: luminance decreases as the surface turns toward the light "
        f"(worst step {d.min():.6g} at dot(N,L)={ndotl[int(np.argmin(d))]:.3f})"
    )
    assert lum[-1] > lum[0] + 0.01, (
        f"{mode}: fully lit ({lum[-1]:.4f}) is not brighter than fully unlit "
        f"({lum[0]:.4f}) — the light term is missing or inverted"
    )


@pytest.mark.parametrize("mode", ["SOLID", "OPAQUE", "MEDICAL", "CARTOON"])
def test_flipping_the_light_flips_which_side_is_lit(mode):
    """Catches a sign error in the light direction that monotonicity alone misses."""
    n = np.array([[0.0, 0.0, 1.0]])
    v = Varyings(
        normal=n, view_pos=np.array([[0.0, 0.0, -100.0]]),
        world_pos=np.zeros((1, 3)), vertex_color=np.zeros((1, 3)),
        frag_coord=np.array([[8.5, 8.5]]), point_coord=np.array([[0.5, 0.5]]),
    )
    lit = ff_luma(evaluate(mode, v, Uniforms(light_dir=vec3(0.0, 0.0, 1.0)))[:, :3])
    unlit = ff_luma(evaluate(mode, v, Uniforms(light_dir=vec3(0.0, 0.0, -1.0)))[:, :3])
    assert lit[0] > unlit[0] + 0.01, (
        f"{mode}: a light along +N is not brighter than one along -N "
        f"({lit[0]:.4f} vs {unlit[0]:.4f})"
    )


# ---------------------------------------------------------------------------
# Silhouette (Fresnel) behaviour
# ---------------------------------------------------------------------------

def test_ff_edge_is_zero_facing_and_one_at_the_silhouette():
    view = vec3(0.0, 0.0, 1.0)
    assert ff_edge(vec3(0.0, 0.0, 1.0), view) == pytest.approx(0.0)
    assert ff_edge(vec3(0.0, 0.0, -1.0), view) == pytest.approx(0.0), (
        "ffEdge uses abs(), so a back-facing normal is also 'facing'"
    )
    assert ff_edge(vec3(1.0, 0.0, 0.0), view) == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["XRAY", "HOLOGRAM", "BLUEPRINT", "ETHEREAL"])
def test_edge_driven_modes_are_brighter_at_the_silhouette(mode):
    """These four exist to highlight silhouettes; verify they actually do."""
    u = Uniforms()
    view_pos = np.array([[0.0, 0.0, -100.0]])
    facing = Varyings(
        normal=np.array([[0.0, 0.0, 1.0]]), view_pos=view_pos,
        world_pos=np.zeros((1, 3)), vertex_color=np.zeros((1, 3)),
        frag_coord=np.array([[8.5, 8.5]]), point_coord=np.array([[0.5, 0.5]]),
    )
    silhouette = Varyings(
        normal=np.array([[1.0, 0.0, 0.0]]), view_pos=view_pos,
        world_pos=np.zeros((1, 3)), vertex_color=np.zeros((1, 3)),
        frag_coord=np.array([[8.5, 8.5]]), point_coord=np.array([[0.5, 0.5]]),
    )
    a_face = evaluate(mode, facing, u)[0, 3]
    a_edge = evaluate(mode, silhouette, u)[0, 3]
    assert a_edge > a_face, (
        f"{mode}: silhouette alpha {a_edge:.4f} is not above facing alpha "
        f"{a_face:.4f} — the Fresnel term is inverted or missing"
    )


# ---------------------------------------------------------------------------
# Clip plane
# ---------------------------------------------------------------------------

def test_clip_distance_is_positive_when_the_cutaway_is_off():
    wp = np.array([[1e6, -1e6, 1e6], [0.0, 0.0, 0.0]])
    d = clip_distance(wp, 0, [1.0, 0.0, 0.0, 0.0])
    assert (d > 0).all(), "with uClipEnabled == 0 nothing may be clipped"
    assert np.allclose(d, 1.0), "default.vert writes the constant 1.0 when disabled"


def test_clip_distance_sign_matches_the_old_fragment_test():
    """The port from `discard` to gl_ClipDistance must be behaviour-preserving.

    The old fragment test discarded when
    ``dot(vWorldPos, uClipPlane.xyz) + uClipPlane.w < 0.0``.  Hardware clipping
    discards when gl_ClipDistance < 0.  So the two agree iff the distance is
    exactly that expression -- boundary included.
    """
    plane = np.array([1.0, 0.0, 0.0, -5.0])   # keep x > 5
    wp = np.array([
        [10.0, 0.0, 0.0],   # inside  -> +5
        [5.0, 0.0, 0.0],    # boundary -> 0, kept by both (test is `< 0`)
        [0.0, 0.0, 0.0],    # outside -> -5
        [-100.0, 3.0, 7.0],
    ])
    d = clip_distance(wp, 1, plane)
    expected = wp @ plane[:3] + plane[3]
    assert np.allclose(d, expected)
    assert d[0] > 0 and d[1] == 0.0 and d[2] < 0 and d[3] < 0
    kept_hw = d >= 0.0
    kept_old = ~(expected < 0.0)
    assert (kept_hw == kept_old).all(), "hardware clipping disagrees at the boundary"


def test_clip_distance_is_linear_so_interpolation_is_exact():
    """gl_ClipDistance is interpolated; a linear function makes that exact."""
    plane = np.array([0.3, -0.7, 0.2, 1.5])
    a, b = np.array([[1.0, 2.0, 3.0]]), np.array([[-4.0, 8.0, 0.5]])
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        mid = a * (1 - t) + b * t
        assert clip_distance(mid, 1, plane)[0] == pytest.approx(
            clip_distance(a, 1, plane)[0] * (1 - t) + clip_distance(b, 1, plane)[0] * t
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_ff_normal_renormalises_interpolated_normals():
    v = Varyings(
        normal=np.array([[0.0, 0.0, 7.3], [1.0, 1.0, 1.0]]),
        view_pos=np.zeros((2, 3)), world_pos=np.zeros((2, 3)),
        vertex_color=np.zeros((2, 3)), frag_coord=np.zeros((2, 2)),
    )
    n = ff_normal(v)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)


def test_ff_luma_uses_rec601_weights():
    assert ff_luma(vec3(1.0, 0.0, 0.0)) == pytest.approx(0.299)
    assert ff_luma(vec3(0.0, 1.0, 0.0)) == pytest.approx(0.587)
    assert ff_luma(vec3(0.0, 0.0, 1.0)) == pytest.approx(0.114)
    assert ff_luma(vec3(1.0, 1.0, 1.0)) == pytest.approx(1.0)


def test_ff_specular_peaks_on_the_mirror_direction_and_is_bounded():
    n = vec3(0.0, 0.0, 1.0)
    light = glsl_normalize(vec3(1.0, 0.0, 1.0))
    mirror = glsl_normalize(vec3(-1.0, 0.0, 1.0))   # reflect(-L, N)
    peak = ff_specular(n, mirror, light, 30.0)
    assert peak == pytest.approx(1.0, abs=1e-9)
    off = ff_specular(n, glsl_normalize(vec3(1.0, 0.0, 1.0)), light, 30.0)
    assert 0.0 <= off < peak
    # Higher shininess must tighten the lobe, never widen it.
    a = ff_specular(n, glsl_normalize(mirror + vec3(0.2, 0.0, 0.0)), light, 10.0)
    b = ff_specular(n, glsl_normalize(mirror + vec3(0.2, 0.0, 0.0)), light, 100.0)
    assert b < a


def test_smoothstep_handles_reversed_edges():
    """points.frag calls smoothstep(1.0, 0.8, dist), with e0 > e1."""
    assert glsl_smoothstep(1.0, 0.8, 1.0) == pytest.approx(0.0)
    assert glsl_smoothstep(1.0, 0.8, 0.8) == pytest.approx(1.0)
    assert glsl_smoothstep(1.0, 0.8, 0.9) == pytest.approx(0.5)
    assert 0.0 <= glsl_smoothstep(1.0, 0.8, 1.5) <= 1.0


def test_points_discards_outside_the_unit_disc():
    """points.frag's sprite mask: dot(coord, coord) > 1 must discard."""
    pc = np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0], [0.5, 0.0]])
    v = Varyings(
        normal=np.tile([0.0, 0.0, 1.0], (4, 1)), view_pos=np.zeros((4, 3)),
        world_pos=np.zeros((4, 3)), vertex_color=np.zeros((4, 3)),
        frag_coord=np.zeros((4, 2)), point_coord=pc,
    )
    out = evaluate("POINTS", v, Uniforms())
    assert not np.isnan(out[0]).any(), "sprite centre must not be discarded"
    assert np.isnan(out[1]).all(), "corner (0,0) -> dist 2 must be discarded"
    assert np.isnan(out[2]).all(), "corner (1,1) -> dist 2 must be discarded"
    assert out[0, 3] == pytest.approx(1.0), "centre should be fully opaque"
    assert not np.isnan(out[3]).any() and out[3, 3] < 1.0, "edge should be soft"


def test_vertex_colour_override_is_honoured_where_supported():
    """uUseVertexColor must actually switch the base colour."""
    vc = np.array([[0.9, 0.1, 0.2]])
    v = Varyings(
        normal=np.array([[0.0, 0.4, 1.0]]), view_pos=np.array([[0.0, 0.0, -100.0]]),
        world_pos=np.zeros((1, 3)), vertex_color=vc,
        frag_coord=np.array([[8.5, 8.5]]), point_coord=np.array([[0.5, 0.5]]),
    )
    # POINTS reads uColor only (points.frag has no vVertexColor input at all).
    for mode in ("SOLID", "MEDICAL", "CARTOON", "THERMAL", "WIREFRAME"):
        off = evaluate(mode, v, Uniforms(use_vertex_color=0, color=vec3(0.1, 0.1, 0.9)))
        on = evaluate(mode, v, Uniforms(use_vertex_color=1, color=vec3(0.1, 0.1, 0.9)))
        assert not np.allclose(off[:, :3], on[:, :3]), (
            f"{mode}: uUseVertexColor had no effect on the output"
        )


def test_points_ignores_vertex_colour_by_design():
    v = Varyings(
        normal=np.array([[0.0, 0.0, 1.0]]), view_pos=np.array([[0.0, 0.0, -100.0]]),
        world_pos=np.zeros((1, 3)), vertex_color=np.array([[0.9, 0.1, 0.2]]),
        frag_coord=np.array([[8.5, 8.5]]), point_coord=np.array([[0.5, 0.5]]),
    )
    off = evaluate("POINTS", v, Uniforms(use_vertex_color=0))
    on = evaluate("POINTS", v, Uniforms(use_vertex_color=1))
    assert np.allclose(off, on), (
        "points.frag declares no vVertexColor input, so uUseVertexColor must not "
        "change its output"
    )


def test_solid_and_opaque_are_the_same_shader():
    """OPAQUE differs from SOLID in GL state only, never in shader code."""
    assert MODE_FRAGMENTS["SOLID"] == MODE_FRAGMENTS["OPAQUE"]


def test_hologram_and_blueprint_read_world_position():
    """Both are documented as needing uModelMatrix; verify the dependency is real."""
    base = dict(
        normal=np.array([[0.0, 0.5, 1.0]]), view_pos=np.array([[0.0, 0.0, -100.0]]),
        vertex_color=np.zeros((1, 3)), frag_coord=np.array([[8.5, 8.5]]),
        point_coord=np.array([[0.5, 0.5]]),
    )
    for mode in ("HOLOGRAM", "BLUEPRINT"):
        a = evaluate(mode, Varyings(world_pos=np.array([[0.0, 0.0, 0.0]]), **base), Uniforms())
        b = evaluate(mode, Varyings(world_pos=np.array([[1.7, 2.3, 3.1]]), **base), Uniforms())
        assert not np.allclose(a, b), (
            f"{mode}: output does not depend on vWorldPos, so uModelMatrix would be dead"
        )


def test_modes_that_should_not_read_world_position_do_not():
    """Guards against a mode silently acquiring a uModelMatrix dependency."""
    base = dict(
        normal=np.array([[0.0, 0.5, 1.0]]), view_pos=np.array([[0.0, 0.0, -100.0]]),
        vertex_color=np.zeros((1, 3)), frag_coord=np.array([[8.5, 8.5]]),
        point_coord=np.array([[0.5, 0.5]]),
    )
    for mode in set(MODE_NAMES) - {"HOLOGRAM", "BLUEPRINT"}:
        a = evaluate(mode, Varyings(world_pos=np.array([[0.0, 0.0, 0.0]]), **base), Uniforms())
        b = evaluate(mode, Varyings(world_pos=np.array([[91.0, -55.0, 33.0]]), **base),
                     Uniforms())
        assert np.allclose(a, b, equal_nan=True), (
            f"{mode}: output depends on vWorldPos but is not listed as needing it"
        )


def test_screen_space_modes_depend_on_frag_coord():
    """The hatching/stipple/scanline modes must actually use gl_FragCoord."""
    # The normal points AWAY from the default light, so dot(N, L) < 0 and every
    # mode's shadow term is deep.  With a lit normal the hatching and stipple
    # are switched off by their own `if (shadow > ...)` guards, and the test
    # would pass or fail for the wrong reason.
    base = dict(
        normal=np.array([[-0.3, 0.6, -0.7]]), view_pos=np.array([[0.0, 0.0, -100.0]]),
        world_pos=np.zeros((1, 3)), vertex_color=np.zeros((1, 3)),
        point_coord=np.array([[0.5, 0.5]]),
    )
    for mode in ("ILLUSTRATION", "SEPIA", "COLOR_ATLAS", "PEN_INK", "THERMAL", "HOLOGRAM"):
        found = False
        for coord in ([[0.5, 0.5]], [[1.5, 2.5]], [[3.5, 0.5]], [[2.5, 5.5]], [[7.5, 4.5]]):
            a = evaluate(mode, Varyings(frag_coord=np.array([[0.5, 0.5]]), **base), Uniforms())
            b = evaluate(mode, Varyings(frag_coord=np.array(coord), **base), Uniforms())
            if not np.allclose(a, b):
                found = True
                break
        assert found, f"{mode}: output never changed with gl_FragCoord"
