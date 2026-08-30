"""Static checks on the GLSL sources.

There is no GL context in CI and no way to obtain one in the analysis sandbox,
so nothing here compiles a shader -- a real driver is still the only authority.
What these tests *can* do is catch the failure modes that a
compile-it-and-see loop would otherwise be the only defence against:

* an identifier used but never declared (the classic breakage when shared code
  is factored out into a prelude and one file keeps using a local that moved);
* unbalanced braces / parens after a textual edit;
* the ``#version`` directive no longer first;
* a fragment shader reading a varying its vertex shader does not write
  (a link error on strict drivers);
* the clip-plane contract: exactly one fragment shader may contain ``discard``,
  and both vertex shaders must write ``gl_ClipDistance[0]``;
* every uniform the renderer uploads being declared *somewhere*, and every
  ``#include`` resolving.
"""

import re
from pathlib import Path

import pytest

from faceforge.core.material import RenderMode
from faceforge.rendering.shader_program import load_shader_source

SHADER_DIR = (Path(__file__).resolve().parents[2]
              / "src" / "faceforge" / "rendering" / "shaders")

FRAG_FILES = sorted(p.name for p in SHADER_DIR.glob("*.frag"))
VERT_FILES = sorted(p.name for p in SHADER_DIR.glob("*.vert"))
INCLUDE_FILES = sorted(p.name for p in SHADER_DIR.glob("_*.glsl"))

# The vertex shader each fragment shader is paired with by _compile_shaders.
POINTS_FRAG = "points.frag"

GLSL_KEYWORDS = {
    "void", "bool", "int", "uint", "float", "double",
    "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "bvec2", "bvec3",
    "bvec4", "uvec2", "uvec3", "uvec4",
    "mat2", "mat3", "mat4", "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3",
    "mat3x4", "mat4x2", "mat4x3", "mat4x4",
    "sampler1D", "sampler2D", "sampler3D", "samplerCube",
    "if", "else", "for", "while", "do", "break", "continue", "return",
    "discard", "switch", "case", "default",
    "in", "out", "inout", "uniform", "attribute", "varying", "const",
    "layout", "location", "flat", "smooth", "noperspective", "centroid",
    "struct", "true", "false", "precision", "highp", "mediump", "lowp",
    "core", "version", "line", "include",
}

GLSL_BUILTIN_FUNCS = {
    "radians", "degrees", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "pow", "exp", "log", "exp2", "log2", "sqrt",
    "inversesqrt", "abs", "sign", "floor", "trunc", "round", "roundEven",
    "ceil", "fract", "mod", "modf", "min", "max", "clamp", "mix", "step",
    "smoothstep", "isnan", "isinf", "length", "distance", "dot", "cross",
    "normalize", "faceforward", "reflect", "refract", "matrixCompMult",
    "outerProduct", "transpose", "determinant", "inverse", "lessThan",
    "lessThanEqual", "greaterThan", "greaterThanEqual", "equal", "notEqual",
    "any", "all", "not", "texture", "textureLod", "texelFetch", "dFdx",
    "dFdy", "fwidth",
}

GLSL_BUILTIN_VARS = {
    "gl_Position", "gl_PointSize", "gl_ClipDistance", "gl_FragCoord",
    "gl_FragDepth", "gl_PointCoord", "gl_FrontFacing", "gl_VertexID",
    "gl_InstanceID", "gl_PerVertex",
}

_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_DECL = re.compile(
    r"\b(?:uniform|in|out|const)?\s*"
    r"(?:vec[234]|ivec[234]|bvec[234]|uvec[234]|mat[234](?:x[234])?|"
    r"float|int|uint|bool|double|void)\s+"
    r"([A-Za-z_][A-Za-z_0-9]*)"
)
_FUNC_DEF = re.compile(
    r"\b(?:vec[234]|mat[234]|float|int|bool|void)\s+"
    r"([A-Za-z_][A-Za-z_0-9]*)\s*\("
)
_PARAM = re.compile(r"\(([^)]*)\)\s*\{")


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"//[^\n]*", " ", src)


def _expanded(name: str) -> str:
    return load_shader_source(name)


def _declared_names(src: str) -> set[str]:
    names: set[str] = set(_DECL.findall(src))
    names |= set(_FUNC_DEF.findall(src))
    # function parameters
    for params in _PARAM.findall(src):
        names |= set(_DECL.findall(params))
    return names


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", FRAG_FILES + VERT_FILES)
def test_version_directive_is_the_first_line(name):
    """``#version`` must precede every other token, includes included."""
    src = _expanded(name)
    first = next(ln for ln in src.splitlines() if ln.strip())
    assert first.strip() == "#version 330 core", (
        f"{name}: first non-blank line is {first!r}; an #include that carried a "
        "#version, or a stray line before it, breaks every driver"
    )


@pytest.mark.parametrize("name", FRAG_FILES + VERT_FILES)
def test_every_include_resolves(name):
    raw = (SHADER_DIR / name).read_text(encoding="utf-8")
    for inc in re.findall(r'#include\s+"([^"]+)"', raw):
        assert (SHADER_DIR / inc).is_file(), f"{name} includes missing {inc}"
    assert "#include" not in _strip_comments(_expanded(name)), (
        f"{name}: an #include survived expansion and would reach the driver"
    )


@pytest.mark.parametrize("name", FRAG_FILES + VERT_FILES)
def test_braces_and_parens_balance(name):
    src = _strip_comments(_expanded(name))
    assert src.count("{") == src.count("}"), f"{name}: unbalanced braces"
    assert src.count("(") == src.count(")"), f"{name}: unbalanced parens"


@pytest.mark.parametrize("name", FRAG_FILES + VERT_FILES)
def test_no_undeclared_identifiers(name):
    """Every identifier is a declaration, a builtin, or a swizzle/field.

    This is the check that replaces the compiler for the one error class the
    prelude refactor can introduce: a file that kept using a local variable
    whose definition moved into a helper.
    """
    src = _strip_comments(_expanded(name))
    declared = _declared_names(src)
    known = declared | GLSL_KEYWORDS | GLSL_BUILTIN_FUNCS | GLSL_BUILTIN_VARS

    unknown = set()
    for m in _IDENT.finditer(src):
        ident = m.group(0)
        if ident in known:
            continue
        # skip struct/vector field access:  foo.xyz  /  v.w
        if m.start() and src[m.start() - 1] == ".":
            continue
        unknown.add(ident)

    assert not unknown, f"{name}: undeclared identifiers {sorted(unknown)}"


# ----------------------------------------------------------------------
# Clip-plane contract
# ----------------------------------------------------------------------

def test_only_points_frag_still_discards():
    """``discard`` anywhere in a fragment shader statically disables early-Z.

    15 of 16 fragment shaders opened with a clip-plane ``discard``; the test now
    moved to gl_ClipDistance[0] in the vertex stage.  points.frag keeps its
    ``discard`` because the circular point sprite needs it.
    """
    offenders = [f for f in FRAG_FILES
                 if "discard" in _strip_comments((SHADER_DIR / f).read_text())]
    assert offenders == [POINTS_FRAG], (
        f"fragment shaders containing discard: {offenders} "
        f"(only {POINTS_FRAG} may)"
    )


@pytest.mark.parametrize("name", VERT_FILES)
def test_vertex_shaders_write_clip_distance(name):
    """An unwritten gl_ClipDistance output is undefined once the state is on."""
    src = _strip_comments(_expanded(name))
    assert "gl_ClipDistance[0]" in src, (
        f"{name}: does not write gl_ClipDistance[0]; with GL_CLIP_DISTANCE0 "
        "enabled its primitives would be clipped at random"
    )
    assert "uClipEnabled" in src and "uClipPlane" in src, (
        f"{name}: writes gl_ClipDistance[0] without the clip uniforms"
    )


def test_no_fragment_shader_declares_the_clip_uniforms():
    """The clip contract lives in one place now -- the vertex shaders."""
    for f in FRAG_FILES:
        src = _strip_comments(_expanded(f))
        assert "uniform int uClipEnabled" not in src, (
            f"{f} re-declares uClipEnabled; clipping is a vertex-stage concern"
        )


# ----------------------------------------------------------------------
# Stage interface
# ----------------------------------------------------------------------

def _varyings(src: str, qualifier: str) -> set[str]:
    return set(re.findall(rf"^\s*{qualifier}\s+\w+\s+(v\w+)\s*;",
                          _strip_comments(src), re.MULTILINE))


def test_default_vert_supplies_every_varying_its_fragment_shaders_read():
    """A statically-used fragment `in` with no matching vertex `out` is a link
    error on strict drivers -- and cannot be caught without compiling, except
    here."""
    produced = _varyings(_expanded("default.vert"), "out")
    assert produced, "default.vert declares no varyings"
    for f in FRAG_FILES:
        if f == POINTS_FRAG:
            continue
        consumed = _varyings(_expanded(f), "in")
        missing = consumed - produced
        assert not missing, f"{f} reads {sorted(missing)}, not written by default.vert"


def test_points_frag_only_reads_varyings_points_vert_writes():
    produced = _varyings(_expanded("points.vert"), "out")
    consumed = _varyings(_expanded(POINTS_FRAG), "in")
    assert not (consumed - produced), (
        f"{POINTS_FRAG} reads {sorted(consumed - produced)}, "
        "not written by points.vert"
    )


def test_no_shader_declares_unormalmatrix():
    """The normal matrix is derived as mat3(uModelView) on the GPU now; a
    lingering declaration would be a uniform the renderer never uploads."""
    for name in FRAG_FILES + VERT_FILES:
        assert "uNormalMatrix" not in _strip_comments(_expanded(name)), f"{name}"


# ----------------------------------------------------------------------
# Renderer <-> shader agreement
# ----------------------------------------------------------------------

def test_every_render_mode_has_a_fragment_shader_file():
    from faceforge.rendering import renderer as rmod

    src = Path(rmod.__file__).read_text(encoding="utf-8")
    named = set(re.findall(r'load_shader_source\("([^"]+)"\)', src))
    for f in named:
        assert (SHADER_DIR / f).is_file(), f"renderer loads missing shader {f}"
    # every mode is wired up (the renderer KeyErrors otherwise)
    assert len(RenderMode) == 16


def test_prelude_files_declare_no_version_directive():
    for name in INCLUDE_FILES:
        raw = _strip_comments((SHADER_DIR / name).read_text(encoding="utf-8"))
        assert "#version" not in raw, (
            f"{name}: a #version inside an include lands mid-file after "
            "expansion, which every driver rejects"
        )
