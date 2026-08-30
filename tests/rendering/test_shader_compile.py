"""Compile every shader with a real GLSL compiler.

`test_shader_static.py` checks shader source by inspection: brace balance,
`#version` placement, undeclared-identifier heuristics, varying interface
matching.  Those checks exist because this project's CI (and the sandbox the
2026-08 review ran in) cannot create an OpenGL context, so the driver's own
compiler is unavailable.

Static checks are not a compiler.  `glslangValidator` (Khronos glslang) is a
standalone CPU GLSL front end that needs no GPU, no context and no window
server, so it CAN run in those environments and is a genuine compile.  This
module uses it to compile every shader and link every vertex/fragment pair
that `GLRenderer` actually builds.

Verified to catch the failure mode this project cares about: deleting a
`uniform` declaration that is still referenced — i.e. a shared include that
stopped providing something — is reported as "undeclared identifier".

Install with `conda install -c conda-forge glslang`.  The module skips when
the binary is absent so it never blocks a contributor who has not installed
it, but it should be present in CI.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from faceforge.rendering.shader_program import _SHADER_DIR as SHADER_DIR
from faceforge.rendering.shader_program import load_shader_source

GLSLANG = shutil.which("glslangValidator")

pytestmark = pytest.mark.skipif(
    GLSLANG is None,
    reason="glslangValidator not installed (conda install -c conda-forge glslang)",
)


def _vertex_for(frag_name: str) -> str:
    """The vertex shader GLRenderer pairs with *frag_name*.

    Point-sprite modes use points.vert (it writes gl_PointSize and a
    view-space normal); everything else uses default.vert.
    """
    return "points.vert" if frag_name.startswith("points") else "default.vert"


def _write_expanded(tmp_path, name: str):
    """Expand #include the way the app does, then write for the compiler."""
    path = tmp_path / name
    path.write_text(load_shader_source(name), encoding="utf-8")
    return path


def _compile(*paths) -> subprocess.CompletedProcess:
    return subprocess.run(
        [GLSLANG, *(str(p) for p in paths)],
        capture_output=True, text=True, check=False,
    )


def _shader_names(suffix: str) -> list[str]:
    return sorted(p.name for p in SHADER_DIR.glob(f"*{suffix}"))


@pytest.mark.parametrize("name", _shader_names(".vert") + _shader_names(".frag"))
def test_shader_compiles(tmp_path, name):
    """Every shader compiles standalone, includes expanded."""
    result = _compile(_write_expanded(tmp_path, name))
    assert result.returncode == 0, (
        f"{name} failed to compile:\n{result.stdout}{result.stderr}"
    )


@pytest.mark.parametrize("frag", _shader_names(".frag"))
def test_program_links(tmp_path, frag):
    """Each vertex+fragment pair links, so the varying interfaces agree."""
    vert = _vertex_for(frag)
    result = _compile(
        "-l",
        _write_expanded(tmp_path, vert),
        _write_expanded(tmp_path, frag),
    )
    assert result.returncode == 0, (
        f"{vert} + {frag} failed to link:\n{result.stdout}{result.stderr}"
    )


def test_compiler_rejects_a_missing_declaration(tmp_path):
    """Negative control: prove these tests can actually fail.

    Simulates a shared include that stopped providing a uniform. Without this
    control, a validator that silently passed everything would look identical
    to a clean tree.
    """
    name = "xray.frag"
    source = load_shader_source(name)

    decl = next(
        (d for d in ("uniform float uOpacity;",
                     "uniform vec3 uColor;",
                     "uniform int uClipEnabled;") if d in source),
        None,
    )
    assert decl is not None, (
        f"{name} declares none of the expected uniforms; update this control"
    )

    broken = tmp_path / "broken.frag"
    broken.write_text(source.replace(decl, "", 1), encoding="utf-8")

    result = _compile(broken)
    assert result.returncode != 0, (
        f"removing {decl!r} from {name} still compiled — these tests cannot fail"
    )
    assert "undeclared identifier" in (result.stdout + result.stderr).lower()
