"""Capture the *effective* GL state at every draw call of a frame.

Used to prove that the hot-path optimisations are behaviour-preserving.  The
optimisations deliberately change the GL *call stream* (that is the whole
point), so an A/B on call counts proves nothing about correctness.  What must
be identical is the state the GPU actually sees at the moment each mesh is
drawn: the uniform values in effect for its program, and the blend / depth /
cull / polygon-mode state.

This module instruments the real renderer -- it does not reimplement it:

* every ``ShaderProgram.set_uniform_*`` is wrapped so the (program, name,
  value) it uploads is accumulated into a per-program dict, exactly as a
  driver would;
* the GL enable/disable/depthMask/polygonMode/cullFace calls are read out of
  the ``glrec`` recorder's argument log;
* ``_draw_mesh`` is wrapped so each snapshot is tagged with the mesh name.

Run it once on the unpatched tree and once on the patched tree and diff the
JSON:

    python tests/rendering/equiv_capture.py before.json
    ... apply patches ...
    python tests/rendering/equiv_capture.py after.json
    python tests/rendering/equiv_capture.py --diff before.json after.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402

from tests.support import glrec  # noqa: E402

REC = glrec.install(record_args=True)

from faceforge.core.material import Material, RenderMode  # noqa: E402
from faceforge.core.mesh import BufferGeometry, MeshInstance  # noqa: E402
from faceforge.core.scene_graph import Scene, SceneNode  # noqa: E402
from faceforge.rendering.camera import Camera  # noqa: E402
from faceforge.rendering.lights import LightSetup, PointLight  # noqa: E402
from faceforge.rendering.renderer import GLRenderer  # noqa: E402
from faceforge.rendering.shader_program import ShaderProgram  # noqa: E402

# GL state calls whose arguments we replay into a shadow state dict.
_STATE_CALLS = {
    "glEnable", "glDisable", "glDepthMask", "glPolygonMode", "glCullFace",
    "glBlendFunc", "glDepthFunc",
}


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return [round(float(x), 7) for x in np.asarray(v).ravel()]
    if isinstance(v, (list, tuple)):
        return [round(float(x), 7) for x in v]
    if isinstance(v, (int, np.integer)):
        return int(v)
    return round(float(v), 7)


class Capture:
    """Wrap a renderer so every draw call records the state in effect."""

    def __init__(self, renderer: GLRenderer) -> None:
        self.r = renderer
        self.snapshots: list[dict] = []
        self._uniforms: dict[int, dict] = {}
        self._install()

    # -- instrumentation ------------------------------------------------

    def _install(self) -> None:
        cap = self

        for meth in ("set_uniform_mat4", "set_uniform_mat3", "set_uniform_vec3",
                     "set_uniform_vec4", "set_uniform_float", "set_uniform_int"):
            orig = getattr(ShaderProgram, meth)

            def make(orig=orig):
                def wrapper(self, name, value):
                    # A driver ignores uploads to absent uniforms; mirror that
                    # so the snapshot reflects what the GPU really holds.
                    if self.get_uniform_location(name) >= 0:
                        cap._uniforms.setdefault(
                            self.program_id, {})[name] = _jsonable(value)
                    return orig(self, name, value)
                return wrapper

            setattr(ShaderProgram, meth, make())

        orig_draw = self.r._draw_mesh

        def draw(mesh, world, view, proj, lights):
            before = REC.counts["glDrawArrays"] + REC.counts["glDrawElements"]
            out = orig_draw(mesh, world, view, proj, lights)
            after = REC.counts["glDrawArrays"] + REC.counts["glDrawElements"]
            if after > before:
                cap.snapshots.append({
                    "mesh": mesh.name,
                    "mode": mesh.material.render_mode.name,
                    "program": cap._cur_program(),
                    "uniforms": dict(cap._uniforms.get(cap._cur_program(), {})),
                    "state": cap._gl_state(),
                })
            return out

        self.r._draw_mesh = draw

    # -- shadow GL state ------------------------------------------------

    def _cur_program(self) -> int:
        prog = 0
        for name, args in REC.calls:
            if name == "glUseProgram":
                prog = args[0]
        return prog

    def _gl_state(self) -> dict:
        state: dict = {}
        for name, args in REC.calls:
            if name not in _STATE_CALLS:
                continue
            if name in ("glEnable", "glDisable"):
                state[f"cap{args[0]}"] = (name == "glEnable")
            elif name == "glDepthMask":
                state["depthMask"] = bool(args[0])
            elif name == "glPolygonMode":
                state["polygonMode"] = int(args[1])
            elif name == "glCullFace":
                state["cullFace"] = int(args[0])
            elif name == "glBlendFunc":
                state["blendFunc"] = [int(args[0]), int(args[1])]
            elif name == "glDepthFunc":
                state["depthFunc"] = int(args[0])
        return state


def _scene(n: int, mode: RenderMode, *, opacity=1.0, transparent=False,
           spread=60.0, scale=None):
    rng = np.random.default_rng(7)
    scene = Scene()
    for i in range(n):
        v = rng.normal(0.0, 30.0, size=90).astype(np.float32)
        geom = BufferGeometry(
            positions=v,
            normals=np.tile([0.0, 0.0, 1.0], 30).astype(np.float32),
            vertex_count=30,
        )
        mat = Material(color=(0.3 + 0.001 * i, 0.5, 0.7), opacity=opacity,
                       shininess=15.0 + i, render_mode=mode,
                       transparent=transparent)
        mesh = MeshInstance(name=f"m{i}", geometry=geom, material=mat)
        node = SceneNode(name=f"n{i}")
        node.mesh = mesh
        node.set_position(*(rng.normal(0.0, spread, 3).tolist()))
        if scale is not None:
            node.set_scale(scale, scale, scale)
        scene.add(node)
    scene.update()
    return scene


def _run(label: str, scene, *, clip=None, point_light=False,
         scene_transform=False) -> dict:
    cam = Camera()
    lights = LightSetup()
    if point_light:
        lights.point_light = PointLight(
            position=np.array([10.0, 20.0, 30.0]), enabled=True)
    r = GLRenderer()
    r.init_gl()
    r.resize(1600, 1000)
    if clip is not None:
        r.set_clip_plane(clip[:3], clip[3])
    if scene_transform:
        t = np.eye(4, dtype=np.float64)
        t[:3, 3] = [1.0, 2.0, 3.0]
        r.scene_transform = t
    r.render(scene, cam, lights)          # upload pass
    cap = Capture(r)
    REC.reset()
    r.render(scene, cam, lights)          # steady-state pass, captured
    r.destroy()
    return {"label": label, "draws": cap.snapshots}


def capture_all() -> dict:
    """The scenarios that between them exercise every uniform the app uses."""
    out = {}
    out["solid"] = _run("solid", _scene(12, RenderMode.SOLID))
    out["solid_clip"] = _run("solid_clip", _scene(12, RenderMode.SOLID),
                             clip=(1.0, 0.0, 0.0, -5.0))
    out["solid_pointlight"] = _run("solid_pointlight",
                                   _scene(12, RenderMode.SOLID),
                                   point_light=True)
    out["solid_scenexform"] = _run("solid_scenexform",
                                   _scene(12, RenderMode.SOLID),
                                   scene_transform=True)
    out["hologram"] = _run("hologram", _scene(12, RenderMode.HOLOGRAM))
    out["blueprint"] = _run("blueprint", _scene(12, RenderMode.BLUEPRINT))
    out["wireframe"] = _run("wireframe", _scene(8, RenderMode.WIREFRAME))
    out["points"] = _run("points", _scene(8, RenderMode.POINTS))
    out["opaque"] = _run("opaque", _scene(8, RenderMode.OPAQUE, opacity=0.5))
    out["xray_transparent"] = _run(
        "xray_transparent", _scene(8, RenderMode.XRAY, opacity=0.4,
                                   transparent=True))
    out["uniform_scale"] = _run("uniform_scale",
                                _scene(8, RenderMode.SOLID, scale=2.5))
    return out


# ----------------------------------------------------------------------
# Diff
# ----------------------------------------------------------------------

# Uniforms whose *upload* the patches legitimately remove.  Each is justified
# separately (see the track report), so the diff reports them but does not
# count them as a behavioural difference.
_EXPECTED_ABSENT = {"uNormalMatrix", "uModelMatrix", "uPointSize"}

# Modes whose blend state and draw pass CHANGE on purpose: their fragment
# shader computes a fractional alpha, and they only blended before because the
# STL loader happened to default every structure to opacity 0.7 /
# transparent=True.  Fixing that default to 1.0 would have rendered them as
# dark solids, so gl_material._MODE_NEEDS_BLENDING now asks for blending on the
# mode's own behalf -- which flips GL_BLEND on, glDepthMask off, and moves them
# into the depth-sorted pass.  Everything else about them must stay identical.
_BLEND_POLICY_MODES = {"XRAY", "HOLOGRAM", "BLUEPRINT", "ETHEREAL", "POINTS"}
_BLEND_STATE_KEYS = {"depthMask", "blendFunc"}


def _state_diff_is_blend_policy(mode: str, b: dict, a: dict) -> bool:
    """True if the only state change is 'blending got switched on'."""
    if mode not in _BLEND_POLICY_MODES:
        return False
    if not (b.get("depthMask") is True and a.get("depthMask") is False):
        return False
    changed = {k for k in set(b) | set(a) if b.get(k) != a.get(k)}
    # the GL_BLEND capability id is opaque (glrec mints it), so allow any
    # single capability that went False -> True alongside depthMask/blendFunc
    caps = {k for k in changed if k.startswith("cap")}
    if len(caps) > 1:
        return False
    for c in caps:
        if not (b.get(c) is False and a.get(c) is True):
            return False
    return not (changed - caps - _BLEND_STATE_KEYS)


def diff(before: dict, after: dict) -> int:
    problems = 0
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            print(f"[{key}] scenario missing on one side")
            problems += 1
            continue
        bd, ad = b["draws"], a["draws"]
        if len(bd) != len(ad):
            print(f"[{key}] draw count {len(bd)} -> {len(ad)}")
            problems += 1
            continue
        mode = bd[0]["mode"] if bd else ""
        policy_mode = mode in _BLEND_POLICY_MODES
        # Draw ORDER is compared separately; match by mesh name so that a
        # deliberate re-sort is not reported as a value difference.
        order_changed = [d["mesh"] for d in bd] != [d["mesh"] for d in ad]
        if order_changed and not policy_mode:
            print(f"[{key}] draw ORDER changed: "
                  f"{[d['mesh'] for d in bd]} -> {[d['mesh'] for d in ad]}")
            problems += 1
        bmap = {d["mesh"]: d for d in bd}
        amap = {d["mesh"]: d for d in ad}
        n_state = n_val = n_absent = n_policy = 0
        for name, bs in bmap.items():
            as_ = amap[name]
            for u, bv in bs["uniforms"].items():
                if u not in as_["uniforms"]:
                    if u not in _EXPECTED_ABSENT:
                        print(f"[{key}/{name}] uniform {u} no longer uploaded")
                        problems += 1
                    n_absent += 1
                elif as_["uniforms"][u] != bv:
                    print(f"[{key}/{name}] uniform {u}: {bv} -> "
                          f"{as_['uniforms'][u]}")
                    problems += 1
                    n_val += 1
            # Blend/depth/cull/polygon state must match exactly, unless this is
            # the deliberate mode-blending policy change.
            if bs["state"] != as_["state"]:
                if _state_diff_is_blend_policy(mode, bs["state"], as_["state"]):
                    n_policy += 1
                else:
                    print(f"[{key}/{name}] GL state {bs['state']} -> "
                          f"{as_['state']}")
                    problems += 1
                    n_state += 1
        note = ""
        if n_policy:
            note = (f"; {n_policy} draws newly blended by the {mode} "
                    "blending policy (intended)")
        if order_changed and policy_mode:
            note += "; moved into the depth-sorted pass (intended)"
        print(f"[{key}] {len(bd)} draws, "
              f"{len(bmap[bd[0]['mesh']]['uniforms'])} uniforms/draw checked; "
              f"value diffs={n_val} state diffs={n_state} "
              f"expected-absent={n_absent}{note}")
    print("\nEQUIVALENT" if problems == 0 else f"\n{problems} DIFFERENCES")
    return problems


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--diff":
        b = json.load(open(sys.argv[2]))
        a = json.load(open(sys.argv[3]))
        sys.exit(1 if diff(b, a) else 0)
    data = capture_all()
    out_path = sys.argv[1] if len(sys.argv) > 1 else "capture.json"
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    n = sum(len(v["draws"]) for v in data.values())
    print(f"{len(data)} scenarios, {n} draw snapshots -> {out_path}")
