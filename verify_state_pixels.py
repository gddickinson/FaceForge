"""Prove that applying a saved SceneState reproduces the SAME PIXELS.

Round-trip equality of a JSON file proves the file is faithful to itself.  It
does not prove the file is faithful to a *render*.  This script closes that gap
using the repo's own headless GL path:

  A. Build the fixed skull scene, perturb camera, lights, materials, render
     modes and renderer globals, capture a SceneState, save it, render -> A.
  B. Build the scene again from scratch with a fresh renderer and default
     camera/lights, load the state file, apply it, render -> B.
  C. Load the state file again, change ONE float, apply, render -> C.

The claim is A == B bit-for-bit (a state file reproduces its render) AND
A != C (the comparison is capable of detecting a change, so the first result
is not vacuous).

Read-only with respect to the repo: it imports tools.glcontext and
tools.capture_golden, and writes nothing outside the output directory given on
the command line.

The context reached here is Apple's software rasteriser.  It is correct for
pixels and useless for timing; no frame time from this script is a renderer
performance number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def build_scene(count: int, stl_dir: Path):
    """The first *count* meshes of tools.capture_golden.FIXED_MESHES."""
    from faceforge.core.material import Material
    from faceforge.core.mesh import MeshInstance
    from faceforge.core.scene_graph import Scene, SceneNode
    from faceforge.loaders.stl_batch_loader import load_fma_labels
    from faceforge.loaders.stl_parser import load_stl_file
    from tools.capture_golden import FIXED_MESHES

    fma = load_fma_labels()
    scene = Scene()
    group = SceneNode(name="skullGroup")
    scene.add(group)
    meshes = []
    for fma_id, label in FIXED_MESHES[:count]:
        geom = load_stl_file(stl_dir / f"{fma_id}.stl")
        entry = fma.get(fma_id, {})
        mi = MeshInstance(
            name=label,
            geometry=geom,
            material=Material(color=(0.82, 0.76, 0.68), opacity=1.0),
            source_id=fma_id,
            ontology_id=(f"FMA:{entry['fma_id']}" if entry.get("fma_id") else ""),
            preferred_label=entry.get("preferred_label", ""),
        )
        node = SceneNode(name=label)
        node.mesh = mi
        group.add(node)
        meshes.append(mi)
    return scene, meshes


def perturb(scene, camera, lights, renderer, width, height):
    """Move every knob the state format claims to capture off its default."""
    import numpy as np

    from faceforge.core.material import RenderMode
    from faceforge.core.math_utils import vec3
    from faceforge.core.scene_state import mesh_paths
    from faceforge.rendering.lights import PointLight

    entries = sorted(mesh_paths(scene))
    pos = np.concatenate([n.mesh.positions.reshape(-1, 3) for _, n in entries])
    centroid = pos.mean(axis=0).astype(np.float64)
    radius = float(np.linalg.norm(pos - centroid, axis=1).max())
    d = np.array([-0.62, -0.68, 0.39])
    eye = centroid + (d / np.linalg.norm(d)) * radius * 2.9

    camera.fov = 37.123456789012345
    camera.near = 0.30000000000000004
    camera.far = 1234.5678901234567
    camera.look_at(vec3(*eye), vec3(*centroid), vec3(0.0, 0.0, 1.0))
    camera.set_aspect(width, height)

    lights.ambient_color = vec3(0.31, 0.27, 0.4)
    lights.light_dir = vec3(0.5773502691896258, -0.5773502691896258, 0.5773502691896258)
    lights.light_color = vec3(0.9, 0.82, 0.7000000000000001)
    lights.point_light = PointLight(
        position=np.array([centroid[0] + 40.0, centroid[1] - 60.0, centroid[2] + 90.0]),
        color=(1.0, 0.6, 0.30000000000000004),
        intensity=2.25,
        range=320.5,
        enabled=True,
    )

    # A deliberately mixed scene: a global mode plus per-structure overrides,
    # per-structure colours and opacities, and two structures hidden.
    modes = [RenderMode.SOLID, RenderMode.SOLID, RenderMode.XRAY, RenderMode.SOLID,
             RenderMode.ILLUSTRATION, RenderMode.SOLID, RenderMode.SOLID,
             RenderMode.THERMAL]
    for i, (_, node) in enumerate(entries):
        m = node.mesh
        m.material.render_mode = modes[i % len(modes)]
        m.material.color = (0.5 + 0.05 * i, 0.4 + 0.03 * i, 0.30000000000000004)
        m.material.opacity = 1.0 if i % 3 else 0.6499999999999999
        m.material.transparent = m.material.opacity < 1.0
        m.material.shininess = 12.0 + i
        m.visible = i != 2
        node.visible = i != 5

    renderer.CLEAR_COLOR = (0.07, 0.09, 0.13, 1.0)
    renderer._bg_color_dirty = True
    renderer.set_clip_plane((1.0, 0.0, 0.0), -3.5)


def make_fbo(width, height):
    from tools.capture_golden import _make_fbo

    return _make_fbo(width, height)


def read_frame(width, height):
    from tools.capture_golden import _read_frame

    return _read_frame(width, height)


def render_once(scene, camera, lights, renderer, width, height):
    t0 = time.perf_counter()
    renderer.render(scene, camera, lights)
    img = read_frame(width, height)
    return img, (time.perf_counter() - t0) * 1000.0


def diff_report(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    changed = (d.max(axis=2) > 0)
    return {
        "identical": bool(np.array_equal(a, b)),
        "max_abs_channel_diff": int(d.max()),
        "changed_pixels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "total_pixels": int(changed.size),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--meshes", type=int, default=8)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args(argv)

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    width = height = args.size
    stl_dir = repo / "assets" / "stl"

    from tools.glcontext import acquire_offscreen_gl

    gl_info = acquire_offscreen_gl("software")
    print(gl_info.banner())

    from PIL import Image

    from faceforge.core import scene_state as ss
    from faceforge.core.state import StateManager
    from faceforge.rendering.camera import Camera
    from faceforge.rendering.lights import LightSetup
    from faceforge.rendering.renderer import GLRenderer

    make_fbo(width, height)
    results: dict = {"gl": gl_info.as_manifest_dict(),
                     "viewport": [width, height], "meshes": args.meshes}

    # ---- A: perturbed scene, captured and rendered -----------------------
    scene_a, meshes_a = build_scene(args.meshes, stl_dir)
    results["triangles"] = int(sum(m.geometry.triangle_count for m in meshes_a))
    camera_a, lights_a = Camera(), LightSetup()
    renderer_a = GLRenderer()
    renderer_a.init_gl()
    renderer_a.resize(width, height)
    perturb(scene_a, camera_a, lights_a, renderer_a, width, height)

    sm = StateManager()
    sm.face.set_au("AU12", 0.65)
    sm.body.gender = 0.35
    state = ss.capture_scene_state(
        scene=scene_a, camera=camera_a, lights=lights_a, renderer=renderer_a,
        state=sm, tier=1, skull_mode="original", stl_dir=stl_dir,
        gender_applied=0.35,
    )
    state_path = out / "scene.state.json"
    ss.save(state, state_path)
    img_a, ms_a = render_once(scene_a, camera_a, lights_a, renderer_a, width, height)
    Image.fromarray(img_a, mode="RGBA").save(out / "A_captured.png")

    # ---- B: fresh everything, state applied ------------------------------
    scene_b, _ = build_scene(args.meshes, stl_dir)
    camera_b, lights_b = Camera(), LightSetup()
    renderer_b = GLRenderer()
    renderer_b.init_gl()
    renderer_b.resize(width, height)

    loaded = ss.load(state_path)
    report = ss.apply_scene_state(
        loaded, scene=scene_b, camera=camera_b, lights=lights_b,
        renderer=renderer_b, state_manager=StateManager(), strict=True,
    )
    results["apply_report"] = report.summary()
    img_b, ms_b = render_once(scene_b, camera_b, lights_b, renderer_b, width, height)
    Image.fromarray(img_b, mode="RGBA").save(out / "B_reapplied.png")

    # ---- C: negative control, one field changed --------------------------
    from dataclasses import replace

    nudged = replace(loaded, camera=replace(loaded.camera,
                                            fov_deg=loaded.camera.fov_deg + 0.05))
    ss.apply_scene_state(nudged, scene=scene_b, camera=camera_b, lights=lights_b,
                         renderer=renderer_b, strict=True)
    img_c, _ = render_once(scene_b, camera_b, lights_b, renderer_b, width, height)
    Image.fromarray(img_c, mode="RGBA").save(out / "C_negative_control.png")

    # ---- D: second negative control, one material opacity changed --------
    tgt = next(i for i, s in enumerate(loaded.structures) if s.material.opacity == 1.0)
    nudged2 = replace(
        loaded,
        structures=loaded.structures[:tgt]
        + (replace(loaded.structures[tgt],
                   material=replace(loaded.structures[tgt].material, opacity=0.9)),)
        + loaded.structures[tgt + 1:],
    )
    ss.apply_scene_state(nudged2, scene=scene_b, camera=camera_b, lights=lights_b,
                         renderer=renderer_b, strict=True)
    img_d, _ = render_once(scene_b, camera_b, lights_b, renderer_b, width, height)
    Image.fromarray(img_d, mode="RGBA").save(out / "D_negative_control_opacity.png")

    results["A_vs_B"] = diff_report(img_a, img_b)
    results["A_vs_C_fov_plus_0.05deg"] = diff_report(img_a, img_c)
    results["A_vs_D_opacity_1.0_to_0.9"] = diff_report(img_a, img_d)
    results["frame_ms_cpu_rasteriser_not_a_benchmark"] = {
        "A": round(ms_a, 1), "B": round(ms_b, 1)}
    results["state_bytes"] = state_path.stat().st_size
    results["structures"] = len(loaded.structures)
    results["visible_structures"] = len(loaded.visible_structures)

    # Content check, so a pair of blank frames cannot pass as "identical".
    from tools.capture_golden import frame_content_fraction
    clear = tuple(int(round(c * 255)) for c in loaded.render.clear_color[:3])
    results["content_fraction_A"] = round(frame_content_fraction(img_a, clear), 6)
    results["content_fraction_B"] = round(frame_content_fraction(img_b, clear), 6)

    (out / "pixel_roundtrip_result.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))

    ok = (
        results["A_vs_B"]["identical"]
        and not results["A_vs_C_fov_plus_0.05deg"]["identical"]
        and not results["A_vs_D_opacity_1.0_to_0.9"]["identical"]
        and results["content_fraction_A"] > 0.01
    )
    print("PIXEL ROUND-TRIP:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
