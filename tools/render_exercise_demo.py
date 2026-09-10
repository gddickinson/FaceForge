"""Render an exercise demonstration headlessly, for evaluation.

Drives the same :class:`faceforge.exercise.runtime.ExerciseRuntime` the app
uses, on the scene built by ``tools/headless_loader.py`` (the app's own
skeleton, joint pivots and skinning), through the real GL renderer via
:class:`faceforge.session.Session`.  What comes out is what the app shows:
the posed skeleton, the muscles the exercise colours by activation, the
equipment in the hands, the gym room.

Usage::

    python -m tools.render_exercise_demo --exercise bodyweight_squat
    python -m tools.render_exercise_demo --exercise barbell_back_squat --frames 36 \
        --camera side --size 720x900 --out results/exercise_demo
    python -m tools.render_exercise_demo --probe --exercise conventional_deadlift
    python -m tools.render_exercise_demo --probe --all      # every exercise, no GL

``--probe`` needs no GL: it applies each phase's pose and prints where the
feet, hands and equipment ended up relative to the floor, so a definition can
be checked numerically before it is rendered.  Outputs: numbered PNG frames,
a contact sheet of the phase key positions, and (if ffmpeg is on PATH) an
MP4 and a GIF.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

from faceforge.body.muscle_activation import MuscleActivationSystem
from faceforge.core.state import BodyState
from faceforge.exercise.catalog import get_exercise_catalog
from faceforge.exercise.muscle_groups import ALL_MUSCLE_REGIONS, regions_for_groups
from faceforge.exercise.stabilisers import with_implied_stabilisers
from faceforge.exercise.runtime import ExerciseRuntime
from faceforge.scene.scene_animation import AnimationPlayer
from faceforge.scene.scene_mode_controller import SceneModeController

logger = logging.getLogger("render_exercise_demo")

DEFAULT_OUT = Path("results/exercise_demo")


# ── Scene assembly ────────────────────────────────────────────────────

class DemoScene:
    """The headless scene plus the player, runtime and activation system."""

    def __init__(self, layers: list[str], with_skin: bool = False) -> None:
        from tools.headless_loader import load_headless_scene, load_layer, register_layer

        t0 = time.perf_counter()
        self.hs = load_headless_scene()
        self.activation = MuscleActivationSystem()
        for layer in layers + (["skin"] if with_skin else []):
            meshes = load_layer(self.hs, layer)
            register_layer(self.hs, meshes, layer)
            if layer != "skin":
                for mesh in meshes:
                    self.activation.register_muscle(mesh, mesh.name)
        # The skinning captures its neutral reference on its first update; the
        # app does that in the clinical view, before any scene wrapper exists.
        from tools.headless_loader import apply_pose
        apply_pose(self.hs, BodyState())
        self.scene = self.hs.scene
        self.body_root = self.hs.named_nodes["bodyRoot"]
        self.smc = SceneModeController()
        self.player = AnimationPlayer()
        self.runtime: ExerciseRuntime | None = None
        logger.info("scene ready in %.1f s (%d muscles registered)",
                    time.perf_counter() - t0, len(self.activation.muscle_names))

    def activate(self, camera, lights, scene_type: str = "gym") -> None:
        self.smc.activate(self.body_root, self.scene, camera, lights, scene_type=scene_type)
        self.hs.skinning.scene_wrapper = self.smc.wrapper_node
        self.scene.update()

    def start(self, defn, reps: int, tempo: float, equipment: bool = True) -> None:
        jp = self.hs.pipeline.joint_setup
        self.player.on_wrapper_transform = self.smc.set_wrapper_transform
        self.player.on_body_state = lambda d: self.hs.body_state.set_from_js_dict(d)
        self.runtime = ExerciseRuntime(
            player=self.player, scene=self.scene, wrapper=self.smc.wrapper_node,
            pivots=jp.pivots, joint_positions=jp.joint_positions,
            muscle_activation=self.activation, show_equipment=equipment,
            body_animation=self.hs.body_animation,
        )
        self.activation.set_enabled(True)
        self.runtime.start(defn, reps=reps, tempo=tempo, autoplay=False)

    def evaluate(self, t: float) -> None:
        """Pose the whole scene at clip time ``t`` (seconds)."""
        hs = self.hs
        duration = self.player.duration
        self.player.seek(0.0 if duration <= 0 else t / duration)
        self.runtime.after_animation(0.0)
        hs.body_constraints.clamp(hs.body_state)
        hs.body_animation.apply(hs.body_state, dt=1 / 60)
        self.scene.update()
        hs.skinning._last_signature = ""
        hs.skinning.update(hs.body_state)
        self.activation.update(hs.body_state)
        self.scene.update()
        self.runtime.after_scene_update()

    # -- measurements ------------------------------------------------------------

    def measure(self) -> dict:
        piv = self.hs.pipeline.joint_setup.pivots

        def y(name):
            n = piv.get(name)
            return None if n is None else float(n.get_world_position()[1])

        feet = [y(n) for n in [f"ankle_{s}" for s in "RL"] +
                [f"toe_{s}_{d}_mt" for s in "RL" for d in (1, 3, 5)]]
        feet = [v for v in feet if v is not None]
        hands = [y(f"wrist_{s}") for s in "RL"]
        hands = [v for v in hands if v is not None]
        out = {
            "foot_min_y": min(feet) if feet else None,
            "ankle_R_y": y("ankle_R"), "toe_R_1_y": y("toe_R_1_mt"),
            "hand_min_y": min(hands) if hands else None,
            "hand_max_y": max(hands) if hands else None,
            "wrapper_y": float(self.smc.wrapper_node.position[1]),
        }
        for item in self.runtime.rig.items:
            out[f"equip_{item.spec.kind}_{item.spec.attach}"] = [
                round(float(v), 1) for v in item.node.position]
        out["joints"] = {
            n: [round(float(v), 1) for v in piv[n].get_world_position()]
            for n in ("shoulder_R", "wrist_R", "hip_R", "knee_R", "ankle_R", "toe_R_3_mt",
                      "hip_L", "knee_L", "ankle_L", "toe_L_3_mt") if n in piv}
        return out


# ── Probe (no GL) ─────────────────────────────────────────────────────

def probe(defn, demo: DemoScene) -> list[dict]:
    """Report placement at the end of every phase of the first rep."""
    demo.start(defn, reps=1, tempo=1.0)
    rows = []
    built = demo.runtime.built
    for span in built.spans:
        demo.evaluate(span.t1 - 1e-3)
        m = demo.measure()
        flags = []
        if defn.anchor == "feet" and m["foot_min_y"] is not None:
            if abs(m["foot_min_y"] - 8.0) > 4.0:
                flags.append(f"feet not on floor (min y {m['foot_min_y']:.1f})")
        if defn.anchor == "hands" and m["hand_min_y"] is not None and defn.anchor_point is None:
            if abs(m["hand_min_y"] - 3.0) > 4.0:
                flags.append(f"hands not on floor (min y {m['hand_min_y']:.1f})")
        if m["foot_min_y"] is not None and m["foot_min_y"] < -2.0:
            flags.append("feet below floor")
        if m["hand_min_y"] is not None and m["hand_min_y"] < -2.0:
            flags.append("hands below floor")
        rows.append({"phase": span.name, "t": round(span.t1, 2), **m, "flags": flags})
    demo.runtime.stop()
    return rows


def print_probe(defn, rows: list[dict]) -> None:
    print(f"\n== {defn.id} ({defn.orientation}, anchor={defn.anchor}) ==")
    for r in rows:
        fm = "-" if r["foot_min_y"] is None else f"{r['foot_min_y']:6.1f}"
        hm = "-" if r["hand_min_y"] is None else f"{r['hand_min_y']:6.1f}"
        eq = {k: v for k, v in r.items() if k.startswith("equip_")}
        eq_s = " ".join(f"{k[6:]}={v}" for k, v in eq.items())
        flag = ("  <-- " + "; ".join(r["flags"])) if r["flags"] else ""
        print(f"  {r['phase']:<28s} t={r['t']:5.2f} foot_min={fm} hand_min={hm} "
              f"wrapper_y={r['wrapper_y']:6.1f} {eq_s}{flag}")
        if logger.isEnabledFor(logging.INFO):
            print("      " + "  ".join(f"{k}={v}" for k, v in r["joints"].items()))


# ── Rendering ─────────────────────────────────────────────────────────

def render(defn, demo: DemoScene, out: Path, frames: int, size: tuple[int, int],
           camera_preset: str | None, reps: int, tempo: float) -> dict:
    from PIL import Image, ImageDraw

    from faceforge.session import Session, write_png

    out.mkdir(parents=True, exist_ok=True)
    with Session.create(width=size[0], height=size[1]) as session:
        session.adopt_scene(demo.scene)
        demo.activate(session.camera, session.lights, "gym")
        demo.start(defn, reps=reps, tempo=tempo)
        preset = camera_preset or defn.camera
        demo.smc.set_camera_preset(session.camera, preset, target=defn.camera_target)
        session.camera.set_aspect(size[0], size[1])
        built = demo.runtime.built
        duration = built.duration
        times = [duration * i / frames for i in range(frames)]
        frame_paths = []
        for i, t in enumerate(times):
            demo.evaluate(t)
            image = session.render()
            path = out / f"{defn.id}_{i:03d}.png"
            write_png(path, image)
            frame_paths.append(path)
            span = built.span_at(t)
            logger.info("frame %d/%d t=%.2fs %s", i + 1, frames, t, span.name if span else "")

        # Key positions: the end of each phase of the first rep.
        keys = []
        for span in built.spans[:len(defn.phases)]:
            demo.evaluate(span.t1 - 1e-3)
            image = session.render()
            keys.append((span, Image.fromarray(image[:, :, :3])))
        demo.runtime.stop()

    sheet = _contact_sheet(keys, defn)
    sheet_path = out / f"{defn.id}_phases.png"
    sheet.save(sheet_path)
    videos = _encode(out, defn.id, frames, duration)
    return {"frames": [str(p) for p in frame_paths], "sheet": str(sheet_path), **videos}


def _contact_sheet(keys, defn):
    from PIL import Image, ImageDraw

    if not keys:
        return Image.new("RGB", (10, 10))
    w, h = keys[0][1].size
    scale = min(1.0, 360 / w)
    tw, th = int(w * scale), int(h * scale)
    cols = min(4, len(keys))
    rows = (len(keys) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 26)), (24, 26, 32))
    draw = ImageDraw.Draw(sheet)
    for i, (span, img) in enumerate(keys):
        x, y = (i % cols) * tw, (i // cols) * (th + 26)
        sheet.paste(img.resize((tw, th)), (x, y + 26))
        draw.text((x + 6, y + 6), f"{span.name} ({span.kind.value})", fill=(230, 230, 230))
    return sheet


def _encode(out: Path, stem: str, frames: int, duration: float) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or frames < 2:
        return {}
    fps = max(4, min(30, round(frames / max(duration, 0.5))))
    pattern = str(out / f"{stem}_%03d.png")
    mp4 = out / f"{stem}.mp4"
    gif = out / f"{stem}.gif"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", str(mp4)], check=False)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                    "-vf", "fps=12,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];"
                    "[s1][p]paletteuse", str(gif)], check=False)
    return {"mp4": str(mp4) if mp4.exists() else "", "gif": str(gif) if gif.exists() else "",
            "fps": fps}


# ── CLI ───────────────────────────────────────────────────────────────

def parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exercise", default="bodyweight_squat",
                    help="exercise id, or a comma list for --probe")
    ap.add_argument("--all", action="store_true", help="probe every exercise")
    ap.add_argument("--probe", action="store_true", help="measure placement, no GL")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--tempo", type=float, default=1.0)
    ap.add_argument("--size", default="640x800")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--layers", default=None, help="comma list; default: the exercise's")
    ap.add_argument("--skin", action="store_true")
    ap.add_argument("--no-equipment", action="store_true")
    ap.add_argument("--all-muscles", action="store_true", help="load every body muscle layer")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(name)s: %(message)s")

    catalog = get_exercise_catalog()
    ids = list(catalog) if args.all else [x.strip() for x in args.exercise.split(",") if x.strip()]
    for exercise_id in ids:
        if exercise_id not in catalog:
            print(f"unknown exercise {exercise_id!r}; have: {', '.join(catalog)}")
            return 2

    if args.probe:
        layers = ["leg_muscles"] if args.layers is None else args.layers.split(",")
        demo = DemoScene(layers, with_skin=False)
        demo.activate(_NullCamera(), _NullLights(), "gym")
        for exercise_id in ids:
            rows = probe(catalog[exercise_id], demo)
            print_probe(catalog[exercise_id], rows)
        return 0

    defn = catalog[ids[0]]
    if args.layers is not None:
        layers = args.layers.split(",")
    elif args.all_muscles:
        layers = list(ALL_MUSCLE_REGIONS)
    else:
        layers = regions_for_groups(with_implied_stabilisers(defn).muscle_groups)
    demo = DemoScene(layers, with_skin=args.skin)
    result = render(defn, demo, Path(args.out), args.frames, parse_size(args.size),
                    args.camera, args.reps or defn.default_reps, args.tempo)
    if args.no_equipment:
        pass
    (Path(args.out) / f"{defn.id}_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
    return 0


class _NullCamera:
    """Stands in for a Camera when probing without GL."""

    def __init__(self):
        self.position = np.zeros(3)
        self.target = np.zeros(3)
        self.up = np.array([0.0, 1.0, 0.0])
        self._view_dirty = True

    def set_position(self, *p):
        self.position = np.array(p, dtype=float)

    def set_target(self, *p):
        self.target = np.array(p, dtype=float)


class _NullLights:
    point_light = None


if __name__ == "__main__":
    sys.exit(main())
