"""Automated GUI screenshot capture for the README.

Launches FaceForge, waits for loading, applies different views/expressions/
render modes, and captures screenshots using QWidget.grab().

Usage::

    PYTHONPATH=src python -m tools.capture_gui_screenshots
"""

from __future__ import annotations

# Must disable before any GL imports (same as app.py)
import OpenGL
OpenGL.ERROR_CHECKING = False

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from faceforge.core.events import EventBus, EventType
from faceforge.core.material import RenderMode
from faceforge.core.state import StateManager
from faceforge.core.clock import DeltaClock
from faceforge.loaders.asset_manager import AssetManager
from faceforge.coordination.scene_builder import SceneBuilder
from faceforge.coordination.visibility import VisibilityManager
from faceforge.coordination.simulation import Simulation
from faceforge.coordination.loading_pipeline import LoadingPipeline
from faceforge.animation.preset_manager import PresetManager
from faceforge.body.body_animation import BodyAnimationSystem
from faceforge.body.body_constraints import BodyConstraints
from faceforge.rendering.gl_widget import GLViewport, create_gl_format
from faceforge.ui.main_window import MainWindow

OUTPUT_DIR = Path("docs/images")

# Camera presets from app.py
CAM = {
    "head_front":         ((0, -35, -5),    (0, 0, -5)),
    "head_three_quarter": ((18, -26, 3),    (0, 0, -5)),
    "head_right":         ((35, 0, -5),     (0, 0, -5)),
    "body_front":         ((0, -150, -80),  (0, 0, -80)),
    "body_three_quarter": ((80, -110, -50), (0, 0, -80)),
    "body_right":         ((150, 0, -80),   (0, 0, -80)),
    # Full body views - pulled back to show head to feet
    # Body spans Z +29 (skull top) to -197 (feet), center ~-84, height ~226
    "full_body_front":         ((0, -280, -84),    (0, 0, -84)),
    "full_body_three_quarter": ((140, -230, -60),  (0, 0, -84)),
    "full_body_right":         ((280, 0, -84),     (0, 0, -84)),
}

# Auto-background colours per render mode (from app.py)
MODE_BG = {
    RenderMode.ILLUSTRATION: (0.96, 0.94, 0.90, 1.0),
    RenderMode.HOLOGRAM:     (0.02, 0.03, 0.06, 1.0),
    RenderMode.BLUEPRINT:    (0.05, 0.12, 0.28, 1.0),
    RenderMode.XRAY:         None,  # keep default
    RenderMode.MEDICAL:      (0.12, 0.14, 0.18, 1.0),
    RenderMode.WIREFRAME:    None,
    RenderMode.SOLID:        None,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    from PySide6.QtGui import QSurfaceFormat
    fmt = create_gl_format()
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)

    # ── Build application systems ───────────────────────────────────
    event_bus = EventBus()
    state = StateManager()
    clock = DeltaClock()

    assets = AssetManager()
    assets.init_transform()

    visibility = VisibilityManager()
    builder = SceneBuilder(assets, visibility)
    scene, named_nodes = builder.build()

    simulation = Simulation(state, scene)

    presets = PresetManager()
    try:
        presets.load()
    except FileNotFoundError:
        pass

    gl_widget = GLViewport()
    gl_widget.scene = scene

    window = MainWindow(event_bus, state, gl_widget)

    # ── Loading pipeline ────────────────────────────────────────────
    pipeline = LoadingPipeline(assets, event_bus, named_nodes)

    print("Loading head...")
    try:
        pipeline.load_head()
    except Exception as e:
        print(f"  Head: {e}")

    print("Loading body skeleton...")
    try:
        pipeline.load_body_skeleton()
    except Exception as e:
        print(f"  Skeleton: {e}")

    # Wire up simulation systems
    if pipeline.facs_engine is not None:
        simulation.facs_engine = pipeline.facs_engine
    if pipeline.head_rotation is not None:
        simulation.head_rotation = pipeline.head_rotation
        simulation.vertebrae_pivots = pipeline.vertebrae_pivots
        simulation.neck_constraints = pipeline.neck_constraints
        simulation.skull_group = named_nodes.get("skullGroup")
        simulation.face_group = named_nodes.get("faceGroup")
        simulation.brain_group = named_nodes.get("brainGroup")
        simulation.stl_muscle_group = named_nodes.get("stlMuscleGroup")
        simulation.expr_muscle_group = named_nodes.get("exprMuscleGroup")
        simulation.face_feature_group = named_nodes.get("faceFeatureGroup")
    if pipeline.neck_muscles is not None:
        simulation.neck_muscles = pipeline.neck_muscles
    if pipeline.jaw_muscles is not None:
        simulation.jaw_muscles = pipeline.jaw_muscles
    if pipeline.expression_muscles is not None:
        simulation.expression_muscles = pipeline.expression_muscles

    # Body animation
    if pipeline.joint_setup is not None:
        body_anim = BodyAnimationSystem(pipeline.joint_setup)
        body_anim.load_fractions()
        if pipeline.skeleton is not None:
            body_anim.set_thoracic_pivots(
                pipeline.skeleton.pivots.get("thoracic", []))
            body_anim.set_lumbar_pivots(
                pipeline.skeleton.pivots.get("lumbar", []))
            if pipeline.skeleton.rib_nodes:
                body_anim.set_rib_nodes(pipeline.skeleton.rib_nodes)
        simulation.body_animation = body_anim

    body_constraints = BodyConstraints()
    body_constraints.load()
    simulation.body_constraints = body_constraints

    # ── Wire up render mode + clip plane handlers ───────────────────
    _prev_bg_color = None

    def on_render_mode_changed(mode=RenderMode.WIREFRAME, **kw):
        nonlocal _prev_bg_color
        meshes = scene.collect_meshes()
        for mesh, _ in meshes:
            mesh.material.render_mode = mode
        renderer = gl_widget.renderer
        if renderer is not None:
            auto_bg = MODE_BG.get(mode)
            if auto_bg is not None:
                if _prev_bg_color is None:
                    _prev_bg_color = renderer.CLEAR_COLOR
                renderer.CLEAR_COLOR = auto_bg
                renderer._bg_color_dirty = True
            elif _prev_bg_color is not None:
                renderer.CLEAR_COLOR = _prev_bg_color
                renderer._bg_color_dirty = True
                _prev_bg_color = None

    def on_clip_plane_changed(enabled=False, axis="x", offset=0.0,
                              flip=False, **kw):
        renderer = gl_widget.renderer
        if renderer is None:
            return
        if not enabled:
            renderer.clip_plane = None
            return
        import numpy as np
        normal = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
        normal = np.array(normal, dtype=np.float32)
        if flip:
            normal = -normal
        renderer.clip_plane = (*normal, offset)

    # Wire up on-demand muscle loading (mirrors app.py)
    _loaded_muscles = set()

    def on_layer_toggled(layer="", visible=True, **kw):
        muscle_regions = {
            "back_muscles": "back_muscles.json",
            "shoulder_muscles": "shoulder_muscles.json",
            "arm_muscles": "arm_muscles.json",
            "torso_muscles": "torso_muscles.json",
            "hip_muscles": "hip_muscles.json",
            "leg_muscles": "leg_muscles.json",
        }
        if layer in muscle_regions and visible and layer not in _loaded_muscles:
            try:
                body_root = named_nodes.get("bodyRoot")
                result = assets.load_body_muscles(muscle_regions[layer])
                if body_root and result.group:
                    body_root.add(result.group)
                visibility.register(layer, result.group)
                _loaded_muscles.add(layer)
                # Apply current render mode
                for mesh in result.meshes:
                    existing = scene.collect_meshes()
                    if existing:
                        mesh.material.render_mode = existing[0][0].material.render_mode
            except Exception as e:
                print(f"  Muscle load failed ({layer}): {e}")

        visibility.set_visible(layer, visible)

    event_bus.subscribe(EventType.RENDER_MODE_CHANGED, on_render_mode_changed)
    event_bus.subscribe(EventType.CLIP_PLANE_CHANGED, on_clip_plane_changed)
    event_bus.subscribe(EventType.LAYER_TOGGLED, on_layer_toggled)

    # ── Label overlay ───────────────────────────────────────────────
    from faceforge.ui.widgets.label_overlay import LabelOverlay
    from faceforge.core.math_utils import transform_point
    label_overlay = LabelOverlay(gl_widget)
    _labels_enabled = False
    _labels_dirty = True

    def enable_labels():
        nonlocal _labels_enabled, _labels_dirty
        _labels_enabled = True
        _labels_dirty = True
        label_overlay.set_enabled(True)

    def disable_labels():
        nonlocal _labels_enabled
        _labels_enabled = False
        label_overlay.set_enabled(False)

    def rebuild_labels():
        nonlocal _labels_dirty
        _labels_dirty = False
        meshes = scene.collect_meshes()
        labels = []
        for mesh, world_mat in meshes:
            name = mesh.name
            if not name or name in ("face",):
                continue
            center = mesh.geometry.get_bounding_center()
            world_center = transform_point(world_mat, center)
            labels.append((name, world_center))
        label_overlay.set_labels(labels)

    def update_labels():
        if _labels_enabled:
            if _labels_dirty:
                rebuild_labels()
            label_overlay.set_view_proj(gl_widget.camera.get_view_projection())
            label_overlay.update()

    # ── Color helpers ───────────────────────────────────────────────
    def color_meshes_by_group():
        """Apply distinct colors to different anatomical groups."""
        group_colors = {
            "skullGroup":       (0.85, 0.82, 0.75),  # warm bone
            "faceGroup":        (0.92, 0.78, 0.65),  # skin tone
            "stlMuscleGroup":   (0.75, 0.28, 0.28),  # deep red
            "exprMuscleGroup":  (0.78, 0.40, 0.28),  # rust orange
            "neckMuscleGroup":  (0.72, 0.25, 0.25),  # darker red
            "faceFeatureGroup": (0.45, 0.65, 0.82),  # steel blue
            "vertebraeGroup":   (0.88, 0.80, 0.65),  # pale gold
        }
        for group_name, color in group_colors.items():
            node = named_nodes.get(group_name)
            if node is None:
                continue
            def _set_color(n, c=color):
                if n.mesh is not None:
                    n.mesh.material.color = c
            node.traverse(_set_color)

    def color_body_muscles(color):
        """Color all loaded body muscle meshes."""
        for layer_name in _loaded_muscles:
            nodes = visibility._toggles.get(layer_name, [])
            for node in nodes:
                def _set(n, c=color):
                    if n.mesh is not None:
                        n.mesh.material.color = c
                node.traverse(_set)

    def color_skeleton(color):
        """Color body skeleton meshes."""
        if pipeline.skeleton is not None:
            for gname, node in pipeline.skeleton.groups.items():
                def _set(n, c=color):
                    if n.mesh is not None:
                        n.mesh.material.color = c
                node.traverse(_set)

    print("Loading complete.\n")

    # ── Helpers ──────────────────────────────────────────────────────
    def pump(n=10, dt=0.05):
        for _ in range(n):
            simulation.step(clock.get_delta())
            scene.update()
            update_labels()
            app.processEvents()
            time.sleep(dt)

    def grab(name: str):
        pump(5, 0.03)
        gl_widget.update()
        pump(3, 0.03)
        pixmap = window.grab()
        path = OUTPUT_DIR / name
        pixmap.save(str(path), "PNG")
        print(f"  Saved: {path} ({pixmap.width()}x{pixmap.height()})")

    def set_cam(preset: str):
        pos, target = CAM[preset]
        cam = gl_widget.camera
        if cam:
            cam.set_position(*pos)
            cam.set_target(*target)
            gl_widget.update()

    def set_cam_custom(pos, target):
        cam = gl_widget.camera
        if cam:
            cam.set_position(*pos)
            cam.set_target(*target)
            gl_widget.update()

    def set_render_mode(mode: RenderMode):
        on_render_mode_changed(mode=mode)
        gl_widget.update()

    def reset_expression():
        presets.set_expression("neutral", state)
        state.target_head.head_yaw = 0.0
        state.target_head.head_pitch = 0.0
        state.target_head.head_roll = 0.0

    def set_tab(idx: int):
        """Switch the control panel tab. 0=ANIMATE 1=BODY 2=LAYERS 3=ALIGN 4=DISPLAY 5=DEBUG"""
        window.control_panel.tabs.setCurrentIndex(idx)

    # ── Screenshot sequence ─────────────────────────────────────────
    def capture_all():
        n = 11
        print(f"Capturing {n} screenshots...\n")

        # ── 1. Hero shot: Head 3/4 view with anatomy ────────────────
        print(f"[1/{n}] Head anatomy (3/4 view)...")
        set_tab(0)  # Animate tab
        set_cam("head_three_quarter")
        pump(25, 0.05)
        grab("gui_main.png")

        # ── 2. Surprised expression (most dramatic) ─────────────────
        print(f"[2/{n}] Surprised expression...")
        presets.set_expression("surprised", state)
        state.target_head.head_yaw = -0.1
        state.target_head.head_pitch = 0.05
        set_cam_custom((5, -32, -3), (0, 0, -5))  # near-front
        pump(50, 0.04)
        grab("gui_expression.png")

        # ── 3. Full body front (entire body visible) ────────────────
        print(f"[3/{n}] Full body (front view)...")
        reset_expression()
        set_cam("full_body_front")
        pump(30, 0.04)
        grab("gui_body.png")

        # ── 4. Illustration render mode ─────────────────────────────
        print(f"[4/{n}] Illustration render mode...")
        set_tab(4)  # Display tab
        set_render_mode(RenderMode.ILLUSTRATION)
        set_cam("head_three_quarter")
        pump(20, 0.04)
        grab("gui_illustration.png")

        # ── 5. Hologram render mode — full body ─────────────────────
        print(f"[5/{n}] Hologram render mode (full body)...")
        set_render_mode(RenderMode.HOLOGRAM)
        set_cam("full_body_three_quarter")
        pump(15, 0.04)
        grab("gui_hologram.png")

        # ── 6. Blueprint render mode — full body ────────────────────
        print(f"[6/{n}] Blueprint render mode (full body)...")
        set_render_mode(RenderMode.BLUEPRINT)
        set_cam("full_body_three_quarter")
        pump(20, 0.04)
        grab("gui_blueprint.png")

        # ── 7. Body pose (sitting) — wireframe, full body ──────────
        print(f"[7/{n}] Body sitting pose...")
        set_render_mode(RenderMode.WIREFRAME)
        set_tab(1)  # Body tab
        presets.set_body_pose("sitting", state)
        set_cam("full_body_three_quarter")
        pump(40, 0.04)
        grab("gui_body_pose.png")

        # ── 8. X-Ray render mode — head close-up ───────────────────
        print(f"[8/{n}] X-Ray head view...")
        presets.set_body_pose("anatomical", state)
        set_render_mode(RenderMode.XRAY)
        set_tab(4)  # Display tab
        presets.set_expression("thinking", state)
        set_cam("head_three_quarter")
        pump(40, 0.04)
        grab("gui_xray.png")

        # ── 9. Head with colored layers + labels ────────────────────
        print(f"[9/{n}] Head layers with labels...")
        reset_expression()
        set_render_mode(RenderMode.SOLID)
        color_meshes_by_group()
        set_cam("head_three_quarter")
        set_tab(2)  # Layers tab
        enable_labels()
        pump(30, 0.04)
        grab("gui_labels_head.png")

        # ── 10. Full body with colored muscles + labels ─────────────
        print(f"[10/{n}] Full body muscles with labels...")
        # Load body muscles
        for layer in ["back_muscles", "shoulder_muscles", "arm_muscles", "torso_muscles"]:
            on_layer_toggled(layer=layer, visible=True)
            app.processEvents()
        pump(10, 0.05)

        # Color the muscles distinctly
        color_body_muscles((0.72, 0.25, 0.22))
        # Color skeleton lighter to contrast
        color_skeleton((0.90, 0.85, 0.75))

        set_cam("full_body_three_quarter")
        pump(30, 0.04)
        grab("gui_labels_body.png")

        # ── 11. Head close-up front with colored anatomy + labels ───
        print(f"[11/{n}] Head front with anatomy layers...")
        set_cam_custom((0, -35, -5), (0, 0, -5))  # head_front
        pump(20, 0.04)
        grab("gui_layers_head_front.png")

        disable_labels()

        print(f"\nDone! {n} screenshots saved to {OUTPUT_DIR}/")
        app.quit()

    # ── Launch ──────────────────────────────────────────────────────
    window.resize(1400, 900)
    window.show()
    pump(5, 0.05)

    QTimer.singleShot(1500, capture_all)
    app.exec()


if __name__ == "__main__":
    main()
