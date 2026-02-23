"""FaceForge application entry point.

Wires together all systems: scene graph, asset loading, simulation, rendering, UI.
"""

# Disable PyOpenGL's per-call error checking BEFORE any GL imports.
# macOS Metal translation layer leaves stale GL errors that cause
# PyOpenGL's automatic error checker to raise on every GL call.
import OpenGL
OpenGL.ERROR_CHECKING = False

import logging
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from faceforge.core.events import EventBus, EventType
from faceforge.core.material import RenderMode
from faceforge.core.state import StateManager
from faceforge.core.clock import DeltaClock
from faceforge.core.math_utils import vec3, transform_point
from faceforge.core.config_loader import load_config, load_muscle_config
from faceforge.loaders.asset_manager import AssetManager
from faceforge.coordination.scene_builder import SceneBuilder
from faceforge.coordination.visibility import VisibilityManager
from faceforge.coordination.simulation import Simulation
from faceforge.coordination.loading_pipeline import LoadingPipeline
from faceforge.animation.preset_manager import PresetManager
from faceforge.constants import set_jaw_pivot
from faceforge.body.joint_pivots import JointPivotSetup
from faceforge.body.body_animation import BodyAnimationSystem
from faceforge.body.soft_tissue import SoftTissueSkinning
from faceforge.body.body_constraints import BodyConstraints
from faceforge.body.body_muscles import BodyMuscleManager
from faceforge.body.organs import OrganManager
from faceforge.body.vasculature import VasculatureManager
from faceforge.body.brain import BrainManager
from faceforge.rendering.gl_widget import GLViewport, create_gl_format
from faceforge.scene.scene_mode_controller import SceneModeController
from faceforge.scene.scene_animation import AnimationPlayer
from faceforge.scene.builtin_animations import get_builtin_clips
from faceforge.ui.widgets.label_overlay import LabelOverlay


# Camera presets: (position, target)
# Body coordinate system: X=lateral, Y=depth (−Y=anterior), Z=vertical (head≈0, feet≈−200).
# Body visual center ≈ (0, 0, −80).  Head center ≈ (0, 0, −5).
_CAMERA_PRESETS = {
    # Body views (radius ~150, target at body visual center)
    "body_front":         ((0, -150, -80),   (0, 0, -80)),
    "body_left":          ((-150, 0, -80),   (0, 0, -80)),
    "body_right":         ((150, 0, -80),    (0, 0, -80)),
    "body_top":           ((0, 0, 80),       (0, 0, -80)),
    "body_back":          ((0, 150, -80),    (0, 0, -80)),
    "body_three_quarter": ((80, -110, -50),  (0, 0, -80)),
    # Head views (radius ~35, target at head center)
    "head_front":         ((0, -35, -5),     (0, 0, -5)),
    "head_left":          ((-35, 0, -5),     (0, 0, -5)),
    "head_right":         ((35, 0, -5),      (0, 0, -5)),
    "head_top":           ((0, 0, 30),       (0, 0, -5)),
    "head_back":          ((0, 35, -5),      (0, 0, -5)),
    "head_three_quarter": ((18, -26, 3),     (0, 0, -5)),
}


def main():
    """Launch the FaceForge application."""
    # Enable logging so warnings are visible
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # Set OpenGL format before creating QApplication
    from PySide6.QtGui import QSurfaceFormat
    fmt = create_gl_format()
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)

    # Core systems
    event_bus = EventBus()
    state = StateManager()
    clock = DeltaClock()

    # Asset manager
    assets = AssetManager()
    assets.init_transform()

    # Visibility manager
    visibility = VisibilityManager()

    # Build scene graph
    builder = SceneBuilder(assets, visibility)
    scene, named_nodes = builder.build()

    # Simulation
    simulation = Simulation(state, scene)

    # Physiology system
    from faceforge.body.physiology import PhysiologySystem
    physiology = PhysiologySystem()
    simulation.physiology = physiology

    # Preset manager
    presets = PresetManager()
    try:
        presets.load()
    except FileNotFoundError:
        pass

    # GL viewport
    gl_widget = GLViewport()
    gl_widget.scene = scene

    # Main window (imported here to avoid circular imports with gl_widget)
    from faceforge.ui.main_window import MainWindow
    window = MainWindow(event_bus, state, gl_widget)

    # Label overlay (child of GL widget)
    label_overlay = LabelOverlay(gl_widget)
    _labels_enabled = False
    _labels_dirty = True

    # ── Virtual scanner ──────────────────────────────────────────────
    from faceforge.scanner.tissue_map import TissueMapper
    from faceforge.scanner.engine import ScannerEngine
    from faceforge.scanner.scan_plane import ScanPlaneViz

    _tissue_mapper = TissueMapper()
    _scanner_engine = ScannerEngine(_tissue_mapper)
    _scan_plane_viz = ScanPlaneViz(scene)
    _scanner_window = None

    def _open_scanner():
        nonlocal _scanner_window
        from faceforge.scanner.scanner_window import ScannerWindow

        if _scanner_window is not None and _scanner_window.isVisible():
            _scanner_window.raise_()
            _scanner_window.activateWindow()
            return

        _scanner_window = ScannerWindow(window, _scanner_engine)

        def _on_scan():
            # Update 3D plane visualization immediately
            _update_scan_plane(_scanner_window.plane_params)
            # Collect meshes and run scan (synchronous with processEvents)
            meshes = scene.collect_meshes()
            _scanner_window.run_scan(meshes)

        def _on_plane_changed(plane_params):
            _update_scan_plane(plane_params)

        def _update_scan_plane(params):
            _scan_plane_viz.update(
                origin=params["origin"],
                normal=params["normal"],
                right=params["right"],
                up=params["up"],
                width=params["width"],
                height=params["height"],
            )

        _scanner_window.scan_requested.connect(_on_scan)
        _scanner_window.plane_changed.connect(_on_plane_changed)
        _scanner_window.closed.connect(lambda: _scan_plane_viz.set_visible(False))
        _scanner_window.show()

        # Show the scan plane immediately with initial params
        _scan_plane_viz.set_visible(True)
        _update_scan_plane(_scanner_window.plane_params)

    window.scanner_requested.connect(_open_scanner)

    # ── Wire UI events → state ──────────────────────────────────────
    # The UI tabs publish events; these handlers update the target state
    # that the simulation interpolates toward each frame.

    def on_au_changed(au_id: str = "", value: float = 0.0, **kw):
        if au_id == "eye_look_x":
            state.face.eye_look_x = value
        elif au_id == "eye_look_y":
            state.face.eye_look_y = value
        elif au_id == "ear_wiggle":
            state.target_ear_wiggle = value
        elif au_id == "pupil_dilation":
            state.face.pupil_dilation = value
        else:
            state.target_au.set(au_id, value)

    def on_expression_set(name: str = "", values: dict = None, **kw):
        if values is None:
            values = {}
        # Reset all AU targets, then apply preset
        for au_id in state.target_au.to_dict():
            state.target_au.set(au_id, values.get(au_id, 0.0))
        # Apply head rotation from expression if present
        state.target_head.head_yaw = values.get("headYaw", values.get("head_yaw", 0.0))
        state.target_head.head_pitch = values.get("headPitch", values.get("head_pitch", 0.0))
        state.target_head.head_roll = values.get("headRoll", values.get("head_roll", 0.0))
        state.face.current_expression = name

    def on_head_changed(head_yaw: float = 0.0, head_pitch: float = 0.0,
                        head_roll: float = 0.0, **kw):
        state.target_head.head_yaw = head_yaw
        state.target_head.head_pitch = head_pitch
        state.target_head.head_roll = head_roll

    def on_body_changed(field: str = "", value: float = 0.0, **kw):
        if field:
            py_field = state.target_body._JS_KEY_MAP.get(field, field)
            if hasattr(state.target_body, py_field):
                setattr(state.target_body, py_field, value)
                # For bool fields, also set on live state directly
                # (interpolation skips bools)
                if isinstance(getattr(state.target_body, py_field), bool):
                    setattr(state.body, py_field, value)

    def on_body_pose_set(name: str = "", values: dict = None, **kw):
        if values:
            state.target_body.set_from_js_dict(values)

    # Mutable container for the chain-building function (populated by load_assets)
    _gender_helpers: dict = {}

    def on_gender_changed(gender: float = 0.0, **kw):
        """Handle live gender slider drag — morph body surface only (fast)."""
        state.body.gender = gender
        if pipeline.gender_morph is not None and pipeline.gender_morph.loaded:
            pipeline.gender_morph.set_gender(gender)

    def on_gender_released(gender: float = 0.0, **kw):
        """Handle gender slider release — full bone scaling + re-registration."""
        state.body.gender = gender
        gm = pipeline.gender_morph
        if gm is None or not gm.loaded:
            return
        gm.set_gender(gender)

        # Collect ALL bone meshes from entire body_root subtree
        # (includes bones reparented under pivot nodes by setup_from_skeleton)
        bone_meshes: list[tuple[str, 'MeshInstance']] = []
        body_root = named_nodes.get("bodyRoot")
        if body_root is not None:
            def _collect_meshes(node):
                # Skip non-bone meshes (body surface mesh, skin, soft tissue)
                if node.mesh is not None and node.name and node.name != "body_surface":
                    bone_meshes.append((node.name, node.mesh))
                for ch in node.children:
                    _collect_meshes(ch)
            _collect_meshes(body_root)
        if bone_meshes:
            n_scaled = gm.scale_skeleton(bone_meshes)
            logging.getLogger(__name__).info(
                "Gender %.2f: scaled %d/%d bone meshes",
                gender, n_scaled, len(bone_meshes),
            )

            # Update rest poses for scaled meshes
            for _, mesh in bone_meshes:
                mesh.store_rest_pose()

        # Note: We do NOT re-call setup_from_skeleton() here. Pivots were
        # already built during initial loading and bones are reparented under
        # them. Re-calling would fail because bones are no longer in their
        # original skeleton groups. The bone scaling above updates vertex
        # positions in-place, which is sufficient.

        # Rebuild skin joints and re-register all meshes
        if simulation.soft_tissue is not None:
            body_root_node = named_nodes.get("bodyRoot")
            if body_root_node is not None:
                body_root_node.update_world_matrix(force=True)
            scene.update()

            build_fn = _gender_helpers.get("build_chains")
            new_chains = build_fn() if build_fn else []
            if new_chains:
                skinning = simulation.soft_tissue
                # Save currently registered meshes for re-registration
                old_bindings = list(skinning.bindings)
                skinning.clear_bindings()
                skinning.rebuild_skin_joints(new_chains)

                # Re-register all previously registered meshes
                for binding in old_bindings:
                    try:
                        skinning.register_skin_mesh(
                            binding.mesh,
                            is_muscle=binding.is_muscle,
                            muscle_name=binding.muscle_name,
                        )
                    except Exception as e:
                        logging.getLogger(__name__).warning(
                            "Re-registration failed for %s: %s",
                            binding.mesh.name, e,
                        )

                logging.getLogger(__name__).info(
                    "Gender re-registration complete: %d meshes",
                    len(skinning.bindings),
                )

    # On-demand loaders (created after skeleton loads)
    demand_loaders: dict = {}

    # Face feature categories that live under faceFeatureGroup
    _FACE_FEATURE_CATS = {
        "eyes": "eyes", "ears": "ears",
        "nose_cart": "nose", "eyebrows": "eyebrows",
        "throat": "throat",
    }

    def _sync_face_feature_group_visibility():
        """Ensure faceFeatureGroup is visible when any sub-category is on."""
        ff_group = named_nodes.get("faceFeatureGroup")
        if ff_group is None or pipeline.face_features is None:
            return
        # Show parent if any category has visible nodes
        any_visible = False
        for nodes in pipeline.face_features.categories.values():
            for node in nodes:
                if node.visible:
                    any_visible = True
                    break
            if any_visible:
                break
        ff_group.visible = any_visible

    def _apply_current_render_mode(meshes):
        """Apply the current global render mode to newly loaded meshes."""
        # Grab mode from an existing mesh, or default to WIREFRAME
        existing = scene.collect_meshes()
        if existing:
            mode = existing[0][0].material.render_mode
        else:
            mode = RenderMode.WIREFRAME
        for mesh in meshes:
            mesh.material.render_mode = mode

    def on_layer_toggled(layer: str = "", visible: bool = True, **kw):
        # Individual structure toggles (any prefixed toggle_id)
        _item_prefixes = (
            "organ_", "muscle_", "vasc_", "brain_",
            "pelvic_floor_", "ligaments_", "oral_",
            "cardiac_additional_", "intestinal_", "cns_additional_",
        )
        if any(layer.startswith(p) for p in _item_prefixes):
            visibility.set_visible(layer, visible)
            return

        # Face feature sub-categories use the FaceFeatureSystem directly
        if pipeline.face_features is not None and layer in _FACE_FEATURE_CATS:
            pipeline.face_features.set_category_visible(_FACE_FEATURE_CATS[layer], visible)
            _sync_face_feature_group_visibility()

        # On-demand body muscle loading (trigger load on first enable)
        muscle_regions = {
            "back_muscles": "back_muscles.json",
            "shoulder_muscles": "shoulder_muscles.json",
            "arm_muscles": "arm_muscles.json",
            "torso_muscles": "torso_muscles.json",
            "hip_muscles": "hip_muscles.json",
            "leg_muscles": "leg_muscles.json",
        }
        if layer in muscle_regions and visible:
            _load_body_muscle_region(layer, muscle_regions[layer])

        # On-demand organ/vasculature/brain/skin loading
        _demand_map = {
            "organs": _load_organs,
            "vasculature": _load_vasculature,
            "brain": _load_brain,
            "skin": _load_skin,
            "hand_muscles": _load_hand_muscles,
            "foot_muscles": _load_foot_muscles,
            "pelvic_floor": _load_pelvic_floor,
            "ligaments": _load_ligaments,
            "oral": _load_oral,
            "cardiac_additional": _load_cardiac_additional,
            "intestinal": _load_intestinal,
            "cns_additional": _load_cns_additional,
        }
        if layer in _demand_map and visible:
            _demand_map[layer]()

        # Single-STL misc toggles
        _integument_map = {
            "head_hair": "FMA70751",
            "pubic_hair": "FMA70754",
            "epicranial_aponeurosis": "FMA46768",
            "spinal_central_canal": "FMA78497",
        }
        if layer in _integument_map and visible:
            _load_single_stl(layer, _integument_map[layer])

        # Teeth toggle: controls upper_teeth + lower_teeth nodes in skull
        if layer == "teeth":
            skull_grp = named_nodes.get("skullGroup")
            if skull_grp is not None:
                for name in ("upper_teeth", "lower_teeth"):
                    node = skull_grp.find(name)
                    if node is not None:
                        node.visible = visible

        visibility.set_visible(layer, visible)

    def on_auto_blink(enabled: bool = True, **kw):
        state.face.auto_blink = enabled

    def on_auto_breathing(enabled: bool = True, **kw):
        state.face.auto_breathing = enabled

    def on_eye_tracking(enabled: bool = False, **kw):
        state.face.eye_tracking = enabled

    def on_micro_expressions(enabled: bool = False, **kw):
        state.face.micro_expressions = enabled

    def on_eye_color_set(name: str = "brown", color: tuple = (0.42, 0.26, 0.13), **kw):
        state.face.eye_color = name
        if pipeline.face_features is not None:
            pipeline.face_features.set_eye_color(color)

    # ── Scene mode controller ──
    scene_controller = SceneModeController()

    # ── Animation player ──
    anim_player = AnimationPlayer()
    builtin_clips = get_builtin_clips()

    def _on_anim_wrapper_transform(pos, quat):
        scene_controller.set_wrapper_transform(pos, quat)

    def _on_anim_body_state(state_dict):
        state.target_body.set_from_js_dict(state_dict)

    def _on_anim_face(aus_dict, head_dict):
        if aus_dict:
            for au_id, val in aus_dict.items():
                state.target_au.set(au_id, val)
        if head_dict:
            state.target_head.head_yaw = head_dict.get("headYaw", 0.0)
            state.target_head.head_pitch = head_dict.get("headPitch", 0.0)
            state.target_head.head_roll = head_dict.get("headRoll", 0.0)

    def _on_anim_complete():
        event_bus.publish(EventType.ANIM_PROGRESS, progress=1.0, time=anim_player.duration,
                          duration=anim_player.duration)
        window.control_panel.display_tab.transport.set_playing(False)

    anim_player.on_wrapper_transform = _on_anim_wrapper_transform
    anim_player.on_body_state = _on_anim_body_state
    # Camera left under user control — no on_camera callback
    anim_player.on_face = _on_anim_face
    anim_player.on_complete = _on_anim_complete

    def on_anim_play(**kw):
        anim_player.play()

    def on_anim_pause(**kw):
        anim_player.pause()

    def on_anim_stop(**kw):
        anim_player.stop()

    def on_anim_seek(position: float = 0.0, **kw):
        anim_player.seek(position)

    def on_anim_speed(speed: float = 1.0, **kw):
        anim_player.set_speed(speed)

    def on_anim_clip_selected(clip_name: str = "", **kw):
        clip = builtin_clips.get(clip_name)
        if clip is not None:
            anim_player.load(clip)
            window.control_panel.display_tab.transport.set_duration(clip.duration)

    def on_scene_mode_toggled(enabled: bool = False, scene_type: str = "examination", **kw):
        body_root = named_nodes.get("bodyRoot")
        if body_root is None:
            return
        if enabled:
            scene_controller.activate(
                body_root, scene, gl_widget.camera, gl_widget.lights,
                scene_type=scene_type,
            )
            # Tell soft tissue about the wrapper so it cancels the rotation
            # from joint delta matrices (prevents double-rotation).
            if simulation.soft_tissue is not None:
                simulation.soft_tissue.scene_wrapper = scene_controller.wrapper_node
            # Force immediate world matrix rebuild so the first render
            # sees correct matrices (not stale identity from before activation).
            scene.update()
            # Auto-load appropriate clip based on scene type
            if scene_type == "dance_studio":
                clip = builtin_clips.get("Contemporary")
            else:
                clip = builtin_clips.get("Wake Up")
            if clip is not None:
                anim_player.load(clip)
                anim_player.seek(0)
                window.control_panel.display_tab.transport.set_duration(
                    clip.duration,
                )
            # Rebuild world matrices again after seek(0) fired callbacks
            # that may have modified the wrapper transform.
            scene.update()
        else:
            # Stop animation when leaving scene mode
            anim_player.stop()
            window.control_panel.display_tab.transport.set_playing(False)
            scene_controller.deactivate(
                body_root, scene, gl_widget.camera, gl_widget.lights,
            )
            # Clear soft tissue wrapper
            if simulation.soft_tissue is not None:
                simulation.soft_tissue.scene_wrapper = None
        # Renderer no longer uses scene_transform (wrapper node handles it)
        gl_widget.renderer.scene_transform = None
        gl_widget.orbit_controls.reset_from_camera()
        # Apply current render mode to environment meshes
        existing = scene.collect_meshes()
        if existing:
            scene_controller.set_render_mode(existing[0][0].material.render_mode)

    def on_scene_camera_changed(preset: str = "", **kw):
        if scene_controller.is_active:
            scene_controller.set_camera_preset(gl_widget.camera, preset)
            gl_widget.orbit_controls.reset_from_camera()

    def on_wrapper_nudge(axis: str = "", delta: float = 0.0, **kw):
        """Manual wrapper transform nudge for debugging positioning."""
        import math as _math
        wrapper = scene_controller.wrapper_node
        if not scene_controller.is_active:
            return

        if axis == "reset":
            # Reset to default supine (empirically determined)
            from faceforge.core.math_utils import quat_from_axis_angle, quat_multiply
            q = quat_multiply(
                quat_from_axis_angle(vec3(0, 0, 1), _math.pi / 2),
                quat_multiply(
                    quat_from_axis_angle(vec3(0, 1, 0), _math.pi / 2),
                    quat_from_axis_angle(vec3(1, 0, 0), -_math.pi / 2),
                ),
            )
            wrapper.set_position(-85.0, 105.0, 0.0)
            wrapper.set_quaternion(q)
            print(f"[WRAPPER RESET] pos=(-85, 105, 0) quat={q.round(4)}")
        elif axis.startswith("p"):
            # Position nudge
            pos = wrapper.position.copy()
            idx = {"px": 0, "py": 1, "pz": 2}[axis]
            pos[idx] += delta
            wrapper.set_position(*pos)
            print(f"[WRAPPER POS] {axis}+={delta} → pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        elif axis.startswith("r"):
            # Rotation nudge: compose incremental rotation
            from faceforge.core.math_utils import quat_from_axis_angle, quat_multiply
            angle = _math.radians(delta)
            ax_vec = {"rx": vec3(1, 0, 0), "ry": vec3(0, 1, 0), "rz": vec3(0, 0, 1)}[axis]
            dq = quat_from_axis_angle(ax_vec, angle)
            new_q = quat_multiply(dq, wrapper.quaternion)
            wrapper.set_quaternion(new_q)
            print(f"[WRAPPER ROT] {axis}+={delta}° → quat=({new_q[0]:.4f}, {new_q[1]:.4f}, {new_q[2]:.4f}, {new_q[3]:.4f})")

        # Force world matrix rebuild and re-render
        scene.update()
        print(f"[WRAPPER] world_matrix diag={wrapper.world_matrix.diagonal().round(3)}")
        print(f"[WRAPPER] world_matrix pos={wrapper.world_matrix[:3, 3].round(1)}")

    # ── Display tab: render mode ──
    _prev_bg_color = None  # stash for restoring bg after styled modes

    # Auto-background colours per render mode (modes not listed keep current bg)
    _MODE_BG = {
        RenderMode.ILLUSTRATION: (0.96, 0.94, 0.90, 1.0),  # warm paper
        RenderMode.SEPIA:        (0.92, 0.86, 0.74, 1.0),  # aged parchment
        RenderMode.COLOR_ATLAS:  (0.96, 0.94, 0.90, 1.0),  # paper
        RenderMode.PEN_INK:      (1.00, 1.00, 1.00, 1.0),  # pure white
        RenderMode.MEDICAL:      (0.12, 0.14, 0.18, 1.0),  # dark slate
        RenderMode.HOLOGRAM:     (0.02, 0.03, 0.06, 1.0),  # near-black
        RenderMode.CARTOON:      (0.18, 0.20, 0.25, 1.0),  # medium dark
        RenderMode.PORCELAIN:    (0.88, 0.87, 0.85, 1.0),  # soft grey
        RenderMode.BLUEPRINT:    (0.05, 0.12, 0.28, 1.0),  # blueprint blue
        RenderMode.THERMAL:      (0.02, 0.02, 0.04, 1.0),  # near-black
        RenderMode.ETHEREAL:     (0.04, 0.02, 0.08, 1.0),  # deep purple-black
    }

    def on_render_mode_changed(mode: RenderMode = RenderMode.WIREFRAME, **kw):
        nonlocal _prev_bg_color
        meshes = scene.collect_meshes()
        for mesh, _ in meshes:
            mesh.material.render_mode = mode
        # Also update environment meshes if scene mode is active
        if scene_controller.is_active:
            scene_controller.set_render_mode(mode)
        # Auto-switch background for styled modes
        renderer = gl_widget.renderer
        auto_bg = _MODE_BG.get(mode)
        if auto_bg is not None:
            if _prev_bg_color is None:
                _prev_bg_color = renderer.CLEAR_COLOR
            renderer.CLEAR_COLOR = auto_bg
            renderer._bg_color_dirty = True
        elif _prev_bg_color is not None:
            renderer.CLEAR_COLOR = _prev_bg_color
            renderer._bg_color_dirty = True
            _prev_bg_color = None

    # ── Display tab: camera presets ──
    def on_camera_preset(preset: str = "", **kw):
        cam_data = _CAMERA_PRESETS.get(preset)
        if cam_data is None:
            return
        pos, target = cam_data
        gl_widget.camera.set_position(*pos)
        gl_widget.camera.set_target(*target)
        gl_widget.orbit_controls.reset_from_camera()

    # ── Display tab: color changes ──
    def on_color_changed(target: str = "", color: tuple = (0.8, 0.8, 0.8), **kw):
        if target == "background":
            gl_widget.renderer.CLEAR_COLOR = (*color, 1.0)
            # Re-apply clear color on next frame
            gl_widget.renderer._bg_color_dirty = True
        else:
            # Find meshes by name prefix and change their color
            meshes = scene.collect_meshes()
            for mesh, _ in meshes:
                if target == "skull" and mesh.name and "cranium" in mesh.name:
                    mesh.material.color = color
                elif target == "face" and mesh.name == "face":
                    mesh.material.color = color

    # ── Align tab: alignment changes ──
    def on_alignment_changed(field: str = "", value: float = 0.0, **kw):
        from faceforge.anatomy.face import update_alignment
        face_group = named_nodes.get("faceGroup")
        if face_group is None:
            return
        # Gather current alignment from all sliders
        align_tab = window.control_panel.align_tab
        vals = align_tab.get_alignment()
        vals[field] = value  # Update the changed field
        update_alignment(
            face_group,
            scale=vals.get("scale", 1.14),
            offset_x=vals.get("offset_x", -0.2),
            offset_y=vals.get("offset_y", -10.6),
            offset_z=vals.get("offset_z", 9.5),
            rot_x_deg=vals.get("rot_x", 88.5),
        )

    # ── Labels ──
    def on_labels_toggled(enabled: bool = False, **kw):
        nonlocal _labels_enabled, _labels_dirty
        _labels_enabled = enabled
        _labels_dirty = True
        label_overlay.set_enabled(enabled)

    def _rebuild_labels():
        """Rebuild label list from currently visible meshes."""
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

    def _on_layer_toggled_labels(**kw):
        nonlocal _labels_dirty
        _labels_dirty = True

    # ── Clip plane ──
    def on_clip_plane_changed(enabled: bool = False, axis: str = "x",
                              offset: float = 0.0, flip: bool = False, **kw):
        renderer = gl_widget.renderer
        if not enabled:
            renderer.clear_clip_plane()
            return
        normal_map = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
        normal = list(normal_map.get(axis, (1, 0, 0)))
        if flip:
            normal = [-n for n in normal]
        renderer.set_clip_plane(tuple(normal), offset)

    # ── Skull mode switching ──
    def on_skull_mode_changed(mode: str = "original", **kw):
        """Rebuild skull hierarchy in the requested mode."""
        from faceforge.anatomy.skull import build_skull, get_jaw_pivot_node
        skull_grp = named_nodes.get("skullGroup")
        if skull_grp is None:
            return

        # Clear existing skull children
        for child in list(skull_grp.children):
            skull_grp.remove(child)

        # Rebuild
        new_skull, new_meshes, new_pivot = build_skull(assets, mode=mode)
        for child in list(new_skull.children):
            new_skull.remove(child)
            skull_grp.add(child)

        pipeline.skull_meshes = new_meshes
        pipeline.skull_mode = mode
        pipeline.jaw_pivot = new_pivot

        # Update global and system pivots
        set_jaw_pivot(*new_pivot)
        if pipeline.jaw_muscles is not None:
            pipeline.jaw_muscles.set_jaw_pivot(*new_pivot)
        if pipeline.head_rotation is not None:
            pipeline.head_rotation.set_head_pivot(*new_pivot)

        # Re-wire jaw pivot node
        simulation.jaw_pivot_node = get_jaw_pivot_node(skull_grp)

        # Apply current render mode to new meshes
        new_mesh_list = [m for m in new_meshes.values() if m is not None]
        _apply_current_render_mode(new_mesh_list)

        logging.getLogger(__name__).info("Skull mode switched to: %s, pivot: %s", mode, new_pivot)

    # ── On-demand body loading helpers ──
    skin_chain_ids: dict[str, int] = {}  # filled by load_assets(), read by on-demand loaders
    _after_registration_hooks: list = []  # callbacks run after soft tissue mesh registration
    _loaded_muscle_regions: set = set()
    _organs_loaded = False
    _vasculature_loaded = False
    _brain_loaded = False
    _skin_loaded = False

    # Mapping: muscle region → default skinning chains.
    # Use "arm" / "leg" as side-neutral tokens — resolved to the muscle's
    # own side (R or L) at registration time by _resolve_sided_chains().
    _MUSCLE_CHAIN_MAP: dict[str, list[str]] = {
        "back_muscles":     ["spine", "ribs"],
        "torso_muscles":    ["spine", "ribs"],
        "shoulder_muscles": ["spine", "arm"],
        "arm_muscles":      ["spine", "arm", "hand"],
        "hip_muscles":      ["spine", "leg"],
        "leg_muscles":      ["spine", "leg", "foot"],
    }

    # Per-muscle chain overrides — keyed by muscle name.
    # Muscles not listed here use their region's default from _MUSCLE_CHAIN_MAP.
    # Use "arm" / "leg" (side-neutral) — resolved per-muscle at load time.
    #
    # Anatomical rationale:
    #   "spine+ribs only" = rib cage structures that should NOT follow arm movement
    #   "spine+ribs+arm"  = muscles spanning from spine/ribs TO humerus/scapula
    _MUSCLE_CHAIN_OVERRIDES: dict[str, list[str]] = {}
    # -- shoulder_muscles overrides: rib-only muscles --
    for _n in ("Serratus Ant.", "Subclavius"):
        for _s in ("R", "L"):
            _MUSCLE_CHAIN_OVERRIDES[f"{_n} {_s}"] = ["spine", "ribs"]
    # -- torso_muscles overrides: pectorals insert on humerus, need arm chain --
    for _n in ("Pect. Major Clav.", "Pect. Major Stern.", "Pect. Major Abd.",
               "Pect. Minor"):
        for _s in ("R", "L"):
            _MUSCLE_CHAIN_OVERRIDES[f"{_n} {_s}"] = ["spine", "ribs", "arm"]
    # -- torso_muscles: pure rib/spine structures (no arm follow) --
    for _n in ("Ext. Intercostal", "Int. Intercostal", "Innermost Intercostal",
               "Diaphragm", "Linea Alba"):
        _MUSCLE_CHAIN_OVERRIDES[_n] = ["spine", "ribs"]
    for _n in ("Trans. Thoracis", "Lev. Costarum Longi", "Lev. Costarum Breves"):
        for _s in ("R", "L"):
            _MUSCLE_CHAIN_OVERRIDES[f"{_n} {_s}"] = ["spine", "ribs"]
    # -- back_muscles overrides: muscles that connect to shoulder/arm --
    for _n in ("Asc. Trapezius", "Trans. Trapezius", "Desc. Trapezius",
               "Latissimus Dorsi", "Rhomboid Major", "Rhomboid Minor"):
        for _s in ("R", "L"):
            _MUSCLE_CHAIN_OVERRIDES[f"{_n} {_s}"] = ["spine", "ribs", "arm"]
    # -- back_muscles: rib-attached posterior serratus --
    for _n in ("Serratus Post. Sup.", "Serratus Post. Inf."):
        for _s in ("R", "L"):
            _MUSCLE_CHAIN_OVERRIDES[f"{_n} {_s}"] = ["spine", "ribs"]
    # Clean up temp loop vars
    del _n, _s

    def _resolve_chain_set(chain_names: list[str]) -> set[int] | None:
        """Convert chain name list to chain ID set using skin_chain_ids."""
        chains = set()
        for cn in chain_names:
            cid = skin_chain_ids.get(cn)
            if cid is not None:
                chains.add(cid)
        return chains if chains else None

    def _resolve_sided_chains(chain_names: list[str], muscle_name: str) -> set[int] | None:
        """Resolve side-neutral chain tokens ("arm", "leg") to the muscle's side.

        A muscle name ending in " R" or " L" binds only to that side's limb
        chain (e.g. "arm" → "arm_R").  Midline muscles (no side suffix) bind
        to both sides as a fallback.
        """
        side = None
        if muscle_name.endswith(" R"):
            side = "R"
        elif muscle_name.endswith(" L"):
            side = "L"

        resolved: list[str] = []
        for cn in chain_names:
            if cn in ("arm", "leg"):
                if side is not None:
                    resolved.append(f"{cn}_{side}")
                else:
                    resolved.append(f"{cn}_R")
                    resolved.append(f"{cn}_L")
            elif cn in ("hand", "foot"):
                # Expand to all 5 digit chains per side
                sides = [side] if side is not None else ["R", "L"]
                for s in sides:
                    for digit in range(1, 6):
                        resolved.append(f"{cn}_{s}_{digit}")
            else:
                resolved.append(cn)
        return _resolve_chain_set(resolved)

    def _load_body_muscle_region(layer: str, config_name: str):
        nonlocal _loaded_muscle_regions
        if layer in _loaded_muscle_regions:
            return
        _loaded_muscle_regions.add(layer)
        body_root = named_nodes.get("bodyRoot")
        if body_root is None:
            return
        try:
            result = assets.load_body_muscles(config_name)
            body_root.add(result.group)
            visibility.register(layer, result.group)
            _apply_current_render_mode(result.meshes)

            # Load definitions to get muscle names for per-muscle chain assignment
            defs = load_muscle_config(config_name)

            # Default chain names for this region (may contain side-neutral tokens)
            default_chain_names = _MUSCLE_CHAIN_MAP.get(layer, ["spine"])

            # Register with soft tissue skinning — per-muscle chain selection.
            # Each muscle's side determines which limb chain it follows.
            if simulation.soft_tissue is not None:
                for mesh, defn in zip(result.meshes, defs):
                    muscle_name = defn.get("name", mesh.name)
                    override = _MUSCLE_CHAIN_OVERRIDES.get(muscle_name)
                    chain_names = override if override else default_chain_names
                    ac = _resolve_sided_chains(chain_names, muscle_name)
                    head_follow = defn.get("headFollow")
                    simulation.soft_tissue.register_skin_mesh(
                        mesh, is_muscle=True, allowed_chains=ac,
                        head_follow_config=head_follow,
                        muscle_name=muscle_name,
                    )
                    # Register bone attachment constraints (Layers 2-3)
                    origin_bones = defn.get("originBones")
                    insertion_bones = defn.get("insertionBones")
                    if (origin_bones and insertion_bones
                            and simulation.soft_tissue.attachment_system is not None):
                        binding = simulation.soft_tissue.bindings[-1]
                        fascia_regions = defn.get("fasciaRegions", [])
                        simulation.soft_tissue.attachment_system.register_muscle(
                            binding, origin_bones, insertion_bones,
                            fascia_regions=fascia_regions,
                        )

                # Remove digit/limb cross-chain blending for arm and leg
                # muscles.  Digit chain joints inherit all limb transforms
                # via scene graph (digit pivots are children of wrist/ankle),
                # so blending with a limb chain double-counts the parent
                # contribution.
                if layer in ("arm_muscles", "leg_muscles"):
                    digit_cids: set[int] = set()
                    prefix = "hand" if layer == "arm_muscles" else "foot"
                    for side in ("R", "L"):
                        for digit in range(1, 6):
                            cid = skin_chain_ids.get(f"{prefix}_{side}_{digit}")
                            if cid is not None:
                                digit_cids.add(cid)
                    if digit_cids:
                        simulation.soft_tissue.snap_hierarchy_blends(digit_cids)
                        simulation.soft_tissue.reassign_orphan_vertices(digit_cids)

            # Register muscles with physiology system (fasciculation)
            if simulation.physiology is not None:
                simulation.physiology.muscle_groups.append(result.group)
                for mesh, defn in zip(result.meshes, defs):
                    simulation.physiology.register_muscle(
                        mesh, defn.get("name", mesh.name),
                    )

            # Wire back-of-neck muscle pinning handler for back_muscles layer
            if layer == "back_muscles" and simulation.soft_tissue is not None:
                from faceforge.anatomy.back_neck_muscles import BackNeckMuscleHandler
                handler = BackNeckMuscleHandler()
                muscle_defs_map = {d["name"]: d for d in defs if "name" in d}
                handler.register(simulation.soft_tissue, muscle_defs_map)
                if handler.registered:
                    if simulation.fascia is not None:
                        handler.set_fascia_system(simulation.fascia)
                    simulation.back_neck_muscles = handler

            # Register individual nodes for per-item toggling
            items = []
            for node, defn in zip(result.nodes, defs):
                name = defn.get("name", node.name)
                toggle_id = f"muscle_{layer}_{name}"
                visibility.register(toggle_id, node)
                items.append({"toggle_id": toggle_id, "name": name})
            event_bus.publish(EventType.STRUCTURES_LOADED, group_id=layer, items=items)
            logging.getLogger(__name__).info("Loaded body muscles: %s (%d meshes)",
                                              layer, len(result.meshes))
            for hook in _after_registration_hooks:
                hook()
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to load %s: %s", layer, e)

    def _load_organs():
        nonlocal _organs_loaded
        if _organs_loaded:
            return
        _organs_loaded = True
        body_root = named_nodes.get("bodyRoot")
        if body_root is None:
            return
        try:
            result = assets.load_organs()
            body_root.add(result.group)
            visibility.register("organs", result.group)
            _apply_current_render_mode(result.meshes)
            # Register with soft tissue skinning — spine-only (organs don't follow limbs)
            if simulation.soft_tissue is not None:
                spine_id = skin_chain_ids.get("spine")
                ac = {spine_id} if spine_id is not None else None
                for mesh in result.meshes:
                    simulation.soft_tissue.register_skin_mesh(mesh, is_muscle=False, allowed_chains=ac)
            # Register individual organ nodes for per-item toggling
            defs = load_config("organs.json")
            items = []
            for node, defn in zip(result.nodes, defs):
                name = defn.get("name", node.name)
                category = defn.get("category", "")
                toggle_id = f"organ_{name}"
                visibility.register(toggle_id, node)
                items.append({"toggle_id": toggle_id, "name": name, "category": category})
            event_bus.publish(EventType.STRUCTURES_LOADED, group_id="organs", items=items)
            # Register organs with physiology system
            if simulation.physiology is not None:
                simulation.physiology.organ_group = result.group
                for mesh, defn in zip(result.meshes, defs):
                    simulation.physiology.register_organ(
                        mesh, defn.get("name", ""), defn.get("category", ""),
                    )
            logging.getLogger(__name__).info("Loaded organs: %d meshes", len(result.meshes))
            for hook in _after_registration_hooks:
                hook()
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to load organs: %s", e)

    def _load_vasculature():
        nonlocal _vasculature_loaded
        if _vasculature_loaded:
            return
        _vasculature_loaded = True
        body_root = named_nodes.get("bodyRoot")
        if body_root is None:
            return
        try:
            result = assets.load_vasculature()
            body_root.add(result.group)
            visibility.register("vasculature", result.group)
            _apply_current_render_mode(result.meshes)
            # Register with soft tissue skinning — spine-only
            if simulation.soft_tissue is not None:
                spine_id = skin_chain_ids.get("spine")
                ac = {spine_id} if spine_id is not None else None
                for mesh in result.meshes:
                    simulation.soft_tissue.register_skin_mesh(mesh, is_muscle=False, allowed_chains=ac)
            # Register individual vasculature nodes for per-item toggling
            defs = load_config("vascular.json")
            items = []
            for node, defn in zip(result.nodes, defs):
                name = defn.get("name", node.name)
                vtype = defn.get("type", "")
                toggle_id = f"vasc_{name}"
                visibility.register(toggle_id, node)
                items.append({"toggle_id": toggle_id, "name": name, "type": vtype})
            event_bus.publish(EventType.STRUCTURES_LOADED, group_id="vasculature", items=items)
            # Register vasculature with physiology system
            if simulation.physiology is not None:
                simulation.physiology.vascular_group = result.group
                for mesh, defn in zip(result.meshes, defs):
                    simulation.physiology.register_vascular(
                        mesh, defn.get("name", ""), defn.get("type", ""),
                    )
            logging.getLogger(__name__).info("Loaded vasculature: %d meshes", len(result.meshes))
            for hook in _after_registration_hooks:
                hook()
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to load vasculature: %s", e)

    def _load_brain():
        nonlocal _brain_loaded
        if _brain_loaded:
            return
        _brain_loaded = True
        # Brain goes under brainGroup (independent of skull visibility,
        # follows head rotation via explicit pivot rotation in HeadRotationSystem)
        brain_group = named_nodes.get("brainGroup")
        if brain_group is None:
            return
        try:
            result = assets.load_brain()
            brain_group.add(result.group)
            _apply_current_render_mode(result.meshes)
            # Register individual brain nodes for per-item toggling
            defs = load_config("brain.json")
            items = []
            for node, defn in zip(result.nodes, defs):
                name = defn.get("name", node.name)
                toggle_id = f"brain_{name}"
                visibility.register(toggle_id, node)
                items.append({"toggle_id": toggle_id, "name": name})
            event_bus.publish(EventType.STRUCTURES_LOADED, group_id="brain", items=items)
            logging.getLogger(__name__).info("Loaded brain: %d meshes", len(result.meshes))
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to load brain: %s", e)

    # ── Additional anatomy on-demand loading ──
    _pelvic_floor_loaded = False
    _ligaments_loaded = False
    _oral_loaded = False
    _cardiac_additional_loaded = False
    _intestinal_loaded = False
    _cns_additional_loaded = False
    _hand_muscles_loaded = False
    _foot_muscles_loaded = False

    def _load_generic_group(loader_method, layer_id, config_name, config_loader_fn,
                            parent_key="bodyRoot", toggle_prefix=""):
        """Generic on-demand loader for additional anatomy groups."""
        parent = named_nodes.get(parent_key)
        if parent is None:
            return
        try:
            result = loader_method()
            parent.add(result.group)
            visibility.register(layer_id, result.group)
            _apply_current_render_mode(result.meshes)
            defs = config_loader_fn(config_name)
            items = []
            for node, defn in zip(result.nodes, defs):
                name = defn.get("name", node.name)
                tid = f"{toggle_prefix}{name}" if toggle_prefix else f"{layer_id}_{name}"
                visibility.register(tid, node)
                item = {"toggle_id": tid, "name": name}
                cat = defn.get("category") or defn.get("type")
                if cat:
                    item["category"] = cat
                items.append(item)
            event_bus.publish(EventType.STRUCTURES_LOADED, group_id=layer_id, items=items)
            logging.getLogger(__name__).info("Loaded %s: %d meshes", layer_id, len(result.meshes))
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to load %s: %s", layer_id, e)

    def _load_hand_muscles():
        nonlocal _hand_muscles_loaded
        if _hand_muscles_loaded:
            return
        _hand_muscles_loaded = True
        body_root = named_nodes.get("bodyRoot")
        if body_root is None:
            return
        try:
            result = assets.load_hand_muscles()
            body_root.add(result.group)
            visibility.register("hand_muscles", result.group)
            _apply_current_render_mode(result.meshes)
            defs = load_muscle_config("hand_muscles.json")
            items = []
            for node, defn in zip(result.nodes, defs):
                name = defn.get("name", node.name)
                tid = f"muscle_hand_muscles_{name}"
                visibility.register(tid, node)
                items.append({"toggle_id": tid, "name": name})
            event_bus.publish(EventType.STRUCTURES_LOADED, group_id="hand_muscles", items=items)
            # Register with soft tissue skinning — digit chains ONLY.
            # Digit chain joints inherit wrist/elbow/shoulder transforms via
            # parent pivots, so digit chain deltas already include all arm
            # movement.  Excluding the arm chain prevents the extrapolated
            # wrist segment from competing with digit joints in the palm area.
            if simulation.soft_tissue is not None:
                hand_chains: set[int] = set()
                for side in ("R", "L"):
                    for digit in range(1, 6):
                        cid = skin_chain_ids.get(f"hand_{side}_{digit}")
                        if cid is not None:
                            hand_chains.add(cid)
                ac = hand_chains if hand_chains else None
                for mesh in result.meshes:
                    simulation.soft_tissue.register_skin_mesh(
                        mesh, is_muscle=True, allowed_chains=ac,
                    )
            logging.getLogger(__name__).info("Loaded hand muscles: %d meshes", len(result.meshes))
            for hook in _after_registration_hooks:
                hook()
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to load hand muscles: %s", e)

    def _load_foot_muscles():
        nonlocal _foot_muscles_loaded
        if _foot_muscles_loaded:
            return
        _foot_muscles_loaded = True
        body_root = named_nodes.get("bodyRoot")
        if body_root is None:
            return
        try:
            result = assets.load_foot_muscles()
            body_root.add(result.group)
            visibility.register("foot_muscles", result.group)
            _apply_current_render_mode(result.meshes)
            defs = load_muscle_config("foot_muscles.json")
            items = []
            for node, defn in zip(result.nodes, defs):
                name = defn.get("name", node.name)
                tid = f"muscle_foot_muscles_{name}"
                visibility.register(tid, node)
                items.append({"toggle_id": tid, "name": name})
            event_bus.publish(EventType.STRUCTURES_LOADED, group_id="foot_muscles", items=items)
            # Register with soft tissue skinning — digit chains ONLY.
            # Digit chain joints inherit ankle/knee/hip transforms via parent
            # pivots, so digit chain deltas already include all leg movement.
            # Excluding the leg chain prevents the extrapolated ankle
            # segment from competing with digit joints in the foot area.
            if simulation.soft_tissue is not None:
                foot_chains: set[int] = set()
                for side in ("R", "L"):
                    for digit in range(1, 6):
                        cid = skin_chain_ids.get(f"foot_{side}_{digit}")
                        if cid is not None:
                            foot_chains.add(cid)
                ac = foot_chains if foot_chains else None
                for mesh in result.meshes:
                    simulation.soft_tissue.register_skin_mesh(
                        mesh, is_muscle=True, allowed_chains=ac,
                    )
            logging.getLogger(__name__).info("Loaded foot muscles: %d meshes", len(result.meshes))
            for hook in _after_registration_hooks:
                hook()
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to load foot muscles: %s", e)

    def _load_pelvic_floor():
        nonlocal _pelvic_floor_loaded
        if _pelvic_floor_loaded:
            return
        _pelvic_floor_loaded = True
        _load_generic_group(assets.load_pelvic_floor, "pelvic_floor",
                            "pelvic_floor.json", load_config)

    # Ligament category → chain name mapping.
    # Each category specifies which kinematic chains the ligament follows.
    # Side-neutral tokens ("arm", "leg") are resolved per-structure at load time.
    _LIGAMENT_CHAIN_MAP: dict[str, list[str]] = {
        "upper_limb": ["spine", "arm"],
        "lower_limb": ["spine", "leg"],
        "trunk":      ["spine", "ribs"],
        "hip":        ["spine", "leg"],
    }

    def _load_ligaments():
        nonlocal _ligaments_loaded
        if _ligaments_loaded:
            return
        _ligaments_loaded = True
        body_root = named_nodes.get("bodyRoot")
        if body_root is None:
            return
        try:
            result = assets.load_ligaments()
            body_root.add(result.group)
            visibility.register("ligaments", result.group)
            _apply_current_render_mode(result.meshes)
            defs = load_config("ligaments.json")

            # Register with soft tissue skinning using per-structure chain assignment
            if simulation.soft_tissue is not None:
                for mesh, defn in zip(result.meshes, defs):
                    cat = defn.get("category", "trunk")
                    lig_name = defn.get("name", mesh.name)
                    chain_names = _LIGAMENT_CHAIN_MAP.get(cat, ["spine"])
                    ac = _resolve_sided_chains(chain_names, lig_name)
                    simulation.soft_tissue.register_skin_mesh(
                        mesh, is_muscle=True, allowed_chains=ac,
                        muscle_name=lig_name,
                    )
                    # Register ligament bone attachments (Layer 5)
                    origin_bones = defn.get("originBones")
                    insertion_bones = defn.get("insertionBones")
                    if (origin_bones and insertion_bones
                            and simulation.soft_tissue.attachment_system is not None):
                        binding = simulation.soft_tissue.bindings[-1]
                        simulation.soft_tissue.attachment_system.register_muscle(
                            binding, origin_bones, insertion_bones,
                        )

            # Register individual nodes for per-item toggling
            items = []
            for node, defn in zip(result.nodes, defs):
                name = defn.get("name", node.name)
                cat = defn.get("category", "")
                tid = f"ligaments_{name}"
                visibility.register(tid, node)
                items.append({"toggle_id": tid, "name": name, "category": cat})
            event_bus.publish(EventType.STRUCTURES_LOADED, group_id="ligaments", items=items)
            logging.getLogger(__name__).info("Loaded ligaments: %d meshes", len(result.meshes))
            for hook in _after_registration_hooks:
                hook()
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to load ligaments: %s", e)

    def _load_oral():
        nonlocal _oral_loaded
        if _oral_loaded:
            return
        _oral_loaded = True
        _load_generic_group(assets.load_oral, "oral",
                            "oral.json", load_config)

    def _load_cardiac_additional():
        nonlocal _cardiac_additional_loaded
        if _cardiac_additional_loaded:
            return
        _cardiac_additional_loaded = True
        _load_generic_group(assets.load_cardiac_additional, "cardiac_additional",
                            "cardiac_additional.json", load_config)

    def _load_intestinal():
        nonlocal _intestinal_loaded
        if _intestinal_loaded:
            return
        _intestinal_loaded = True
        _load_generic_group(assets.load_intestinal, "intestinal",
                            "intestinal.json", load_config)

    def _load_cns_additional():
        nonlocal _cns_additional_loaded
        if _cns_additional_loaded:
            return
        _cns_additional_loaded = True
        _load_generic_group(assets.load_cns_additional, "cns_additional",
                            "cns_additional.json", load_config,
                            parent_key="brainGroup")

    _loaded_single_stls: set = set()

    def _load_single_stl(layer_id: str, stl_name: str):
        """Load a single STL file as an on-demand toggle."""
        nonlocal _loaded_single_stls
        if layer_id in _loaded_single_stls:
            return
        _loaded_single_stls.add(layer_id)
        body_root = named_nodes.get("bodyRoot")
        if body_root is None:
            return
        try:
            from faceforge.loaders.stl_batch_loader import load_stl_batch
            result = load_stl_batch(
                [{"name": layer_id, "stl": stl_name, "color": 0xcccccc}],
                label=layer_id,
                transform=assets.transform,
                stl_dir=assets.stl_dir,
            )
            body_root.add(result.group)
            visibility.register(layer_id, result.group)
            _apply_current_render_mode(result.meshes)
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to load %s: %s", layer_id, e)

    def _load_skin():
        nonlocal _skin_loaded
        if _skin_loaded:
            return
        _skin_loaded = True
        body_root = named_nodes.get("bodyRoot")
        if body_root is None:
            return
        try:
            result = assets.load_skin()
            body_root.add(result.group)
            visibility.register("skin", result.group)
            _apply_current_render_mode(result.meshes)
            # Register skin with soft tissue skinning — all chains so it
            # deforms with spine, limbs, digits, and breathing.
            # Two-tier spatial filtering prevents cross-region binding:
            # 1. chain_z_margin: Z-axis AABB with proportional margin per chain
            #    (small chains like hands/feet get tight margins)
            # 2. spatial_limit: Euclidean distance guard catches remaining
            #    overlap where arm and leg chains meet at the hip level
            if simulation.soft_tissue is not None:
                all_chains = set(skin_chain_ids.values()) if skin_chain_ids else None
                for mesh in result.meshes:
                    simulation.soft_tissue.register_skin_mesh(
                        mesh, is_muscle=False, allowed_chains=all_chains,
                        chain_z_margin=15.0,
                        spatial_limit=25.0,
                    )
            logging.getLogger(__name__).info("Loaded skin: %d meshes", len(result.meshes))
            for hook in _after_registration_hooks:
                hook()
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to load skin: %s", e)

    event_bus.subscribe(EventType.AU_CHANGED, on_au_changed)
    event_bus.subscribe(EventType.EXPRESSION_SET, on_expression_set)
    event_bus.subscribe(EventType.HEAD_ROTATION_CHANGED, on_head_changed)
    event_bus.subscribe(EventType.BODY_STATE_CHANGED, on_body_changed)
    event_bus.subscribe(EventType.BODY_POSE_SET, on_body_pose_set)
    event_bus.subscribe(EventType.GENDER_CHANGED, on_gender_changed)
    event_bus.subscribe(EventType.GENDER_RELEASED, on_gender_released)
    event_bus.subscribe(EventType.LAYER_TOGGLED, on_layer_toggled)
    event_bus.subscribe(EventType.AUTO_BLINK_TOGGLED, on_auto_blink)
    event_bus.subscribe(EventType.AUTO_BREATHING_TOGGLED, on_auto_breathing)
    event_bus.subscribe(EventType.EYE_TRACKING_TOGGLED, on_eye_tracking)
    event_bus.subscribe(EventType.MICRO_EXPRESSIONS_TOGGLED, on_micro_expressions)
    event_bus.subscribe(EventType.RENDER_MODE_CHANGED, on_render_mode_changed)
    event_bus.subscribe(EventType.CAMERA_PRESET, on_camera_preset)
    event_bus.subscribe(EventType.SCENE_MODE_TOGGLED, on_scene_mode_toggled)
    event_bus.subscribe(EventType.SCENE_CAMERA_CHANGED, on_scene_camera_changed)
    event_bus.subscribe(EventType.SCENE_WRAPPER_NUDGE, on_wrapper_nudge)
    event_bus.subscribe(EventType.ANIM_PLAY, on_anim_play)
    event_bus.subscribe(EventType.ANIM_PAUSE, on_anim_pause)
    event_bus.subscribe(EventType.ANIM_STOP, on_anim_stop)
    event_bus.subscribe(EventType.ANIM_SEEK, on_anim_seek)
    event_bus.subscribe(EventType.ANIM_SPEED, on_anim_speed)
    event_bus.subscribe(EventType.ANIM_CLIP_SELECTED, on_anim_clip_selected)
    event_bus.subscribe(EventType.COLOR_CHANGED, on_color_changed)
    event_bus.subscribe(EventType.ALIGNMENT_CHANGED, on_alignment_changed)
    event_bus.subscribe(EventType.LABELS_TOGGLED, on_labels_toggled)
    event_bus.subscribe(EventType.LAYER_TOGGLED, _on_layer_toggled_labels)
    event_bus.subscribe(EventType.CLIP_PLANE_CHANGED, on_clip_plane_changed)

    # Wire label style controls → label overlay
    def _on_label_style_changed():
        style = window.control_panel.display_tab.get_label_style()
        label_overlay.apply_style(style)

    window.control_panel.display_tab.label_style_changed.connect(_on_label_style_changed)
    event_bus.subscribe(EventType.SKULL_MODE_CHANGED, on_skull_mode_changed)
    event_bus.subscribe(EventType.EYE_COLOR_SET, on_eye_color_set)
    event_bus.subscribe(
        EventType.STRUCTURES_LOADED,
        window.control_panel.layers_tab.on_structures_loaded,
    )

    # ── Feature 1: Muscle Activation Heatmap ──
    from faceforge.body.muscle_activation import MuscleActivationSystem
    _muscle_activation = MuscleActivationSystem()
    simulation.muscle_activation = _muscle_activation

    def on_heatmap_toggled(enabled: bool = False, **kw):
        _muscle_activation.set_enabled(enabled)

    event_bus.subscribe(EventType.HEATMAP_TOGGLED, on_heatmap_toggled)

    # ── Feature 2: Natural Language Structure Search ──
    from faceforge.anatomy.structure_search import AnatomySearchIndex
    _search_index = AnatomySearchIndex()
    _search_highlighting = False

    def on_structure_search(query: str = "", **kw):
        nonlocal _search_highlighting
        if not query:
            if _search_highlighting:
                # Restore all opacities
                for mesh, _ in scene.collect_meshes():
                    mesh.material.opacity = 1.0
                _search_highlighting = False
            return
        results = _search_index.search(query)
        if not results:
            return
        matched_names = {r.mesh_name for r in results}
        # Dim non-matching meshes
        for mesh, _ in scene.collect_meshes():
            if mesh.name in matched_names:
                mesh.material.opacity = 1.0
            else:
                mesh.material.opacity = 0.15
        _search_highlighting = True

    event_bus.subscribe(EventType.STRUCTURE_SEARCH, on_structure_search)

    # ── Feature 3: Physiological Simulation Overlays (particles) ──
    from faceforge.rendering.particle_system import ParticleSystem

    _blood_particles = ParticleSystem(max_particles=3000)
    _neural_particles = ParticleSystem(max_particles=2000)

    def on_physiology_blood_flow(enabled: bool = False, **kw):
        if not enabled:
            _blood_particles.clear()

    def on_physiology_neural(enabled: bool = False, **kw):
        if not enabled:
            _neural_particles.clear()

    # ── Feature 4: Speech & Phoneme Animation ──
    from faceforge.animation.speech import SpeechEngine
    _speech_engine = SpeechEngine()
    _speech_playing = False
    _speech_sequence = []
    _speech_time = 0.0

    def on_speech_play(text: str = "", speed: float = 1.0, **kw):
        nonlocal _speech_playing, _speech_sequence, _speech_time
        if not text:
            return
        _speech_sequence = _speech_engine.generate_au_sequence(text, speed=speed)
        _speech_time = 0.0
        _speech_playing = True

    event_bus.subscribe(EventType.SPEECH_PLAY, on_speech_play)

    # ── Feature 5: Video/GIF Export ──
    from faceforge.export.video_export import VideoExporter
    _video_exporter = VideoExporter(gl_widget)

    _export_dialog = None

    def _on_export_requested():
        nonlocal _export_dialog
        from faceforge.ui.export_dialog import ExportDialog
        if _export_dialog is None:
            _export_dialog = ExportDialog(window)

            def _do_export(config):
                mode = config.get("mode", "turntable")
                fps = config.get("fps", 30)
                duration = config.get("duration", 10.0)
                output = config.get("output_path", "export.mp4")
                w = config.get("width", gl_widget.width())
                h = config.get("height", gl_widget.height())

                if mode == "screenshot":
                    _video_exporter.export_screenshot(output, width=w, height=h)
                elif mode == "turntable":
                    _video_exporter.export_turntable(
                        output, duration=duration, fps=fps,
                        width=w, height=h,
                    )
                elif mode == "animation":
                    _video_exporter.export_animation(
                        anim_player, simulation, output, fps=fps,
                        width=w, height=h,
                    )

            _export_dialog.export_requested.connect(_do_export)
        _export_dialog.show()

    # ── Feature 6: Interactive Anatomy Quiz ──
    from faceforge.anatomy.quiz_engine import QuizEngine
    _quiz_engine = QuizEngine(_search_index)
    _quiz_dialog = None

    def _on_quiz_requested():
        nonlocal _quiz_dialog
        from faceforge.ui.quiz_dialog import QuizDialog
        if _quiz_dialog is None:
            _quiz_dialog = QuizDialog(_quiz_engine, window)

            def _on_highlight(mesh_name):
                for mesh, _ in scene.collect_meshes():
                    if mesh.name == mesh_name:
                        mesh.material.opacity = 1.0
                    else:
                        mesh.material.opacity = 0.2

            def _on_clear():
                for mesh, _ in scene.collect_meshes():
                    mesh.material.opacity = 1.0

            def _on_quiz_click_mode(enabled):
                gl_widget.quiz_click_mode = enabled
                if enabled:
                    def _on_mesh_clicked(name):
                        if _quiz_dialog is not None:
                            _quiz_dialog.on_mesh_clicked(name)
                    gl_widget.quiz_click_callback = _on_mesh_clicked
                else:
                    gl_widget.quiz_click_callback = None

            _quiz_dialog.highlight_requested.connect(_on_highlight)
            _quiz_dialog.clear_highlight.connect(_on_clear)
            _quiz_dialog.quiz_click_mode.connect(_on_quiz_click_mode)

        _quiz_dialog.show()

    # ── Feature 7: Pathology Visualization ──
    from faceforge.anatomy.pathology import PathologySystem
    _pathology = PathologySystem()
    simulation.pathology = _pathology

    def on_pathology_changed(condition: str = "none", target: str = "",
                             severity: float = 0.0, **kw):
        if condition == "none":
            _pathology.clear_all()
            return
        _pathology.remove_condition(target)
        if severity > 0:
            _pathology.add_condition(target, condition, severity)

    event_bus.subscribe(EventType.PATHOLOGY_CHANGED, on_pathology_changed)

    # ── Feature 8: Custom Animation Timeline Editor ──
    _timeline_editor = None

    def _on_timeline_requested():
        nonlocal _timeline_editor
        from faceforge.ui.timeline_editor import TimelineEditor
        if _timeline_editor is None:
            _timeline_editor = TimelineEditor(window)
            _timeline_editor.set_animation_player(anim_player)
            _timeline_editor.set_state_refs(state, gl_widget.camera)
        _timeline_editor.show()

    # ── Feature 9: Comparative Anatomy Views ──
    _comparison_dialog = None

    def _on_compare_toggled(checked):
        nonlocal _comparison_dialog
        if checked:
            from faceforge.ui.comparison_dialog import ComparisonDialog
            if _comparison_dialog is None:
                _comparison_dialog = ComparisonDialog(window)

                def _on_config_changed(left_cfg, right_cfg):
                    gl_widget.comparison_left_config = left_cfg
                    gl_widget.comparison_right_config = right_cfg

                _comparison_dialog.config_changed.connect(_on_config_changed)

            gl_widget.comparison_mode = True
            _comparison_dialog.show()
        else:
            gl_widget.comparison_mode = False
            if _comparison_dialog is not None:
                _comparison_dialog.hide()

    # Wire display tab tool buttons
    display_tab = window.control_panel.display_tab
    if hasattr(display_tab, '_export_btn'):
        display_tab._export_btn.clicked.connect(_on_export_requested)
    if hasattr(display_tab, '_quiz_btn'):
        display_tab._quiz_btn.clicked.connect(_on_quiz_requested)
    if hasattr(display_tab, '_timeline_btn'):
        display_tab._timeline_btn.clicked.connect(_on_timeline_requested)
    if hasattr(display_tab, '_compare_btn'):
        display_tab._compare_btn.toggled.connect(_on_compare_toggled)

    # Loading pipeline
    pipeline = LoadingPipeline(assets, event_bus, named_nodes)

    def load_assets():
        """Load all assets. Called after GL is initialized."""
        try:
            pipeline.load_head()
        except Exception as e:
            print(f"Warning: Head loading incomplete: {e}")

        # Wire anatomy systems to simulation
        simulation.facs_engine = pipeline.facs_engine
        simulation.jaw_muscles = pipeline.jaw_muscles
        simulation.expression_muscles = pipeline.expression_muscles
        simulation.face_features = pipeline.face_features
        simulation.head_rotation = pipeline.head_rotation
        simulation.neck_muscles = pipeline.neck_muscles
        simulation.neck_constraints = pipeline.neck_constraints
        simulation.vertebrae_pivots = pipeline.vertebrae_pivots

        # Wire scene node references
        simulation.skull_group = named_nodes.get("skullGroup")
        simulation.face_group = named_nodes.get("faceGroup")

        # Wire brain group for independent head rotation
        simulation.brain_group = named_nodes.get("brainGroup")

        # Wire visibility group references for gating expensive updates
        simulation.jaw_muscle_group = named_nodes.get("stlMuscleGroup")
        simulation.expr_muscle_group = named_nodes.get("exprMuscleGroup")
        simulation.platysma_group = named_nodes.get("platysmaGroup")
        simulation.neck_muscle_group = named_nodes.get("neckMuscleGroup")
        simulation.face_feature_group = named_nodes.get("faceFeatureGroup")

        # Wire jawPivot node so simulation can rotate it
        skull_grp = named_nodes.get("skullGroup")
        if skull_grp is not None:
            from faceforge.anatomy.skull import get_jaw_pivot_node
            simulation.jaw_pivot_node = get_jaw_pivot_node(skull_grp)

        # Load body skeleton
        try:
            pipeline.load_body_skeleton()
        except Exception as e:
            print(f"Warning: Body skeleton loading incomplete: {e}")

        # Register body skeleton groups with visibility manager
        # After setup_from_skeleton(), bones are reparented under pivot nodes,
        # so original groups (upper_limb, lower_limb, hand, foot) may be empty.
        # Register both the original groups AND the pivot nodes.
        if pipeline.skeleton is not None:
            skel_toggle_map = {
                "thoracic": "thoracic",
                "lumbar": "lumbar",
                "ribs": "ribs",
                "pelvis": "pelvis",
            }
            for group_key, toggle_name in skel_toggle_map.items():
                grp = pipeline.skeleton.groups.get(group_key)
                if grp is not None:
                    visibility.register(toggle_name, grp)

        # Register limb pivot nodes for visibility toggling
        # JS: upper_limb toggle controls shoulder pivots (which contain all arm bones)
        # JS: lower_limb toggle controls hip pivots (which contain all leg bones)
        # JS: hands toggle controls wrist/elbow pivots' hand children
        # JS: feet toggle controls ankle/knee pivots' foot children
        if pipeline.joint_setup is not None:
            jp = pipeline.joint_setup.pivots
            # Upper limb: shoulder pivots contain humerus→elbow→wrist chain
            for side in ("R", "L"):
                sp = jp.get(f"shoulder_{side}")
                if sp is not None:
                    visibility.register("upper_limb_skel", sp)
            # Lower limb: hip pivots contain femur→knee→ankle chain
            for side in ("R", "L"):
                hp = jp.get(f"hip_{side}")
                if hp is not None:
                    visibility.register("lower_limb_skel", hp)
            # Hands: wrist pivots (or elbow if no wrist) contain hand bones
            for side in ("R", "L"):
                wp = jp.get(f"wrist_{side}")
                if wp is not None:
                    visibility.register("hands_skel", wp)
            # Feet: ankle pivots contain foot bones
            for side in ("R", "L"):
                ap = jp.get(f"ankle_{side}")
                if ap is not None:
                    visibility.register("feet_skel", ap)

        # Wire body animation
        if pipeline.joint_setup is not None:
            body_anim = BodyAnimationSystem(pipeline.joint_setup)
            body_anim.load_fractions()
            if pipeline.skeleton is not None:
                body_anim.set_thoracic_pivots(
                    pipeline.skeleton.pivots.get("thoracic", [])
                )
                body_anim.set_lumbar_pivots(
                    pipeline.skeleton.pivots.get("lumbar", [])
                )
                # Wire rib nodes for breathing animation
                if pipeline.skeleton.rib_nodes:
                    body_anim.set_rib_nodes(pipeline.skeleton.rib_nodes)
                    logging.getLogger(__name__).info(
                        "Rib nodes wired: %d nodes for breathing",
                        len(pipeline.skeleton.rib_nodes),
                    )
            simulation.body_animation = body_anim

        # Wire bone anchors, platysma, and fascia to simulation
        simulation.bone_anchors = pipeline.bone_anchors
        simulation.platysma = pipeline.platysma
        simulation.fascia = pipeline.fascia

        # Create and wire MuscleAttachmentSystem (Layers 2-3)
        if pipeline.bone_anchors is not None and simulation.soft_tissue is not None:
            from faceforge.anatomy.muscle_attachments import MuscleAttachmentSystem
            attachment_sys = MuscleAttachmentSystem(pipeline.bone_anchors)
            simulation.soft_tissue.attachment_system = attachment_sys

        # Create and wire BoneCollisionSystem (Layer 4)
        if pipeline.bone_anchors is not None and simulation.soft_tissue is not None:
            from faceforge.anatomy.bone_collision import BoneCollisionSystem
            collision_sys = BoneCollisionSystem(pipeline.bone_anchors)
            n_capsules = collision_sys.build_capsules()
            if n_capsules > 0:
                simulation.soft_tissue.collision_system = collision_sys

        # Initialize neck body anchor rest positions for body-delta tracking
        simulation.init_neck_body_anchors()

        # Body joint constraints
        body_constraints = BodyConstraints()
        body_constraints.load()
        simulation.body_constraints = body_constraints

        # Soft tissue skinning
        skinning = SoftTissueSkinning()
        simulation.soft_tissue = skinning

        def _build_joint_chains() -> list:
            """Build kinematic joint chains from skeleton and joint pivots.

            Populates the outer skin_chain_ids dict and returns the chain list.
            Convention: chain 0=spine, 1=arm_R, 2=leg_R, 3=arm_L, 4=leg_L, etc.
            """
            chains: list[list[tuple[str, SceneNode]]] = []
            skin_chain_ids.clear()

            # Chain 0: Spine (thoracic top→bottom → lumbar)
            spine_chain: list[tuple[str, SceneNode]] = []
            if pipeline.skeleton is not None:
                for pinfo in pipeline.skeleton.pivots.get("thoracic", []):
                    spine_chain.append((f"thoracic_{pinfo.get('level', 0)}", pinfo["group"]))
                for pinfo in pipeline.skeleton.pivots.get("lumbar", []):
                    spine_chain.append((f"lumbar_{pinfo.get('level', 0)}", pinfo["group"]))
            if spine_chain:
                skin_chain_ids["spine"] = len(chains)
                chains.append(spine_chain)

            # Limb chains (single 3-joint chain per limb)
            if pipeline.joint_setup is not None:
                jp = pipeline.joint_setup.pivots
                for side in ("R", "L"):
                    arm_chain: list[tuple[str, SceneNode]] = []
                    for jn in ("shoulder", "elbow", "wrist"):
                        node = jp.get(f"{jn}_{side}")
                        if node is not None:
                            arm_chain.append((f"{jn}_{side}", node))
                    if arm_chain:
                        skin_chain_ids[f"arm_{side}"] = len(chains)
                        chains.append(arm_chain)

                    leg_chain: list[tuple[str, SceneNode]] = []
                    for jn in ("hip", "knee", "ankle"):
                        node = jp.get(f"{jn}_{side}")
                        if node is not None:
                            leg_chain.append((f"{jn}_{side}", node))
                    if leg_chain:
                        skin_chain_ids[f"leg_{side}"] = len(chains)
                        chains.append(leg_chain)

            # Digit chains (one chain per digit per side)
            n_hand_chains = 0
            n_foot_chains = 0
            if pipeline.joint_setup is not None:
                jp = pipeline.joint_setup.pivots
                for side in ("R", "L"):
                    for digit in range(1, 6):
                        hand_chain: list[tuple[str, SceneNode]] = []
                        for seg in ("mc", "prox", "mid", "dist"):
                            p = jp.get(f"finger_{side}_{digit}_{seg}")
                            if p is not None:
                                hand_chain.append((f"finger_{side}_{digit}_{seg}", p))
                        if hand_chain:
                            skin_chain_ids[f"hand_{side}_{digit}"] = len(chains)
                            chains.append(hand_chain)
                            n_hand_chains += 1

                        foot_chain: list[tuple[str, SceneNode]] = []
                        for seg in ("mt", "prox", "mid", "dist"):
                            p = jp.get(f"toe_{side}_{digit}_{seg}")
                            if p is not None:
                                foot_chain.append((f"toe_{side}_{digit}_{seg}", p))
                        if foot_chain:
                            skin_chain_ids[f"foot_{side}_{digit}"] = len(chains)
                            chains.append(foot_chain)
                            n_foot_chains += 1
            logging.getLogger(__name__).info(
                "Digit chains built: %d hand, %d foot", n_hand_chains, n_foot_chains,
            )

            # Chain: ribs (one pivot per rib, for rib-attached muscles)
            if simulation.body_animation is not None and simulation.body_animation._rib_pivots:
                rib_chain: list[tuple[str, SceneNode]] = []
                for i, pivot in enumerate(simulation.body_animation._rib_pivots):
                    rib_chain.append((f"rib_{i}", pivot))
                if rib_chain:
                    skin_chain_ids["ribs"] = len(chains)
                    chains.append(rib_chain)
                    logging.getLogger(__name__).info(
                        "Rib skinning chain added: %d rib pivots", len(rib_chain),
                    )

            return chains

        joint_chains = _build_joint_chains()
        _gender_helpers["build_chains"] = _build_joint_chains

        if joint_chains:
            # Force a scene graph update so rest matrices are correct
            scene.update()
            skinning.build_skin_joints(joint_chains)
            logging.getLogger(__name__).info(
                "Skin joints built: %d joints in %d chains",
                len(skinning.joints), len(joint_chains),
            )

        # Wire skinning diagnostic callback to debug tab
        def _run_skinning_diagnostic() -> str:
            from faceforge.body.diagnostics import SkinningDiagnostic
            if simulation.soft_tissue is None or not simulation.soft_tissue.bindings:
                return "No soft tissue bindings registered."
            diag = SkinningDiagnostic(simulation.soft_tissue)
            reports = diag.analyze_bindings()
            anomalies = diag.check_displacements(max_displacement=5.0, relative=True)
            distortion = diag.check_mesh_distortion()
            static_verts = diag.check_static_vertices()
            neighbor_stretch = diag.check_neighbor_stretch(max_stretch=3.0)
            return diag.format_report(
                reports, anomalies,
                distortion=distortion,
                static_verts=static_verts,
                neighbor_stretch=neighbor_stretch,
            )

        window.control_panel.debug_tab.set_diagnostic_callback(_run_skinning_diagnostic)

        # ── Stretch/chain visualization + selection + reassignment ──
        from faceforge.body.stretch_viz import StretchVisualizer
        from faceforge.body.chain_reassignment import ChainReassigner
        from faceforge.body.chain_overrides import (
            save_overrides, load_overrides, apply_overrides,
            collect_modified_overrides,
        )
        from faceforge.rendering.selection_tool import SelectionTool

        stretch_viz = StretchVisualizer(skinning)
        reassigner = ChainReassigner(skinning)
        selection_tool = SelectionTool(skinning)
        gl_widget.selection_tool = selection_tool
        print(f"[DEBUG] Created StretchVisualizer, ChainReassigner, SelectionTool"
              f" (skinning id={id(skinning)}, bindings={len(skinning.bindings)})")

        # Track which vertices have been modified for override saving
        _modified_vertices: dict[int, set[int]] = {}

        # Populate chain names in debug tab
        chain_names: list[str] = []
        seen_chains: set[int] = set()
        for ji, joint in enumerate(skinning.joints):
            if joint.chain_id not in seen_chains:
                seen_chains.add(joint.chain_id)
                # Derive friendly name from first joint in chain
                name = joint.name.rsplit("_", 1)[0] if "_" in joint.name else joint.name
                chain_names.append(f"{joint.chain_id}: {name}")
        window.control_panel.debug_tab.set_chain_names(chain_names)

        # Map chain name back to chain ID
        def _chain_name_to_id(name: str) -> int:
            try:
                return int(name.split(":")[0])
            except (ValueError, IndexError):
                return 0

        # Wire visualization toggles
        debug_tab = window.control_panel.debug_tab

        def _on_stretch_viz(enabled):
            print(f"[DEBUG] _on_stretch_viz({enabled}), bindings={len(skinning.bindings)}")
            stretch_viz.stretch_enabled = enabled
            if not enabled:
                stretch_viz.chain_enabled = False
            else:
                # Apply colors immediately (don't wait for simulation tick)
                stretch_viz.update()
            print(f"[DEBUG]   stretch_enabled={stretch_viz.stretch_enabled}, chain_enabled={stretch_viz.chain_enabled}")

        def _on_chain_viz(enabled):
            print(f"[DEBUG] _on_chain_viz({enabled}), bindings={len(skinning.bindings)}")
            stretch_viz.chain_enabled = enabled
            if not enabled:
                stretch_viz.stretch_enabled = False
            else:
                # Apply colors immediately (don't wait for simulation tick)
                stretch_viz.update()
            print(f"[DEBUG]   stretch_enabled={stretch_viz.stretch_enabled}, chain_enabled={stretch_viz.chain_enabled}")

        debug_tab.stretch_viz_toggled.connect(_on_stretch_viz)
        debug_tab.chain_viz_toggled.connect(_on_chain_viz)
        print(f"[DEBUG] Connected stretch_viz_toggled and chain_viz_toggled signals")

        # Wire selection tool
        def _on_selection_mode(enabled):
            selection_tool.active = enabled

        def _on_selection_changed():
            count = selection_tool.selection.total_count
            debug_tab.update_selection_count(count)

        selection_tool.on_selection_changed = _on_selection_changed
        debug_tab.selection_mode_toggled.connect(_on_selection_mode)

        def _on_clear_selection():
            selection_tool.selection.clear()
            _on_selection_changed()

        debug_tab.clear_selection_clicked.connect(_on_clear_selection)

        # Wire reassignment
        def _on_reassign(chain_name_str):
            chain_id = _chain_name_to_id(chain_name_str)
            total = 0
            for bi, vis in selection_tool.selection.get_flat_indices():
                count = reassigner.reassign(bi, vis, chain_id)
                total += count
                # Track modified vertices
                if bi not in _modified_vertices:
                    _modified_vertices[bi] = set()
                _modified_vertices[bi].update(vis)
            if total > 0:
                stretch_viz.invalidate_chain_cache()
                debug_tab.set_undo_enabled(reassigner.can_undo)
                print(f"[FaceForge] Reassigned {total} vertices to chain {chain_id}")

        debug_tab.reassign_clicked.connect(_on_reassign)

        # Wire undo
        def _on_undo():
            if reassigner.undo():
                stretch_viz.invalidate_chain_cache()
                debug_tab.set_undo_enabled(reassigner.can_undo)
                print("[FaceForge] Undo reassignment")

        debug_tab.undo_clicked.connect(_on_undo)

        # Wire override save/load
        def _on_save_overrides():
            overrides = collect_modified_overrides(skinning, _modified_vertices)
            if overrides:
                path = save_overrides(skinning, overrides)
                count = sum(len(v) for v in overrides.values())
                debug_tab.set_override_count(count)
                print(f"[FaceForge] Saved {count} overrides to {path}")
            else:
                print("[FaceForge] No modified vertices to save")

        def _on_load_overrides():
            overrides = load_overrides()
            if overrides:
                count = apply_overrides(skinning, overrides)
                if count > 0:
                    debug_tab.set_override_count(count)
                    stretch_viz.invalidate_chain_cache()
                    skinning._last_signature = ()
                    print(f"[FaceForge] Loaded and applied {count} overrides")
                else:
                    mesh_names = {b.mesh.name for b in skinning.bindings}
                    override_names = set(overrides.keys())
                    missing = override_names - mesh_names
                    if missing:
                        print(f"[FaceForge] 0 overrides applied — mesh layers not loaded: {missing}"
                              " (enable the Skin layer first)")
                    else:
                        print("[FaceForge] 0 overrides applied — no matching vertices")
            else:
                print("[FaceForge] No overrides file found")

        debug_tab.save_overrides_clicked.connect(_on_save_overrides)
        debug_tab.load_overrides_clicked.connect(_on_load_overrides)

        # Load overrides on startup (if file exists).
        # Overrides are applied event-driven: whenever a layer loads and
        # registers meshes with soft tissue, we check for pending overrides.
        _pending_overrides = load_overrides()

        def _try_apply_pending_overrides():
            nonlocal _pending_overrides
            if _pending_overrides is None:
                return
            if not skinning.bindings:
                return
            count = apply_overrides(skinning, _pending_overrides)
            if count > 0:
                debug_tab.set_override_count(count)
                stretch_viz.invalidate_chain_cache()
                # Force skinning recompute on next frame
                skinning._last_signature = ()
                print(f"[FaceForge] Auto-loaded {count} overrides on startup")
            _pending_overrides = None  # Applied (or no matches), clear pending

        _after_registration_hooks.append(_try_apply_pending_overrides)

        # Hook stretch viz update into simulation
        _original_soft_tissue_update = skinning.update

        _viz_update_log_counter = [0]
        _skinning_err_reported = [False]

        def _skinning_update_with_viz(body_state):
            try:
                _original_soft_tissue_update(body_state)
            except Exception as e:
                if not _skinning_err_reported[0]:
                    print(f"[DEBUG] EXCEPTION in soft tissue update: {e}")
                    import traceback; traceback.print_exc()
                    _skinning_err_reported[0] = True
            if stretch_viz.stretch_enabled or stretch_viz.chain_enabled:
                if _viz_update_log_counter[0] < 3:
                    print(f"[DEBUG] _skinning_update_with_viz: calling stretch_viz.update() "
                          f"(stretch={stretch_viz.stretch_enabled}, chain={stretch_viz.chain_enabled})")
                    _viz_update_log_counter[0] += 1
                stretch_viz.update()

        skinning.update = _skinning_update_with_viz

        # Store signal handler references on debug_tab to prevent GC
        debug_tab._viz_handler_stretch = _on_stretch_viz
        debug_tab._viz_handler_chain = _on_chain_viz

        # ── Region label visualization + body mesh selection (isolated) ──
        try:
            from faceforge.body.region_labels import (
                BodyRegion, compute_region_colors, segment_mh_mesh,
                save_region_overrides, load_region_overrides, apply_region_overrides,
            )

            # Populate region names in debug tab
            region_names = [f"{r.value}: {r.name}" for r in BodyRegion]
            debug_tab.set_region_names(region_names)

            _region_color_cache: dict[int, bytes] = {}
            _region_viz_active = [False]
            _region_modified: dict[int, int] = {}

            def _on_region_viz(enabled):
                print(f"[DEBUG] _on_region_viz({enabled}), bindings={len(skinning.bindings)}")
                if not enabled:
                    if _region_viz_active[0]:
                        _region_viz_active[0] = False
                        for binding in skinning.bindings:
                            if not binding.is_muscle:
                                binding.mesh.material.vertex_colors_active = False
                    return

                # Disable stretch/chain first so they don't fight
                stretch_viz.stretch_enabled = False
                stretch_viz.chain_enabled = False

                gm = pipeline.gender_morph
                skel_lm = gm.skel_landmarks if gm is not None else None
                if skel_lm is None:
                    print("[FaceForge] Region viz: no skeleton landmarks available yet")
                    return

                _region_viz_active[0] = True
                for binding in skinning.bindings:
                    if binding.is_muscle:
                        continue
                    key = id(binding)
                    if key not in _region_color_cache:
                        pos = binding.mesh.geometry.positions.reshape(-1, 3)
                        labels = segment_mh_mesh(pos.astype(float), skel_lm)
                        _region_color_cache[key] = compute_region_colors(labels)
                    binding.mesh.geometry.vertex_colors = _region_color_cache[key]
                    binding.mesh.geometry.colors_dirty = True
                    binding.mesh.material.vertex_colors_active = True

            debug_tab.region_viz_toggled.connect(_on_region_viz)

            def _region_name_to_id(name: str) -> int:
                try:
                    return int(name.split(":")[0])
                except (ValueError, IndexError):
                    return 0

            def _on_region_reassign(region_name_str):
                region_id = _region_name_to_id(region_name_str)
                gm = pipeline.gender_morph
                if gm is None or gm.mh_region_labels is None:
                    print("[FaceForge] Region reassign: no region labels computed yet")
                    return
                count = 0
                for vi in selection_tool.body_selection:
                    if 0 <= vi < len(gm.mh_region_labels):
                        gm.mh_region_labels[vi] = region_id
                        _region_modified[vi] = region_id
                        count += 1
                if count > 0:
                    gm._region_kdtrees = None
                    _region_color_cache.clear()
                    debug_tab.update_region_override_count(len(_region_modified))
                    print(f"[FaceForge] Set {count} body mesh vertices to region {region_id}")
                    if _region_viz_active[0]:
                        _on_region_viz(True)

            debug_tab.region_reassign_clicked.connect(_on_region_reassign)

            # Deferred body mesh reference for selection tool
            def _ensure_body_mesh_ref():
                if selection_tool.body_mesh is None:
                    gm = pipeline.gender_morph
                    if gm is not None and gm.body_mesh is not None:
                        selection_tool.body_mesh = gm.body_mesh

            # Patch selection mode to defer body mesh ref
            _original_on_selection_mode = debug_tab.selection_mode_toggled
            old_selection_handler = [None]

            def _on_selection_mode_with_body(enabled):
                _ensure_body_mesh_ref()
                selection_tool.active = enabled
                count = selection_tool.selection.total_count + len(selection_tool.body_selection)
                debug_tab.update_selection_count(count)

            # Replace selection mode handler
            debug_tab.selection_mode_toggled.disconnect()
            debug_tab.selection_mode_toggled.connect(_on_selection_mode_with_body)

            # Replace selection changed to include body count
            def _on_selection_changed_with_body():
                count = selection_tool.selection.total_count + len(selection_tool.body_selection)
                debug_tab.update_selection_count(count)

            selection_tool.on_selection_changed = _on_selection_changed_with_body

            # Replace clear to also clear body selection
            def _on_clear_with_body():
                selection_tool.selection.clear()
                selection_tool.body_selection.clear()
                _on_selection_changed_with_body()

            debug_tab.clear_selection_clicked.disconnect()
            debug_tab.clear_selection_clicked.connect(_on_clear_with_body)

            print("[FaceForge] Region label visualization wired successfully")
        except Exception as e:
            print(f"[FaceForge] Region label features unavailable: {e}")
            import traceback
            traceback.print_exc()

        # Wire eye tracking cursor from GL widget
        def on_mouse_move_for_tracking(x, y):
            w = gl_widget.width()
            h = gl_widget.height()
            if w > 0 and h > 0:
                norm_x = (x / w) * 2.0 - 1.0
                norm_y = -((y / h) * 2.0 - 1.0)  # Invert Y
                simulation.eye_tracking.set_cursor_position(norm_x, norm_y)
        gl_widget.mouse_move_callback = on_mouse_move_for_tracking

        # Wire animation player to simulation
        simulation.anim_player = anim_player

        # Populate animation clip selector in display tab
        window.control_panel.display_tab.set_animation_clips(list(builtin_clips.keys()))

        # Report loaded meshes
        meshes = scene.collect_meshes()
        print(f"[FaceForge] Scene has {len(meshes)} renderable meshes")
        for m, _ in meshes[:5]:
            g = m.geometry
            print(f"  - {m.name}: {g.vertex_count} verts, "
                  f"{'indexed' if g.has_indices else 'non-indexed'}, "
                  f"mode={m.material.render_mode.name}")

        # Build search index from loaded meshes (Feature 2)
        mesh_names = [m.name for m, _ in meshes if m.name]
        _search_index.build_from_names(mesh_names)
        print(f"[FaceForge] Search index built: {len(_search_index._entries)} entries")

        # Populate pathology target combo (Feature 7)
        window.control_panel.layers_tab.set_pathology_targets(mesh_names)

        # Register loaded muscle meshes for heatmap (Feature 1)
        # Register all meshes for pathology (Feature 7)
        for m, _ in meshes:
            if m.name:
                _pathology.register_mesh(m, m.name)
                if "muscle" in m.name.lower() or "Muscle" in m.name:
                    _muscle_activation.register_muscle(m, m.name)

        # Apply startup preset (after all systems wired, scene ready)
        if _startup_illustration:
            from faceforge.ui.illustration_presets import apply_illustration_preset
            apply_illustration_preset(
                _startup_illustration,
                window.control_panel.layers_tab,
                event_bus,
                gl_widget,
                window.control_panel.display_tab,
                label_overlay,
                scene,
            )
            print(f"[FaceForge] Applied illustration preset: {_startup_illustration}")
        elif _startup_preset and _startup_preset != "Default":
            apply_preset(
                _startup_preset,
                window.control_panel.layers_tab,
                event_bus,
                gl_widget=gl_widget,
            )
            print(f"[FaceForge] Applied startup preset: {_startup_preset}")

    # Startup preset dialog (must run BEFORE QTimer is scheduled,
    # because QDialog.exec() runs a nested event loop that would
    # fire the timer before _startup_preset is assigned)
    from faceforge.ui.startup_dialog import StartupDialog, apply_preset
    startup_dialog = StartupDialog()
    startup_dialog.exec()
    _startup_preset = startup_dialog.selected_preset
    _startup_illustration = startup_dialog.selected_illustration

    # Schedule asset loading after GL init
    QTimer.singleShot(100, load_assets)

    # Simulation loop (driven by GL widget's paint timer)
    original_paint = gl_widget.paintGL

    def _rebuild_labels_illustration():
        """Rebuild illustration-mode labels from current scene meshes."""
        nonlocal _labels_dirty
        _labels_dirty = False
        from faceforge.ui.illustration_presets import rebuild_illustration_labels
        rebuild_illustration_labels(label_overlay, scene)

    def simulation_paint():
        dt = clock.get_delta()

        # Drive speech AU targets (Feature 4)
        nonlocal _speech_playing, _speech_time
        if _speech_playing:
            _speech_time += dt
            # Find current viseme and apply AU targets
            active_viseme = None
            for viseme in _speech_sequence:
                if viseme.start_time <= _speech_time <= viseme.end_time:
                    active_viseme = viseme
                    break
            if active_viseme is not None:
                for au_id, val in active_viseme.au_targets.items():
                    state.target_au.set(au_id, val)
            elif _speech_time > 0 and (not _speech_sequence or _speech_time > _speech_sequence[-1].end_time):
                # Speech finished — reset mouth AUs
                for au_id in ("AU25", "AU26", "AU22", "AU20"):
                    state.target_au.set(au_id, 0.0)
                _speech_playing = False

        simulation.step(dt)
        original_paint()
        # Update animation transport controls
        if anim_player.is_playing or anim_player.progress > 0:
            window.control_panel.display_tab.update_animation_progress(
                anim_player.progress,
                anim_player.current_time,
                anim_player.duration,
            )
        # Update label overlay
        if _labels_enabled:
            if _labels_dirty:
                if label_overlay._illustration_mode:
                    _rebuild_labels_illustration()
                else:
                    _rebuild_labels()
            label_overlay.set_view_proj(gl_widget.camera.get_view_projection())
            label_overlay.update()

    gl_widget.paintGL = simulation_paint

    # Show window
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
