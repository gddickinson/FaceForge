"""Exercise demonstrations: the app-side glue around :class:`ExerciseRuntime`.

Selecting an exercise (a) enters the gym scene if scene mode is not already
active, (b) asks the layer controller to load the muscle regions the exercise
colours, (c) starts the runtime, which loads the clip into the shared
animation player -- so the existing transport controls drive it -- and
enables the heatmap, and (d) hooks the runtime into the simulation.  Stopping
reverses all of it except the scene, which the user leaves as they would any
other scene.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from faceforge.core.events import EventType
from faceforge.exercise.catalog import get_exercise_catalog
from faceforge.exercise.muscle_groups import ALL_MUSCLE_REGIONS, regions_for_groups
from faceforge.exercise.runtime import ExerciseRuntime
from faceforge.exercise.stabilisers import with_implied_stabilisers

logger = logging.getLogger(__name__)

#: The node the scene environment hangs off, hidden while exporting.
ENVIRONMENT_NODE = "scene_env_root"


def _find_node(scene, name: str):
    """The first node called *name* anywhere under *scene*, or ``None``."""
    stack = [getattr(scene, "root", scene)]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if getattr(node, "name", None) == name:
            return node
        stack.extend(getattr(node, "children", ()))
    return None


#: How many frames between EXERCISE_STATUS events while nothing changes phase.
STATUS_EVERY_N_FRAMES = 6


class ExerciseController:
    """Handlers for the exercise tab."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.catalog = get_exercise_catalog()
        self.runtime: ExerciseRuntime | None = None
        self.options: dict[str, Any] = {
            "reps": None, "tempo": 1.0, "heatmap": True, "equipment": True,
            "palette": "classic", "load_muscles": True,
            # Load every body muscle layer, not only the exercise's regions
            # (the exercise viewer shows the whole body in every exercise).
            "all_muscles": False,
        }
        self._frames = 0
        self._hooked = False

    def subscribe(self) -> None:
        bus = self.ctx.event_bus
        bus.subscribe(EventType.EXERCISE_SELECTED, self.on_exercise_selected)
        bus.subscribe(EventType.EXERCISE_STOPPED, self.on_exercise_stopped)
        bus.subscribe(EventType.EXERCISE_OPTION_CHANGED, self.on_option_changed)
        bus.subscribe(EventType.EXERCISE_EXPORT_OBJ, self.on_export_obj)

    # -- handlers ---------------------------------------------------------------

    def on_option_changed(self, option: str = "", value: Any = None, **kw) -> None:
        if not option:
            return
        self.options[option] = value
        activation = self.ctx.muscle_activation
        if option == "palette" and activation is not None and value:
            activation.set_palette(str(value))
        elif option == "heatmap" and activation is not None:
            activation.set_enabled(bool(value) and self.runtime is not None
                                   and self.runtime.active)
        elif option in ("reps", "tempo") and self.runtime is not None and self.runtime.active:
            defn = self.runtime.definition
            self._start(defn)
        elif option == "equipment" and self.runtime is not None and self.runtime.active:
            self._start(self.runtime.definition)
        elif option == "all_muscles" and value:
            self.load_all_muscles()

    def on_exercise_selected(self, exercise_id: str = "", reps: int | None = None,
                             tempo: float | None = None, **kw) -> None:
        defn = self.catalog.get(exercise_id)
        if defn is None:
            logger.warning("Unknown exercise %r", exercise_id)
            return
        if reps is not None:
            self.options["reps"] = int(reps)
        if tempo is not None:
            self.options["tempo"] = float(tempo)
        self._enter_gym()
        if self.options.get("load_muscles", True):
            self._load_muscle_regions(defn)
        self._start(defn)

    def on_exercise_stopped(self, **kw) -> None:
        if self.runtime is not None:
            self.runtime.stop()
        controller = self.ctx.scene_controller
        if controller is not None:
            controller.set_camera_target_override(None)
        activation = self.ctx.muscle_activation
        if activation is not None:
            activation.set_enabled(False)
        self._unhook()
        self.ctx.event_bus.publish(EventType.EXERCISE_STATUS, exercise_id="", phase="",
                                   kind="", cue="", motions=[], levels={}, time=0.0, rep=0)

    # -- steps ---------------------------------------------------------------------

    def _enter_gym(self) -> None:
        ctx = self.ctx
        controller = ctx.scene_controller
        if controller is not None and getattr(controller, "is_active", False):
            return
        ctx.event_bus.publish(EventType.SCENE_MODE_TOGGLED, enabled=True, scene_type="gym")
        panel = ctx.control_panel
        display = getattr(panel, "display_tab", None)
        sync = getattr(display, "sync_scene_state", None)
        if callable(sync):
            sync(True, "gym")

    def _load_muscle_regions(self, defn) -> None:
        if self.options.get("all_muscles"):
            regions = list(ALL_MUSCLE_REGIONS)
        else:
            # The implied stabilisers (grip, brace) colour regions the
            # catalogue's mover list alone would not load.
            regions = regions_for_groups(with_implied_stabilisers(defn).muscle_groups)
        self._show_regions(regions)

    def load_all_muscles(self) -> None:
        """Load and show every body muscle layer (the exercise viewer's default)."""
        self._show_regions(list(ALL_MUSCLE_REGIONS))

    def _show_regions(self, regions) -> None:
        for region in regions:
            self.ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer=region, visible=True)
            layers_tab = getattr(self.ctx.control_panel, "layers_tab", None)
            setter = getattr(layers_tab, "set_layer_visible", None)
            if callable(setter):
                setter(region, True)

    def _make_runtime(self) -> ExerciseRuntime | None:
        ctx = self.ctx
        joint_setup = getattr(ctx.pipeline, "joint_setup", None)
        if joint_setup is None or ctx.scene_controller is None:
            logger.warning("Exercise runtime needs the body skeleton and scene mode")
            return None
        state = ctx.state

        def apply_live(d: dict) -> None:
            state.body.set_from_js_dict(d)

        return ExerciseRuntime(
            player=ctx.anim_player, scene=ctx.scene,
            wrapper=ctx.scene_controller.wrapper_node, pivots=joint_setup.pivots,
            joint_positions=joint_setup.joint_positions,
            muscle_activation=ctx.muscle_activation, apply_live_body=apply_live,
            show_equipment=bool(self.options.get("equipment", True)),
            body_animation=getattr(ctx.simulation, "body_animation", None),
        )

    def _start(self, defn) -> None:
        if self.runtime is None:
            self.runtime = self._make_runtime()
            if self.runtime is None:
                return
        self.runtime.show_equipment = bool(self.options.get("equipment", True))
        self.runtime.start(defn, reps=self.options.get("reps"),
                           tempo=float(self.options.get("tempo") or 1.0))
        activation = self.ctx.muscle_activation
        if activation is not None:
            activation.set_palette(str(self.options.get("palette", "classic")))
            activation.set_enabled(bool(self.options.get("heatmap", True)))
        controller = self.ctx.scene_controller
        camera = self.ctx.camera
        if controller is not None:
            # The view buttons re-apply presets while the body hangs or lies
            # elsewhere than a standing one; they look where this exercise does.
            controller.set_camera_target_override(defn.camera_target)
        if controller is not None and camera is not None and defn.camera:
            controller.set_camera_preset(camera, defn.camera, target=defn.camera_target)
            orbit = getattr(self.ctx.gl_widget, "orbit_controls", None)
            if orbit is not None:
                orbit.reset_from_camera()
        transport = getattr(getattr(self.ctx.control_panel, "display_tab", None), "transport", None)
        if transport is not None:
            transport.set_duration(self.runtime.built.duration)
            transport.set_playing(True)
        self._hook()
        self.ctx.scene.update()

    def _hook(self) -> None:
        sim = self.ctx.simulation
        if self._hooked or sim is None or self.runtime is None:
            return
        sim.after_animation_hooks.append(self.runtime.after_animation)
        sim.after_scene_update_hooks.append(self.runtime.after_scene_update)
        self._hooked = True

    def _unhook(self) -> None:
        sim = self.ctx.simulation
        if not self._hooked or sim is None or self.runtime is None:
            return
        for hooks, fn in ((sim.after_animation_hooks, self.runtime.after_animation),
                          (sim.after_scene_update_hooks, self.runtime.after_scene_update)):
            if fn in hooks:
                hooks.remove(fn)
        self._hooked = False

    # -- OBJ export --------------------------------------------------------------------

    def on_export_obj(self, on_progress=None, **kw) -> None:
        """Write the frame on screen to an OBJ and open a viewer on it.

        Synchronous, and deliberately so: the exporter walks the live scene
        graph, and the frame loop rewrites every world matrix in place sixty
        times a second.  Handing that to a worker thread would export a body
        that moved halfway through being written.  Playback is paused first
        for the same reason -- a paused clip is a still scene -- and the
        button is disabled by the tab while this runs.
        """
        from faceforge.export.mesh_export import MeshExportError, export_mesh
        from faceforge.ui.obj_viewer import launch

        bus = self.ctx.event_bus
        scene = getattr(self.ctx, "scene", None)
        if scene is None:
            bus.publish(EventType.EXERCISE_EXPORTED, path="", ok=False,
                        message="No scene to export.")
            return
        self._pause_playback()
        path = self._obj_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._room_hidden(scene), self._viewport_frozen():
                result = export_mesh(scene, path, progress=on_progress)
        except (MeshExportError, OSError) as exc:
            logger.warning("OBJ export failed: %s", exc)
            bus.publish(EventType.EXERCISE_EXPORTED, path=str(path), ok=False,
                        message=f"Export failed: {exc}")
            return
        size = result.bytes_written / 1e6
        message = (f"{path.name} — {result.meshes} meshes, {result.vertices:,} vertices, "
                   f"{result.triangles:,} triangles, {size:,.1f} MB")
        try:
            launch(path)
        except OSError as exc:                 # the file is written either way
            logger.warning("could not start the OBJ viewer: %s", exc)
            message += f"  (saved, but the viewer would not start: {exc})"
        bus.publish(EventType.EXERCISE_EXPORTED, path=str(path), ok=True, message=message)

    @contextmanager
    def _room_hidden(self, scene):
        """Hide the gym itself for the duration of an export.

        `export_mesh` writes every VISIBLE mesh, and in scene mode that is the
        building: measured, a back squat exported the floor, the four walls,
        the ceiling and the lamp along with the athlete, and the file's bounds
        came out as the room (x +-251, z 0..400) rather than the body.  A
        viewer framing that shows a room with a speck in it.
        """
        env = _find_node(scene, ENVIRONMENT_NODE)
        if env is None:                        # clinical view, or no scene mode
            yield
            return
        was = env.visible
        env.visible = False
        try:
            yield
        finally:
            env.visible = was

    @contextmanager
    def _viewport_frozen(self):
        """Stop the 3D viewport repainting for the duration of the export.

        The tab pumps `processEvents` so its progress bar moves; without this
        that would let the ~60 Hz refresh timer run the frame loop, which
        rewrites every world matrix in place -- and the exporter is reading
        those matrices.  Stopping the timer makes the pumped events safe: the
        scene cannot move while nothing is drawing it.
        """
        widget = getattr(self.ctx, "gl_widget", None)
        timer = getattr(widget, "_timer", None)
        running = bool(timer is not None and timer.isActive())
        if running:
            timer.stop()
        try:
            yield
        finally:
            if running:
                timer.start()

    def _pause_playback(self) -> None:
        player = getattr(self.ctx, "animation_player", None)
        for target in (player, getattr(self.runtime, "player", None)):
            if target is not None and hasattr(target, "pause"):
                try:
                    target.pause()
                except Exception:              # pausing is a courtesy, not a contract
                    logger.debug("could not pause playback before the export")

    def _obj_path(self):
        """``results/exercise_obj/<exercise>_<phase>.obj``, or a dated name if idle."""
        from datetime import datetime

        from faceforge.constants import PROJECT_ROOT

        status = self.runtime.status() if self.runtime is not None else None
        if status is None:
            stem = f"scene_{datetime.now():%Y%m%d_%H%M%S}"
        else:
            phase = "".join(c if c.isalnum() else "_" for c in status.phase).strip("_")
            stem = f"{status.exercise_id}_{phase or 'frame'}"
        return PROJECT_ROOT / "results" / "exercise_obj" / f"{stem}.obj"

    # -- per frame ---------------------------------------------------------------------

    def update_frame(self) -> None:
        """Publish the status the tab shows: every phase change, else every few frames."""
        if self.runtime is None or not self.runtime.active:
            return
        self._frames += 1
        changed = self.runtime.phase_changed()
        if not changed and self._frames % STATUS_EVERY_N_FRAMES:
            return
        status = self.runtime.status()
        if status is not None:
            self.ctx.event_bus.publish(EventType.EXERCISE_STATUS, **status.as_event())
