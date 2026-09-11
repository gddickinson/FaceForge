"""The exercise runtime: drives a built clip through the existing animation player.

No Qt here.  :class:`ExerciseRuntime` is given the collaborators it needs
(the animation player, the muscle activation system, the joint pivots, the
scene wrapper and the scene root) and is then hooked into the simulation at
two points:

* :meth:`after_animation` (right after the player ticks): samples the
  activation track at the player's time and hands the levels to the heatmap;
* :meth:`after_scene_update` (after world matrices are fresh): re-anchors the
  feet or hands with the ground lock, places the equipment in the hands, and
  refreshes the scene once more.

The player's body callback is wrapped while an exercise runs so that (a) the
authored lift and travel ride along in the interpolated body dict, and (b)
the live body state is written directly as well as the target, which removes
the interpolator's 0.25 s lag from fast movements (a jump rope at 2 Hz would
otherwise be damped to a shuffle).

``tools/render_exercise_demo.py`` drives this same class headlessly, so what
the demo shows is what the app does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from faceforge.body.ground_contact import GroundLock
from faceforge.exercise.grip_lock import GripWidthLock
from faceforge.exercise.clip_builder import (
    LIFT_KEY, TRAVEL_X_KEY, TRAVEL_Z_KEY, ExerciseClip, build_exercise_clip,
)
from faceforge.exercise.equipment import build_equipment
from faceforge.exercise.equipment_rig import EquipmentRig
from faceforge.exercise.model import ExerciseDefinition
from faceforge.exercise.motion_description import summarise_motions

logger = logging.getLogger(__name__)

#: Spin (rev/s) and hang (units) for held equipment kinds.
_HELD_TUNING: dict[str, dict[str, float]] = {
    "kettlebell": {"hang": 10.0, "grip_offset": 8.0},
    "jump_rope": {"spin": 2.0, "grip_offset": 6.0},
    "medicine_ball": {"hang": 6.0, "grip_offset": 10.0},
    "barbell": {"grip_offset": 7.0},
    "dumbbell": {"grip_offset": 7.0},
    "cable_handle": {"grip_offset": 7.0},
    "band": {"grip_offset": 7.0},
}


def equipment_tuning(spec: EquipmentSpec) -> dict[str, float]:
    """How ``spec`` follows the hands: the kind's defaults, with the spec's ``hang`` on top."""
    tuning = dict(_HELD_TUNING.get(spec.kind, {}))
    tuning["hang"] = float(spec.hang) if spec.hang is not None else float(tuning.get("hang", 0.0))
    return tuning


@dataclass
class ExerciseStatus:
    """What the UI shows for the current frame."""

    exercise_id: str
    time: float
    rep: int
    phase: str
    kind: str
    cues: tuple[str, ...]
    motions: list[str]
    group_levels: dict[str, float]

    def as_event(self) -> dict[str, Any]:
        return {"exercise_id": self.exercise_id, "time": self.time, "rep": self.rep,
                "phase": self.phase, "kind": self.kind, "cue": " · ".join(self.cues),
                "motions": list(self.motions), "levels": dict(self.group_levels)}


@dataclass
class ExerciseRuntime:
    """Runs one exercise demonstration on top of the animation player."""

    player: Any
    scene: Any
    wrapper: Any
    pivots: dict
    joint_positions: dict
    muscle_activation: Any = None
    apply_live_body: Callable[[dict], None] | None = None
    #: The body animation system, for the grip-width lock's forward kinematics
    #: (optional: without it hands anchored to a bar may slide along it).
    body_animation: Any = None
    show_equipment: bool = True
    floor_y: float = 0.0

    built: ExerciseClip | None = None
    ground_lock: GroundLock | None = None
    grip_lock: GripWidthLock | None = None
    rig: EquipmentRig = field(default_factory=EquipmentRig)
    _lift: float = 0.0
    _travel: tuple[float, float] = (0.0, 0.0)
    _orig_on_body: Any = None
    _last_span_index: int = -1

    # -- lifecycle --------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.built is not None

    @property
    def definition(self) -> ExerciseDefinition | None:
        return self.built.definition if self.built else None

    def start(self, defn: ExerciseDefinition, reps: int | None = None, tempo: float = 1.0,
              autoplay: bool = True) -> ExerciseClip:
        """Build the clip, place the body and equipment, and load the player."""
        if self.built is not None:
            self.stop()
        built = build_exercise_clip(defn, reps=reps, tempo=tempo)
        self.built = built
        self._lift, self._travel = 0.0, (0.0, 0.0)
        self._last_span_index = -1

        self.ground_lock = GroundLock(defn.anchor, defn.lock_horizontal, side=defn.anchor_side)
        if defn.anchor != "none":
            self.ground_lock.calibrate(self.pivots, built.base_position,
                                       built.base_quaternion, floor_y=self.floor_y)
            if defn.anchor_point is not None:
                self.ground_lock.set_target(defn.anchor_point)

        if self.show_equipment:
            self._build_equipment(defn)

        # Hands gripping a fixed point (a pull-up bar) must not slide along it.
        self.grip_lock = None
        if (defn.anchor == "hands" and defn.anchor_point is not None
                and self.body_animation is not None):
            self.grip_lock = GripWidthLock(self.body_animation, self.scene, self.pivots)

        if self.muscle_activation is not None:
            self.muscle_activation.set_levels({})

        self._orig_on_body = self.player.on_body_state
        self.player.on_body_state = self._on_body_state
        self.player.load(built.clip)
        self.player.seek(0.0)
        if autoplay:
            self.player.play()
        logger.info("Exercise %s: %d reps, %.1f s, %d equipment nodes",
                    defn.id, built.reps, built.duration, len(self.rig.items))
        return built

    def stop(self) -> None:
        """Unload: stop the player, drop the equipment, release the heatmap."""
        if self.built is None:
            return
        self.player.stop()
        if self._orig_on_body is not None:
            self.player.on_body_state = self._orig_on_body
            self._orig_on_body = None
        for node in self.rig.clear():
            try:
                self.scene.remove(node)
            except (ValueError, AttributeError):
                pass
        if self.muscle_activation is not None:
            self.muscle_activation.set_levels(None)
        self.built = None
        self.ground_lock = None
        self.grip_lock = None

    def _build_equipment(self, defn: ExerciseDefinition) -> None:
        for spec in defn.equipment:
            try:
                node = build_equipment(spec.kind, **spec.params)
            except (KeyError, TypeError) as exc:
                logger.warning("Equipment %r skipped: %s", spec.kind, exc)
                continue
            tuning = equipment_tuning(spec)
            self.scene.add(node)
            self.rig.add(node, spec, grip_offset=tuning.get("grip_offset"),
                         hang=tuning["hang"], spin=tuning.get("spin", 0.0))

    # -- player callback ----------------------------------------------------------

    def _on_body_state(self, state_dict: dict) -> None:
        self._lift = float(state_dict.get(LIFT_KEY, 0.0))
        self._travel = (float(state_dict.get(TRAVEL_X_KEY, 0.0)),
                        float(state_dict.get(TRAVEL_Z_KEY, 0.0)))
        if self.grip_lock is not None:
            self.grip_lock.apply(state_dict)
        if self._orig_on_body is not None:
            self._orig_on_body(state_dict)
        if self.apply_live_body is not None:
            self.apply_live_body(state_dict)

    # -- simulation hooks ---------------------------------------------------------

    def after_animation(self, dt: float = 0.0) -> None:
        """Sample the activation track at the player's time into the heatmap."""
        if self.built is None or self.muscle_activation is None:
            return
        levels = self.built.activation.sample(self.player.current_time)
        self.muscle_activation.set_levels(levels)

    def after_scene_update(self) -> None:
        """Re-anchor the body and place the equipment; refresh world matrices."""
        if self.built is None:
            return
        moved = False
        if self.ground_lock is not None and self.ground_lock.anchor != "none":
            delta = self.ground_lock.update(self.wrapper, self.pivots, lift=self._lift,
                                            travel=self._travel)
            moved = bool((abs(delta) > 1e-6).any())
        if moved:
            self.scene.update()
        if self.rig.items:
            self.rig.update(self.pivots, time=self.player.current_time)
            self.scene.update()

    # -- status ---------------------------------------------------------------------

    def status(self) -> ExerciseStatus | None:
        if self.built is None:
            return None
        t = self.player.current_time
        span = self.built.span_at(t)
        if span is None:
            return None
        levels = self.built.activation.sample_groups(self.built.definition, t)
        return ExerciseStatus(
            exercise_id=self.built.definition.id, time=t, rep=span.rep, phase=span.name,
            kind=span.kind.value, cues=span.cues, motions=summarise_motions(span.motions),
            group_levels=levels,
        )

    def phase_changed(self) -> bool:
        """True once per phase boundary (for UIs that only redraw on change)."""
        if self.built is None:
            return False
        span = self.built.span_at(self.player.current_time)
        idx = span.index if span else -1
        changed = idx != self._last_span_index
        self._last_span_index = idx
        return changed
