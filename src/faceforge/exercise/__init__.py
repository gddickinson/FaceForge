"""Exercise demonstrations: technique, moving joints and working muscles.

Contents
--------
``model``               dataclasses: ExerciseDefinition, Phase, MuscleUse, EquipmentSpec
``muscle_groups``       functional muscle groups -> the mesh names in assets/config/muscles
``pose_library``        pose authoring in degrees, foot-flat rule, hinge/squat helpers
``activation``          per-phase activation model and the time-sampled ActivationTrack
``motion_description``  DOF changes -> anatomical movement terms with degrees
``clip_builder``        ExerciseDefinition -> AnimationClip + phase spans + activation track
``equipment``           procedural gym equipment (barbell, dumbbell, kettlebell, bench, ...)
``equipment_rig``       per-frame placement of hand-held equipment on the wrist pivots
``catalog``             the exercise definitions, by category
"""

from faceforge.exercise.model import (
    Category, ExerciseDefinition, EquipmentSpec, MuscleUse, Phase, PhaseKind, Role,
)

__all__ = [
    "Category", "ExerciseDefinition", "EquipmentSpec", "MuscleUse", "Phase",
    "PhaseKind", "Role",
]
