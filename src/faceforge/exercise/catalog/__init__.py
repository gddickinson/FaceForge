"""The exercise catalogue, by category.

Contents
--------
``_helpers``               constructors (ph, mu, eq), role/kind aliases, shared sources
``lower_body``             squats, deadlifts, lunges and split squats
``lower_body_accessory``   hip thrust, calf raise, wall sit, clamshell, band walk, machines
``upper_push``             bench, incline, push-up, presses, dips, flyes, triceps
``upper_pull``             pull-up, chin-up, pulldown, rows, face pull, curls, rotator cuff
``core_stability``         planks, sit-up, crunch, dead bug, bird dog, twist, knee raise, Pallof
``conditioning``           bike, rower, walking, running, jump rope, jumping jack, climbers, ropes
``athletic``               kettlebell swing, jumps, power clean, medicine ball slam, burpee

:func:`get_exercise_catalog` returns every definition keyed by id, validated.
"""

from __future__ import annotations

from faceforge.exercise.model import Category, ExerciseDefinition


def get_exercise_catalog() -> dict[str, ExerciseDefinition]:
    """Every built-in exercise, keyed by id, in display order."""
    from faceforge.exercise.catalog import (
        athletic, conditioning, core_stability, lower_body, lower_body_accessory,
        upper_pull, upper_push,
    )
    catalog: dict[str, ExerciseDefinition] = {}
    for module in (lower_body, lower_body_accessory, upper_push, upper_pull, core_stability,
                   conditioning, athletic):
        for defn in module.EXERCISES:
            if defn.id in catalog:
                raise ValueError(f"duplicate exercise id {defn.id!r}")
            catalog[defn.id] = defn
    return catalog


def exercises_in_category(catalog: dict[str, ExerciseDefinition],
                          category: Category) -> list[ExerciseDefinition]:
    return [d for d in catalog.values() if d.category is category]
