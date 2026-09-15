"""The exercise catalogue, by category.

Contents
--------
``_helpers``               constructors (ph, mu, eq), role/kind aliases, shared sources
``lower_body``             the squats
``hinges``                 the deadlifts and their variations
``lunges``                 lunges and the split squat
``lower_body_accessory``   hip thrust, calf raise, wall sit, clamshell, band walk, machines
``upper_push``             bench, incline, push-up, presses, dips, flyes, triceps
``bench_variants``         bench press by grip width, incline, decline and the floor press
``press_variants``         push press, Arnold press, close-grip push-up, overhead triceps
``upper_pull``             pull-ups (4 grips), pulldown, curls, rotator cuff
``rows``                   the horizontal pulls: rows, face pull, reverse fly
``core_stability``         planks, sit-up, crunch, dead bug, bird dog, twist, knee raise, Pallof
``conditioning``           bike, rower, walking, running, jump rope, jumping jack, climbers, ropes
``athletic``               kettlebell swing, jumps, power clean, medicine ball slam, burpee
``kettlebell``             cleans, snatches, presses, carries, the windmill and the get-up
``calisthenics``           muscle-up, L-sit, pistol, archer and pike push-ups, Nordic curl
``yoga``                   eight held asanas: chair, warrior II, triangle, tree, dog, cobra, cat-cow
``stretches``              static stretches and mobility drills

:func:`get_exercise_catalog` returns every definition keyed by id, validated.
"""

from __future__ import annotations

from faceforge.exercise.model import Category, ExerciseDefinition


def get_exercise_catalog() -> dict[str, ExerciseDefinition]:
    """Every built-in exercise, keyed by id, in display order."""
    from faceforge.exercise.catalog import (
        athletic, bench_variants, calisthenics, conditioning, core_stability, hinges,
        kettlebell, lower_body, lower_body_accessory, lunges, press_variants, rows,
        stretches, upper_pull, upper_push, yoga,
    )
    catalog: dict[str, ExerciseDefinition] = {}
    for module in (lower_body, hinges, lunges, lower_body_accessory, upper_push,
                   bench_variants, press_variants, upper_pull, rows, core_stability,
                   conditioning, athletic, kettlebell, calisthenics, yoga, stretches):
        for defn in module.EXERCISES:
            if defn.id in catalog:
                raise ValueError(f"duplicate exercise id {defn.id!r}")
            catalog[defn.id] = defn
    return catalog


def exercises_in_category(catalog: dict[str, ExerciseDefinition],
                          category: Category) -> list[ExerciseDefinition]:
    return [d for d in catalog.values() if d.category is category]
