"""Does each exercise build the equipment its own words promise?

Every other equipment audit measures what IS in the scene -- whether the body
touches it, whether it is inside the body, whether it reaches the floor.  None
of them can see a thing that was never built.  A "Seated leg extension
(machine)" whose only item is a bench renders as a man sitting on a box
kicking the air, and every geometric audit passes it.

This reads the catalogue's own claim -- the name and the description -- and
checks the `equipment` tuple against it.

    python -m tools.audit_equipment_promised
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from faceforge.exercise.catalog import get_exercise_catalog

#: A word in the name or description -> the equipment kind it promises.
#: ``None`` means the word is checked by a rule below rather than by kind.
PROMISES: dict[str, str | None] = {
    "machine": None,
    "barbell": "barbell", "dumbbell": "dumbbell", "kettlebell": "kettlebell",
    "cable": "cable_handle", "band": "band", "wall": "wall",
    "medicine ball": "medicine_ball", "treadmill": "treadmill",
    "bike": "bike", "rower": "rower", "mat": "mat",
    # A lift that starts off pins needs something to start off.
    "pins": "rack_pins", "rack pull": "rack_pins",
}
#: Anything here satisfies the word "machine".
MACHINE_LIKE = frozenset({"bike", "rower", "treadmill", "cable_handle",
                          "pulldown_bar", "dip_station", "pullup_bar",
                          "leg_extension_machine", "leg_curl_machine"})
#: Words that turn up in comparisons rather than as a claim about this
#: exercise's own kit ("...than a close-grip BENCH press"), and implements a
#: movement can legitimately be done with either of (a goblet squat).
INTERCHANGEABLE: dict[str, frozenset[str]] = {
    "dumbbell": frozenset({"kettlebell"}),
    "kettlebell": frozenset({"dumbbell"}),
}


def missing_for(defn) -> list[str]:
    kinds = {e.kind for e in defn.equipment}
    text = f"{defn.name} {defn.description}".lower()
    gaps = []
    for word, kind in PROMISES.items():
        if word not in text:
            continue
        if word == "machine":
            if not (kinds & MACHINE_LIKE):
                gaps.append("a machine of any kind")
        elif kind and kind not in kinds and not (kinds & INTERCHANGEABLE.get(kind, frozenset())):
            gaps.append(kind)
    return gaps


def audit(ids) -> int:
    catalog = get_exercise_catalog()
    flagged = 0
    for eid in ids:
        defn = catalog[eid]
        gaps = missing_for(defn)
        if not gaps:
            continue
        flagged += 1
        kinds = sorted({e.kind for e in defn.equipment})
        print(f"  {eid:30s} promises {', '.join(gaps):24s} builds {kinds or '[]'}",
              flush=True)
    print(f"\n{flagged} exercises promise equipment they do not build, of {len(ids)}",
          flush=True)
    return flagged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exercise", default=None, help="comma list; default every exercise")
    args = ap.parse_args(argv)
    catalog = get_exercise_catalog()
    ids = args.exercise.split(",") if args.exercise else sorted(catalog)
    return 0 if audit(ids) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
