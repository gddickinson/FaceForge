"""Muscles an exercise works without anyone listing them.

A catalogue entry names the movers of its movement.  A body performing that
movement is also gripping the bar, bracing its trunk under the load and
standing on its feet, and a physiotherapist expects to see those muscles
coloured too: in a deadlift the hands, forearms, upper arms and back are
working as hard as anything below the hips.  This module derives those
stabilisers from what the definition already says -- what is held, what the
body is anchored by, how it is oriented -- and adds them at the stabiliser
role, so the authored list stays the list of movers.

Levels are fractions of maximal voluntary contraction in the "low" and
"moderate" bands of DiGiovine et al. (0-20 % low, 21-40 % moderate,
41-60 % high), chosen per function:

* **Held load** (a barbell, dumbbell, kettlebell, ball, handle or band in the
  hands): the grip at 0.55 (forearm flexors, hand intrinsics), the wrist
  extensors that stiffen the wrist at 0.35, the elbow flexors and
  brachioradialis holding the elbow at 0.25-0.3, the deltoids and upper
  trapezius carrying the load at 0.25-0.4, the rotator cuff centring the
  humeral head at 0.3, and the trunk brace (erector spinae 0.45, multifidus
  0.35, transversus abdominis 0.35, obliques 0.3, rectus abdominis 0.3,
  quadratus lumborum 0.3).
* **Hanging from a fixed bar**: the grip at 0.6 and the scapular stabilisers
  (serratus anterior, rotator cuff) at 0.3.
* **Hands on the floor**: grip 0.3, serratus anterior 0.35, rotator cuff 0.3.
* **Standing**: the trunk at a bodyweight brace (erector spinae 0.3,
  multifidus 0.3, transversus 0.3, obliques 0.25), hip abductors 0.3 and the
  ankle and foot (tibialis anterior 0.2, soleus 0.25, gastrocnemius 0.2,
  peroneals 0.2, foot intrinsics 0.25).
* **A bar on the back** (a lower-body barbell exercise): the upper and middle
  trapezius, rhomboids and erector spinae carry it (0.4 / 0.3 / 0.5).

A group the definition already lists keeps the authored value; implied
levels never override it.  Where two rules name the same group the higher
level wins.
"""

from __future__ import annotations

from dataclasses import replace

from faceforge.exercise.model import Category, ExerciseDefinition, MuscleUse, Role

#: Equipment kinds that are a load carried in the hands.
HELD_KINDS: frozenset[str] = frozenset({
    "barbell", "dumbbell", "kettlebell", "medicine_ball", "cable_handle", "band",
    "jump_rope", "battle_rope",
})
#: Static equipment the hands hold on to without lifting it.
FIXED_GRIP_KINDS: frozenset[str] = frozenset({"pullup_bar", "dip_station", "rower"})

_GRIP_HELD = {"forearm_flexors": 0.55, "hand_intrinsics": 0.55, "forearm_extensors": 0.35,
              "brachioradialis": 0.3, "biceps_brachii": 0.25, "brachialis": 0.25}
_CARRY = {"deltoid_anterior": 0.3, "deltoid_lateral": 0.25, "deltoid_posterior": 0.25,
          "trapezius_upper": 0.4, "rotator_cuff": 0.3}
_BRACE_LOADED = {"erector_spinae": 0.45, "multifidus": 0.35, "transversus_abdominis": 0.35,
                 "obliques": 0.3, "rectus_abdominis": 0.3, "quadratus_lumborum": 0.3}
_GRIP_HANG = {"forearm_flexors": 0.6, "hand_intrinsics": 0.6, "forearm_extensors": 0.3,
              "brachioradialis": 0.3, "serratus_anterior": 0.3, "rotator_cuff": 0.3}
_GRIP_FLOOR = {"forearm_flexors": 0.3, "hand_intrinsics": 0.3, "serratus_anterior": 0.35,
               "rotator_cuff": 0.3}
_STANDING = {"erector_spinae": 0.3, "multifidus": 0.3, "transversus_abdominis": 0.3,
             "obliques": 0.25, "gluteus_medius": 0.3, "tibialis_anterior": 0.2, "soleus": 0.25,
             "gastrocnemius": 0.2, "peroneals": 0.2, "foot_intrinsics": 0.25}
_BAR_ON_BACK = {"trapezius_upper": 0.4, "trapezius_middle": 0.3, "rhomboids": 0.3,
                "erector_spinae": 0.5}


def _grip_closed(defn: ExerciseDefinition) -> bool:
    return any(float(ph.pose.get(k, 0.0)) > 0.3
               for ph in defn.phases for k in ("finger_curl_r", "finger_curl_l"))


def _bar_racked(defn: ExerciseDefinition) -> bool:
    """Hands up at the shoulders in every phase: a bar on the back or in the front rack."""
    for ph in defn.phases:
        if float(ph.pose.get("elbow_r_flex", 0.0)) < 0.6 or float(ph.pose.get("shoulder_r_flex", 0.0)) > 0.6:
            return False
    return bool(defn.phases)


def implied_levels(defn: ExerciseDefinition) -> dict[str, tuple[float, str]]:
    """``group -> (level, reason)`` for every implied stabiliser of ``defn``."""
    out: dict[str, tuple[float, str]] = {}

    def add(table: dict[str, float], reason: str) -> None:
        for group, level in table.items():
            if group not in out or out[group][0] < level:
                out[group] = (level, reason)

    kinds = {e.kind for e in defn.equipment}
    held = {e.kind for e in defn.equipment
            if e.attach in ("hands", "hand_r", "hand_l") and e.kind in HELD_KINDS}
    fixed = bool(kinds & FIXED_GRIP_KINDS) or (defn.anchor == "hands"
                                               and defn.anchor_point is not None)
    if held:
        add(_GRIP_HELD, "holding a " + ", ".join(sorted(held)).replace("_", " "))
        add(_CARRY, "carrying the load")
        add(_BRACE_LOADED, "bracing under load")
    if fixed or (defn.orientation == "hanging" and _grip_closed(defn)):
        add(_GRIP_HANG, "gripping the bar")
    elif defn.anchor == "hands" and defn.anchor_point is None:
        add(_GRIP_FLOOR, "hands on the floor")
    if defn.orientation == "standing" and defn.anchor == "feet":
        add(_STANDING, "standing")
    if "barbell" in held and defn.category is Category.LOWER_BODY and _bar_racked(defn):
        add(_BAR_ON_BACK, "bar on the back")
    return out


def implied_stabilisers(defn: ExerciseDefinition) -> tuple[MuscleUse, ...]:
    """The implied stabilisers not already in ``defn.muscles``."""
    listed = {use.group for use in defn.muscles}
    uses = []
    for group, (level, reason) in sorted(implied_levels(defn).items()):
        if group in listed:
            continue
        uses.append(MuscleUse(group, Role.STABILISER, level, note=f"implied: {reason}"))
    return tuple(uses)


def with_implied_stabilisers(defn: ExerciseDefinition) -> ExerciseDefinition:
    """``defn`` with the implied stabilisers appended to its muscle list."""
    extra = implied_stabilisers(defn)
    if not extra:
        return defn
    return replace(defn, muscles=tuple(defn.muscles) + extra)
