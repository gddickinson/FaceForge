"""Functional muscle groups -> the muscle mesh names in ``assets/config/muscles``.

Exercise definitions talk about "quadriceps" and "gluteus maximus"; the
heatmap talks about mesh names such as ``"Vastus Lat. R"``.  This table is the
only place the two meet.  A group lists BASE names; :func:`expand_group`
appends the side suffix the configs use (``" R"`` / ``" L"``) unless the
muscle is midline and unsided.  ``tests/exercise/test_muscle_groups.py``
checks every name here against the config files, so a renamed mesh fails the
suite instead of silently going grey in the heatmap.

``GROUP_REGION`` maps a group to the on-demand muscle layer that contains it,
so an exercise can ask for exactly the regions it colours.
"""

from __future__ import annotations

#: Base mesh names per functional group (side suffix added by expand_group).
MUSCLE_GROUPS: dict[str, tuple[str, ...]] = {
    # -- hip and thigh --
    "quadriceps": ("Rectus Femoris", "Vastus Lat.", "Vastus Med.", "Vastus Inter."),
    "gluteus_maximus": ("Gluteus Max.",),
    "gluteus_medius": ("Gluteus Med.", "Gluteus Min."),
    "hamstrings": ("Biceps Fem. Long", "Biceps Fem. Short", "Semitendinosus", "Semimembranosus"),
    "adductors": ("Adductor Magnus", "Adductor Longus", "Adductor Brevis", "Adductor Min.",
                  "Pectineus", "Gracilis"),
    "hip_flexors": ("Psoas Major", "Iliacus", "Sartorius", "Tensor Fasc. Lat."),
    "hip_external_rotators": ("Piriformis", "Obt. Internus", "Obt. Externus", "Gem. Superior",
                              "Gem. Inferior", "Quad. Femoris"),
    "tensor_fasciae_latae": ("Tensor Fasc. Lat.", "Iliotibial Tract"),
    # -- leg and foot --
    "gastrocnemius": ("Gastroc. Med.", "Gastroc. Lat.", "Plantaris"),
    "soleus": ("Soleus",),
    "tibialis_anterior": ("Tibialis Ant.", "Ext. Dig. Long.", "Ext. Hall. Long."),
    "peroneals": ("Fibularis Long.", "Fibularis Brev.", "Fibularis Tert."),
    "tibialis_posterior": ("Tibialis Post.", "Flex. Dig. Long.", "Flex. Hall. Long."),
    # -- trunk --
    "erector_spinae": ("Iliocostalis Lumb.", "Iliocostalis Thor.", "Longissimus Thor.",
                       "Spinalis Thor."),
    "multifidus": ("Multifidus", "Lumbar Rotator", "Semispinalis Thor."),
    "rectus_abdominis": ("Rectus Abdominis", "Pyramidalis"),
    "obliques": ("Ext. Oblique", "Int. Oblique"),
    "transversus_abdominis": ("Trans. Abdominis",),
    "quadratus_lumborum": ("Quadratus Lumb.",),
    # -- shoulder girdle and back --
    "latissimus_dorsi": ("Latissimus Dorsi", "Teres Major"),
    "trapezius_upper": ("Desc. Trapezius",),
    "trapezius_middle": ("Trans. Trapezius",),
    "trapezius_lower": ("Asc. Trapezius",),
    "rhomboids": ("Rhomboid Major", "Rhomboid Minor"),
    "serratus_anterior": ("Serratus Ant.",),
    "deltoid_anterior": ("Deltoid Clav.",),
    "deltoid_lateral": ("Deltoid Acr.",),
    "deltoid_posterior": ("Deltoid Spin.",),
    "rotator_cuff": ("Supraspinatus", "Infraspinatus", "Teres Minor", "Subscapularis"),
    "infraspinatus_teres_minor": ("Infraspinatus", "Teres Minor"),
    "pectoralis_major": ("Pect. Major Clav.", "Pect. Major Stern.", "Pect. Major Abd."),
    "pectoralis_upper": ("Pect. Major Clav.",),
    "pectoralis_minor": ("Pect. Minor",),
    # -- arm --
    "biceps_brachii": ("Biceps Long", "Biceps Short"),
    "brachialis": ("Brachialis",),
    "brachioradialis": ("Brachioradialis",),
    "triceps_brachii": ("Triceps Long", "Triceps Lat.", "Triceps Med.", "Anconeus"),
    "forearm_flexors": ("Flex. Carpi Rad.", "Flex. Carpi Uln. Hum.", "Flex. Carpi Uln. Uln.",
                        "Palmaris Longus", "Flex. Dig. Sup. HU", "Flex. Dig. Sup. Rad.",
                        "Flex. Dig. Prof.", "Flex. Poll. Long."),
    "forearm_extensors": ("Ext. Carpi Rad. Long.", "Ext. Carpi Rad. Brev.", "Ext. Carpi Uln. Hum.",
                          "Ext. Carpi Uln. Uln.", "Ext. Digitorum", "Ext. Dig. Min."),
    "pronators": ("Pronator Teres Hum.", "Pronator Teres Uln.", "Pronator Quadratus"),
    "supinator": ("Supinator",),
    # Intrinsic hand and foot muscles: the configs name these "R Lumbricals",
    # side first (see SIDE_PREFIX_GROUPS).
    "hand_intrinsics": ("Abductor Pollicis Brevis", "Opponens Pollicis",
                        "Flexor Pollicis Brevis Deep", "Flexor Pollicis Brevis Superficial",
                        "Adductor Pollicis Oblique", "Adductor Pollicis Transverse",
                        "Abductor Digiti Minimi", "Flexor Digiti Minimi Brevis",
                        "Opponens Digiti Minimi", "Lumbricals", "Palmar Interossei",
                        "Dorsal Interossei"),
    # The foot's digiti minimi muscles and dorsal interossei carry "(Foot)"
    # in the config: the heatmap registry and the activation track are keyed
    # by mesh name, and the hand config owns the bare names.
    "foot_intrinsics": ("Abductor Hallucis", "Flexor Digitorum Brevis",
                        "Abductor Digiti Minimi (Foot)",
                        "Flexor Accessorius", "First Lumbrical", "Second Lumbrical",
                        "Third Lumbrical", "Fourth Lumbrical",
                        "Flexor Digiti Minimi Brevis (Foot)", "Opponens Digiti Minimi (Foot)",
                        "Dorsal Interossei (Foot)",
                        "Flexor Hallucis Brevis Medial", "Flexor Hallucis Brevis Lateral",
                        "Adductor Hallucis Oblique", "Adductor Hallucis Transverse"),
}

#: Groups whose mesh names carry the side as a PREFIX ("R Lumbricals").
SIDE_PREFIX_GROUPS: frozenset[str] = frozenset({"hand_intrinsics", "foot_intrinsics"})

#: Every body muscle layer, in load order -- what "show all muscles" loads.
ALL_MUSCLE_REGIONS: tuple[str, ...] = (
    "back_muscles", "shoulder_muscles", "arm_muscles", "torso_muscles",
    "hip_muscles", "leg_muscles", "hand_muscles", "foot_muscles",
)

#: Midline muscles that carry no side suffix in the configs.
UNSIDED_MUSCLES: frozenset[str] = frozenset({
    "Thoracic Rotator", "Diaphragm", "Linea Alba", "Ext. Intercostal", "Int. Intercostal",
    "Innermost Intercostal", "Interspinales Lumb.", "Interspinales Thor.",
})

#: Group -> muscle layer (the on-demand loader id and config file stem).
GROUP_REGION: dict[str, str] = {
    "quadriceps": "leg_muscles", "hamstrings": "leg_muscles", "adductors": "leg_muscles",
    "gastrocnemius": "leg_muscles", "soleus": "leg_muscles", "tibialis_anterior": "leg_muscles",
    "peroneals": "leg_muscles", "tibialis_posterior": "leg_muscles",
    "tensor_fasciae_latae": "leg_muscles",
    "gluteus_maximus": "hip_muscles", "gluteus_medius": "hip_muscles",
    "hip_flexors": "hip_muscles", "hip_external_rotators": "hip_muscles",
    "erector_spinae": "back_muscles", "multifidus": "back_muscles",
    "latissimus_dorsi": "back_muscles", "trapezius_upper": "back_muscles",
    "trapezius_middle": "back_muscles", "trapezius_lower": "back_muscles",
    "rhomboids": "back_muscles",
    "rectus_abdominis": "torso_muscles", "obliques": "torso_muscles",
    "transversus_abdominis": "torso_muscles", "quadratus_lumborum": "torso_muscles",
    "pectoralis_major": "torso_muscles", "pectoralis_upper": "torso_muscles",
    "pectoralis_minor": "torso_muscles",
    "serratus_anterior": "shoulder_muscles", "deltoid_anterior": "shoulder_muscles",
    "deltoid_lateral": "shoulder_muscles", "deltoid_posterior": "shoulder_muscles",
    "rotator_cuff": "shoulder_muscles", "infraspinatus_teres_minor": "shoulder_muscles",
    "biceps_brachii": "arm_muscles", "brachialis": "arm_muscles", "brachioradialis": "arm_muscles",
    "triceps_brachii": "arm_muscles", "forearm_flexors": "arm_muscles",
    "forearm_extensors": "arm_muscles", "pronators": "arm_muscles", "supinator": "arm_muscles",
    "hand_intrinsics": "hand_muscles", "foot_intrinsics": "foot_muscles",
}

#: Groups whose members are split between two configs (Tensor fasciae latae
#: lives in hip_muscles, the iliotibial tract in leg_muscles).
_EXTRA_REGIONS: dict[str, tuple[str, ...]] = {
    "tensor_fasciae_latae": ("hip_muscles", "leg_muscles"),
    "hip_flexors": ("hip_muscles", "leg_muscles"),   # Sartorius is a leg config entry
    "quadriceps": ("leg_muscles",),
}

#: Display names for the UI.
GROUP_LABELS: dict[str, str] = {
    "quadriceps": "Quadriceps", "gluteus_maximus": "Gluteus maximus",
    "gluteus_medius": "Gluteus medius / minimus", "hamstrings": "Hamstrings",
    "adductors": "Hip adductors", "hip_flexors": "Hip flexors (iliopsoas)",
    "hip_external_rotators": "Deep hip rotators", "tensor_fasciae_latae": "TFL / IT band",
    "gastrocnemius": "Gastrocnemius", "soleus": "Soleus", "tibialis_anterior": "Tibialis anterior",
    "peroneals": "Peroneals", "tibialis_posterior": "Tibialis posterior",
    "erector_spinae": "Erector spinae", "multifidus": "Multifidus / rotatores",
    "rectus_abdominis": "Rectus abdominis", "obliques": "Obliques",
    "transversus_abdominis": "Transversus abdominis", "quadratus_lumborum": "Quadratus lumborum",
    "latissimus_dorsi": "Latissimus dorsi / teres major", "trapezius_upper": "Upper trapezius",
    "trapezius_middle": "Middle trapezius", "trapezius_lower": "Lower trapezius",
    "rhomboids": "Rhomboids", "serratus_anterior": "Serratus anterior",
    "deltoid_anterior": "Anterior deltoid", "deltoid_lateral": "Lateral deltoid",
    "deltoid_posterior": "Posterior deltoid", "rotator_cuff": "Rotator cuff",
    "infraspinatus_teres_minor": "Infraspinatus / teres minor",
    "pectoralis_major": "Pectoralis major", "pectoralis_upper": "Upper pectoralis (clavicular)",
    "pectoralis_minor": "Pectoralis minor", "biceps_brachii": "Biceps brachii",
    "brachialis": "Brachialis", "brachioradialis": "Brachioradialis",
    "triceps_brachii": "Triceps brachii", "forearm_flexors": "Forearm flexors (grip)",
    "forearm_extensors": "Forearm extensors", "pronators": "Pronators", "supinator": "Supinator",
    "hand_intrinsics": "Hand (intrinsic grip muscles)", "foot_intrinsics": "Foot intrinsics",
}


def all_group_names() -> set[str]:
    return set(MUSCLE_GROUPS)


def group_label(group: str) -> str:
    return GROUP_LABELS.get(group, group.replace("_", " ").capitalize())


def expand_group(group: str, side: str | None = None) -> list[str]:
    """Mesh names for *group*: both sides, or only ``side`` (``"R"``/``"L"``)."""
    names: list[str] = []
    prefixed = group in SIDE_PREFIX_GROUPS
    for base in MUSCLE_GROUPS[group]:
        if base in UNSIDED_MUSCLES:
            names.append(base)
            continue
        for s in (("R", "L") if side is None else (side,)):
            names.append(f"{s} {base}" if prefixed else f"{base} {s}")
    return names


def regions_for_groups(groups) -> list[str]:
    """The muscle layers that must be loaded to colour *groups*, in load order."""
    order = list(ALL_MUSCLE_REGIONS)
    wanted: set[str] = set()
    for g in groups:
        wanted.add(GROUP_REGION[g])
        wanted.update(_EXTRA_REGIONS.get(g, ()))
    return [r for r in order if r in wanted]
