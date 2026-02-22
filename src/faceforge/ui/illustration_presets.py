"""Grey's Anatomy-style illustration presets.

Each preset composes layers, camera angle, anatomical labels with leader lines,
and optionally a clip plane to recreate classic atlas plates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from faceforge.ui.widgets.label_overlay import LabelDef


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class ClipPlaneDef:
    """World-space clip plane definition."""
    axis: str = "x"                # "x", "y", or "z"
    offset: float = 0.0
    flip: bool = False
    enabled: bool = True


@dataclass
class IllustrationPreset:
    """Complete specification for a Grey's Anatomy-style illustration."""
    name: str
    description: str
    icon: str
    layers: dict[str, bool]
    camera: str                    # camera preset name from _CAMERA_PRESETS
    labels: list[LabelDef]
    clip_plane: ClipPlaneDef | None = None
    opacity_overrides: dict[str, float] | None = None  # mesh_name_prefix → opacity


# ── Layer macros ─────────────────────────────────────────────────────────

_ALL_OFF = {
    "skull": False, "face": False, "jaw_muscles": False,
    "expression_muscles": False, "neck_muscles": False,
    "vertebrae": False, "eyes": False, "ears": False,
    "nose_cart": False, "eyebrows": False, "throat": False,
    "teeth": False,
    "thoracic": False, "lumbar": False, "ribs": False, "pelvis": False,
    "upper_limb_skel": False, "lower_limb_skel": False,
    "hands_skel": False, "feet_skel": False,
    "skin": False,
    "back_muscles": False, "shoulder_muscles": False, "arm_muscles": False,
    "torso_muscles": False, "hip_muscles": False, "leg_muscles": False,
    "organs": False, "vasculature": False, "brain": False,
}

_SKELETON = {
    "skull": True, "vertebrae": True, "teeth": True,
    "thoracic": True, "lumbar": True, "ribs": True, "pelvis": True,
    "upper_limb_skel": True, "lower_limb_skel": True,
    "hands_skel": True, "feet_skel": True,
}


# ── Preset definitions ──────────────────────────────────────────────────

ILLUSTRATION_PRESETS: dict[str, IllustrationPreset] = {}


def _illust(preset: IllustrationPreset) -> None:
    ILLUSTRATION_PRESETS[preset.name] = preset


_illust(IllustrationPreset(
    name="Muscles of the Head",
    description="Jaw and expression muscles",
    icon="\U0001F9B4",   # bone
    layers={
        **_ALL_OFF,
        "skull": True, "jaw_muscles": True, "expression_muscles": True,
        "eyes": True, "teeth": True,
    },
    camera="head_front",
    labels=[
        LabelDef("Masseter R", "Masseter", side="right"),
        LabelDef("Masseter L", "Masseter", side="left"),
        LabelDef("Temporalis R", "Temporalis", side="right"),
        LabelDef("Temporalis L", "Temporalis", side="left"),
        LabelDef("Orbicularis Oculi R", "Orbicularis Oculi", side="right"),
        LabelDef("Orbicularis Oculi L", "Orbicularis Oculi", side="left"),
        LabelDef("Frontalis R", "Frontalis", side="right"),
        LabelDef("Zygomaticus Major R", "Zygomaticus Major", side="right"),
        LabelDef("Zygomaticus Major L", "Zygomaticus Major", side="left"),
        LabelDef("Buccinator R", "Buccinator", side="right"),
        LabelDef("Orbicularis Oris R", "Orbicularis Oris", side="right"),
        LabelDef("Mentalis R", "Mentalis", side="left"),
    ],
))

_illust(IllustrationPreset(
    name="Muscles of the Neck",
    description="Cervical muscles and vertebrae",
    icon="\U0001F9B7",   # tooth -> neck area
    layers={
        **_ALL_OFF,
        "skull": True, "vertebrae": True, "neck_muscles": True,
        "throat": True, "teeth": True,
    },
    camera="head_three_quarter",
    labels=[
        LabelDef("SCM R", "Sternocleidomastoid", side="right"),
        LabelDef("SCM L", "Sternocleidomastoid", side="left"),
        LabelDef("Anterior Scalene R", "Anterior Scalene", side="right"),
        LabelDef("Middle Scalene R", "Middle Scalene", side="right"),
        LabelDef("Posterior Scalene R", "Posterior Scalene", side="right"),
        LabelDef("Levator Scapulae R", "Levator Scapulae", side="right"),
        LabelDef("Sternohyoid R", "Sternohyoid", side="left"),
        LabelDef("Sternothyroid R", "Sternothyroid", side="left"),
        LabelDef("Longus Colli R", "Longus Colli", side="right"),
        LabelDef("Rectus Capitis Post. Major R", "Rectus Capitis Post. Major", side="right"),
    ],
))

_illust(IllustrationPreset(
    name="Cervical Vertebrae",
    description="C1-C7 vertebrae and discs",
    icon="\u26D3",   # chain links
    layers={
        **_ALL_OFF,
        "skull": True, "vertebrae": True,
    },
    camera="head_right",
    labels=[
        LabelDef("C1 Atlas", "C1 (Atlas)", side="right"),
        LabelDef("C2 Axis", "C2 (Axis)", side="right"),
        LabelDef("C3", "C3", side="right"),
        LabelDef("C4", "C4", side="right"),
        LabelDef("C5", "C5", side="right"),
        LabelDef("C6", "C6", side="right"),
        LabelDef("C7", "C7", side="right"),
        LabelDef("C2-C3 Disc", "C2-C3 Disc", side="left"),
        LabelDef("C5-C6 Disc", "C5-C6 Disc", side="left"),
        LabelDef("T1", "T1", side="right"),
    ],
    opacity_overrides={"cranium": 0.3},
))

_illust(IllustrationPreset(
    name="Midsagittal Section",
    description="Brain and skull, sagittal cut",
    icon="\U0001F9E0",   # brain
    layers={
        **_ALL_OFF,
        "skull": True, "brain": True, "vertebrae": True,
    },
    camera="head_right",
    labels=[
        LabelDef("Cerebrum R", "Cerebrum", side="right"),
        LabelDef("Cerebellum R", "Cerebellum", side="right"),
        LabelDef("Brainstem", "Brainstem", side="left"),
        LabelDef("Corpus Callosum", "Corpus Callosum", side="right"),
        LabelDef("Thalamus R", "Thalamus", side="left"),
        LabelDef("Hypothalamus", "Hypothalamus", side="left"),
        LabelDef("Pons", "Pons", side="left"),
        LabelDef("Medulla Oblongata", "Medulla Oblongata", side="left"),
    ],
    clip_plane=ClipPlaneDef(axis="x", offset=0, flip=False),
))

_illust(IllustrationPreset(
    name="Thorax Anterior",
    description="Ribcage, thoracic organs",
    icon="\U0001FAC1",   # lungs
    layers={
        **_ALL_OFF,
        "ribs": True, "thoracic": True, "organs": True,
        "skin": True,
    },
    camera="body_front",
    labels=[
        LabelDef("Heart", "Heart", side="left"),
        LabelDef("Right Lung", "Right Lung", side="right"),
        LabelDef("Left Lung", "Left Lung", side="left"),
        LabelDef("Sternum", "Sternum", side="right"),
        LabelDef("Diaphragm", "Diaphragm", side="left"),
        LabelDef("Rib 1 R", "1st Rib", side="right"),
        LabelDef("Rib 5 R", "5th Rib", side="right"),
        LabelDef("Rib 10 R", "10th Rib", side="right"),
        LabelDef("Trachea", "Trachea", side="right"),
        LabelDef("Thymus", "Thymus", side="left"),
    ],
    opacity_overrides={"skin": 0.15},
))

_illust(IllustrationPreset(
    name="Cardiovascular",
    description="Heart and major blood vessels",
    icon="\u2665",   # heart
    layers={
        **_ALL_OFF,
        **_SKELETON,
        "organs": True, "vasculature": True,
    },
    camera="body_front",
    labels=[
        LabelDef("Heart", "Heart", side="left"),
        LabelDef("Aorta", "Aorta", side="left"),
        LabelDef("Superior Vena Cava", "Superior Vena Cava", side="right"),
        LabelDef("Inferior Vena Cava", "Inferior Vena Cava", side="right"),
        LabelDef("Pulmonary Artery", "Pulmonary Arteries", side="left"),
        LabelDef("Common Carotid R", "Common Carotid", side="right"),
        LabelDef("Subclavian Artery R", "Subclavian Artery", side="right"),
        LabelDef("Femoral Artery R", "Femoral Artery", side="right"),
    ],
))

_illust(IllustrationPreset(
    name="Upper Limb Musculature",
    description="Shoulder and arm muscles",
    icon="\U0001F4AA",   # flexed bicep
    layers={
        **_ALL_OFF,
        "upper_limb_skel": True, "hands_skel": True,
        "shoulder_muscles": True, "arm_muscles": True,
        "thoracic": True, "ribs": True,
    },
    camera="body_right",
    labels=[
        LabelDef("Deltoid R", "Deltoid", side="right"),
        LabelDef("Biceps Brachii R", "Biceps Brachii", side="right"),
        LabelDef("Triceps Brachii R", "Triceps Brachii", side="left"),
        LabelDef("Brachioradialis R", "Brachioradialis", side="right"),
        LabelDef("Brachialis R", "Brachialis", side="left"),
        LabelDef("Pronator Teres R", "Pronator Teres", side="right"),
        LabelDef("Extensor Digitorum R", "Extensor Digitorum", side="left"),
        LabelDef("Flexor Carpi Radialis R", "Flexor Carpi Radialis", side="right"),
        LabelDef("Supraspinatus R", "Supraspinatus", side="left"),
        LabelDef("Infraspinatus R", "Infraspinatus", side="left"),
    ],
))

_illust(IllustrationPreset(
    name="Lower Limb Musculature",
    description="Hip and leg muscles",
    icon="\U0001F9B5",   # leg
    layers={
        **_ALL_OFF,
        "lower_limb_skel": True, "feet_skel": True, "pelvis": True,
        "hip_muscles": True, "leg_muscles": True,
    },
    camera="body_front",
    labels=[
        LabelDef("Rectus Femoris R", "Rectus Femoris", side="right"),
        LabelDef("Vastus Lateralis R", "Vastus Lateralis", side="right"),
        LabelDef("Vastus Medialis R", "Vastus Medialis", side="left"),
        LabelDef("Biceps Femoris R", "Biceps Femoris", side="right"),
        LabelDef("Gastrocnemius R", "Gastrocnemius", side="right"),
        LabelDef("Tibialis Anterior R", "Tibialis Anterior", side="left"),
        LabelDef("Soleus R", "Soleus", side="left"),
        LabelDef("Gluteus Maximus R", "Gluteus Maximus", side="right"),
        LabelDef("Sartorius R", "Sartorius", side="left"),
        LabelDef("Adductor Magnus R", "Adductor Magnus", side="left"),
    ],
))

_illust(IllustrationPreset(
    name="Deep Back Muscles",
    description="Spine and back musculature",
    icon="\U0001F9B4",   # bone
    layers={
        **_ALL_OFF,
        **_SKELETON,
        "back_muscles": True,
    },
    camera="body_back",
    labels=[
        LabelDef("Trapezius R", "Trapezius", side="right"),
        LabelDef("Trapezius L", "Trapezius", side="left"),
        LabelDef("Latissimus Dorsi R", "Latissimus Dorsi", side="right"),
        LabelDef("Latissimus Dorsi L", "Latissimus Dorsi", side="left"),
        LabelDef("Erector Spinae R", "Erector Spinae", side="right"),
        LabelDef("Rhomboid Major R", "Rhomboid Major", side="right"),
        LabelDef("Rhomboid Minor R", "Rhomboid Minor", side="right"),
        LabelDef("Serratus Posterior R", "Serratus Posterior Inf.", side="left"),
        LabelDef("Splenius Capitis R", "Splenius Capitis", side="left"),
        LabelDef("Levator Scapulae R", "Levator Scapulae", side="left"),
    ],
))

_illust(IllustrationPreset(
    name="Abdominal Organs",
    description="Digestive and urinary organs",
    icon="\U0001FAC0",   # anatomical heart → abdominal
    layers={
        **_ALL_OFF,
        **_SKELETON,
        "organs": True,
    },
    camera="body_front",
    labels=[
        LabelDef("Liver", "Liver", side="right"),
        LabelDef("Stomach", "Stomach", side="left"),
        LabelDef("Right Kidney", "Right Kidney", side="right"),
        LabelDef("Left Kidney", "Left Kidney", side="left"),
        LabelDef("Small Intestine", "Small Intestine", side="left"),
        LabelDef("Large Intestine", "Large Intestine", side="right"),
        LabelDef("Spleen", "Spleen", side="left"),
        LabelDef("Pancreas", "Pancreas", side="left"),
        LabelDef("Gallbladder", "Gallbladder", side="right"),
        LabelDef("Urinary Bladder", "Urinary Bladder", side="left"),
    ],
    clip_plane=ClipPlaneDef(axis="y", offset=0, flip=True),
))


# ── Apply function ───────────────────────────────────────────────────────

def apply_illustration_preset(
    preset_name: str,
    layers_tab,
    event_bus,
    gl_widget,
    display_tab,
    label_overlay,
    scene,
) -> None:
    """Apply an illustration preset: layers, camera, clip, labels.

    Parameters
    ----------
    preset_name : str
        Key into ``ILLUSTRATION_PRESETS``.
    layers_tab : LayersTab
        For programmatic layer toggling.
    event_bus : EventBus
        For publishing events.
    gl_widget : GLViewport
        For camera and renderer access.
    display_tab : DisplayTab
        For clip plane programmatic control.
    label_overlay : LabelOverlay
        For setting illustration labels.
    scene : Scene
        For collecting mesh world positions.
    """
    from faceforge.ui.startup_dialog import apply_preset, PRESETS
    from faceforge.core.events import EventType
    from faceforge.core.math_utils import transform_point

    preset = ILLUSTRATION_PRESETS.get(preset_name)
    if preset is None:
        return

    # 1. Build a temporary configuration-style preset dict and apply layers + camera
    _temp_preset_name = f"__illust__{preset_name}"
    PRESETS[_temp_preset_name] = {
        "description": preset.description,
        "icon": preset.icon,
        "layers": preset.layers,
        "camera": preset.camera,
    }
    apply_preset(_temp_preset_name, layers_tab, event_bus, gl_widget=gl_widget)
    # Clean up temporary entry
    del PRESETS[_temp_preset_name]

    # 2. Set clip plane
    if preset.clip_plane is not None and preset.clip_plane.enabled:
        display_tab.set_clip_plane(
            enabled=True,
            axis=preset.clip_plane.axis,
            offset=preset.clip_plane.offset,
            flip=preset.clip_plane.flip,
        )
    else:
        display_tab.set_clip_plane(enabled=False)

    # 3. Apply opacity overrides
    if preset.opacity_overrides:
        meshes = scene.collect_meshes()
        for mesh, _ in meshes:
            if mesh.name:
                for prefix, opacity in preset.opacity_overrides.items():
                    if mesh.name.startswith(prefix):
                        mesh.material.opacity = opacity
                        break

    # 4. Build label world positions from scene meshes
    _build_illustration_labels(preset, label_overlay, scene)

    # 5. Enable label overlay
    label_overlay.set_enabled(True)
    event_bus.publish(EventType.LABELS_TOGGLED, enabled=True)


def rebuild_illustration_labels(label_overlay, scene) -> None:
    """Rebuild illustration label positions after layers load.

    Called when `_labels_dirty` is set and illustration mode is active.
    """
    if not label_overlay._illustration_mode or not label_overlay._illustration_labels:
        return

    _build_label_positions(label_overlay._illustration_labels, label_overlay, scene)


def _build_illustration_labels(preset, label_overlay, scene) -> None:
    """Compute world positions for illustration labels and set them on the overlay."""
    _build_label_positions(preset.labels, label_overlay, scene)


def _build_label_positions(labels, label_overlay, scene) -> None:
    """Compute world positions for a list of LabelDef items."""
    from faceforge.core.math_utils import transform_point

    meshes = scene.collect_meshes()
    mesh_positions: dict[str, tuple] = {}
    for mesh, world_mat in meshes:
        if mesh.name and mesh.name not in mesh_positions:
            center = mesh.geometry.get_bounding_center()
            world_center = transform_point(world_mat, center)
            mesh_positions[mesh.name] = tuple(world_center)

    label_overlay.set_illustration_labels(labels, mesh_positions)
