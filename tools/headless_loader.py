"""Headless skeleton + tissue loading for automated diagnostics.

Replicates the loading sequence from ``app.py:load_assets()`` (skeleton,
joint chains, soft tissue skinning) without any Qt or OpenGL imports.

Usage::

    hs = load_headless_scene()
    meshes = load_layer(hs, "skin")
    register_layer(hs, meshes, "skin")
    apply_pose(hs, some_body_state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from faceforge.body.body_animation import BodyAnimationSystem
from faceforge.body.body_constraints import BodyConstraints
from faceforge.body.joint_pivots import JointPivotSetup
from faceforge.body.skeleton import SkeletonBuilder
from faceforge.body.soft_tissue import SoftTissueSkinning
from faceforge.coordination.loading_pipeline import LoadingPipeline
from faceforge.coordination.scene_builder import SceneBuilder
from faceforge.coordination.visibility import VisibilityManager
from faceforge.core.config_loader import load_config, load_muscle_config
from faceforge.core.events import EventBus
from faceforge.core.math_utils import Quat, quat_identity
from faceforge.core.mesh import MeshInstance
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.core.state import BodyState, FaceState, ConstraintState
from faceforge.loaders.asset_manager import AssetManager

logger = logging.getLogger(__name__)


# ── Muscle chain mapping (mirrors app.py) ────────────────────────────

_MUSCLE_CHAIN_MAP: dict[str, list[str]] = {
    "back_muscles":     ["spine", "ribs"],
    "torso_muscles":    ["spine", "ribs"],
    "shoulder_muscles": ["spine", "arm"],
    "arm_muscles":      ["spine", "arm", "forearm", "hand"],
    "hip_muscles":      ["spine", "leg"],
    "leg_muscles":      ["spine", "leg", "lower_leg", "foot"],
}

_MUSCLE_CHAIN_OVERRIDES: dict[str, list[str]] = {}
for _n in ("Serratus Ant.", "Subclavius"):
    for _s in ("R", "L"):
        _MUSCLE_CHAIN_OVERRIDES[f"{_n} {_s}"] = ["spine", "ribs"]
for _n in ("Pect. Major Clav.", "Pect. Major Stern.", "Pect. Major Abd.",
           "Pect. Minor"):
    for _s in ("R", "L"):
        _MUSCLE_CHAIN_OVERRIDES[f"{_n} {_s}"] = ["spine", "ribs", "arm"]
for _n in ("Ext. Intercostal", "Int. Intercostal", "Innermost Intercostal",
           "Diaphragm", "Linea Alba"):
    _MUSCLE_CHAIN_OVERRIDES[_n] = ["spine", "ribs"]
for _n in ("Trans. Thoracis", "Lev. Costarum Longi", "Lev. Costarum Breves"):
    for _s in ("R", "L"):
        _MUSCLE_CHAIN_OVERRIDES[f"{_n} {_s}"] = ["spine", "ribs"]
for _n in ("Asc. Trapezius", "Trans. Trapezius", "Desc. Trapezius",
           "Latissimus Dorsi", "Rhomboid Major", "Rhomboid Minor"):
    for _s in ("R", "L"):
        _MUSCLE_CHAIN_OVERRIDES[f"{_n} {_s}"] = ["spine", "ribs", "arm"]
for _n in ("Serratus Post. Sup.", "Serratus Post. Inf."):
    for _s in ("R", "L"):
        _MUSCLE_CHAIN_OVERRIDES[f"{_n} {_s}"] = ["spine", "ribs"]
del _n, _s

# Layer name → AssetManager loader method + config info
_LAYER_INFO: dict[str, dict] = {
    "skin":              {"loader": "load_skin",       "type": "skin"},
    "back_muscles":      {"loader": "load_body_muscles", "config": "back_muscles.json",      "type": "muscle"},
    "shoulder_muscles":  {"loader": "load_body_muscles", "config": "shoulder_muscles.json",   "type": "muscle"},
    "arm_muscles":       {"loader": "load_body_muscles", "config": "arm_muscles.json",        "type": "muscle"},
    "torso_muscles":     {"loader": "load_body_muscles", "config": "torso_muscles.json",      "type": "muscle"},
    "hip_muscles":       {"loader": "load_body_muscles", "config": "hip_muscles.json",        "type": "muscle"},
    "leg_muscles":       {"loader": "load_body_muscles", "config": "leg_muscles.json",        "type": "muscle"},
    "organs":            {"loader": "load_organs",      "type": "organ"},
    "vasculature":       {"loader": "load_vasculature", "type": "vascular"},
}


@dataclass
class HeadlessScene:
    """Holds all state for a headless skinning session."""
    scene: Scene
    named_nodes: dict[str, SceneNode]
    skinning: SoftTissueSkinning
    body_animation: Optional[BodyAnimationSystem]
    body_constraints: BodyConstraints
    joint_chains: list[list[tuple[str, SceneNode]]]
    chain_ids: dict[str, int]
    pipeline: LoadingPipeline
    skeleton: Optional[SkeletonBuilder]
    assets: AssetManager
    body_state: BodyState = field(default_factory=BodyState)
    face_state: FaceState = field(default_factory=FaceState)
    constraint_state: ConstraintState = field(default_factory=ConstraintState)
    _loaded_layers: set[str] = field(default_factory=set)
    _head_quat: Quat = field(default_factory=quat_identity)


def load_headless_scene() -> HeadlessScene:
    """Load skeleton, build joint chains, create SoftTissueSkinning.

    Mirrors ``app.py:load_assets()`` lines 917-1118 without Qt/GL.
    """
    assets = AssetManager()
    assets.init_transform()

    event_bus = EventBus()
    visibility = VisibilityManager()

    builder = SceneBuilder(assets, visibility)
    scene, named_nodes = builder.build()

    pipeline = LoadingPipeline(assets, event_bus, named_nodes)

    # Phase 0-2: Head (skull + face + muscles + vertebrae)
    try:
        pipeline.load_head()
    except Exception as e:
        logger.warning("Head loading incomplete: %s", e)

    # Phase 3: Body skeleton
    try:
        pipeline.load_body_skeleton()
    except Exception as e:
        logger.warning("Body skeleton loading incomplete: %s", e)

    # Body animation system
    body_anim: Optional[BodyAnimationSystem] = None
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
            if pipeline.skeleton.rib_nodes:
                body_anim.set_rib_nodes(pipeline.skeleton.rib_nodes)

    # Body constraints
    body_constraints = BodyConstraints()
    body_constraints.load()

    # Soft tissue skinning
    skinning = SoftTissueSkinning()

    # Build kinematic chains (mirrors app.py lines 1035-1110)
    joint_chains: list[list[tuple[str, SceneNode]]] = []
    chain_ids: dict[str, int] = {}

    # Chain 0: Spine (thoracic + lumbar)
    spine_chain: list[tuple[str, SceneNode]] = []
    if pipeline.skeleton is not None:
        for pinfo in pipeline.skeleton.pivots.get("thoracic", []):
            spine_chain.append((f"thoracic_{pinfo.get('level', 0)}", pinfo["group"]))
        for pinfo in pipeline.skeleton.pivots.get("lumbar", []):
            spine_chain.append((f"lumbar_{pinfo.get('level', 0)}", pinfo["group"]))
    if spine_chain:
        chain_ids["spine"] = len(joint_chains)
        joint_chains.append(spine_chain)

    # Limb chains (split into upper/lower per limb)
    if pipeline.joint_setup is not None:
        jp = pipeline.joint_setup.pivots
        for side in ("R", "L"):
            # Upper arm chain: shoulder → elbow
            upper_arm: list[tuple[str, SceneNode]] = []
            for jn in ("shoulder", "elbow"):
                node = jp.get(f"{jn}_{side}")
                if node is not None:
                    upper_arm.append((f"{jn}_{side}", node))
            if upper_arm:
                chain_ids[f"arm_{side}"] = len(joint_chains)
                joint_chains.append(upper_arm)

            # Forearm chain: elbow → wrist
            forearm: list[tuple[str, SceneNode]] = []
            for jn in ("elbow", "wrist"):
                node = jp.get(f"{jn}_{side}")
                if node is not None:
                    forearm.append((f"{jn}_{side}", node))
            if forearm:
                chain_ids[f"forearm_{side}"] = len(joint_chains)
                joint_chains.append(forearm)

            # Upper leg chain: hip → knee
            upper_leg: list[tuple[str, SceneNode]] = []
            for jn in ("hip", "knee"):
                node = jp.get(f"{jn}_{side}")
                if node is not None:
                    upper_leg.append((f"{jn}_{side}", node))
            if upper_leg:
                chain_ids[f"leg_{side}"] = len(joint_chains)
                joint_chains.append(upper_leg)

            # Lower leg chain: knee → ankle
            lower_leg: list[tuple[str, SceneNode]] = []
            for jn in ("knee", "ankle"):
                node = jp.get(f"{jn}_{side}")
                if node is not None:
                    lower_leg.append((f"{jn}_{side}", node))
            if lower_leg:
                chain_ids[f"lower_leg_{side}"] = len(joint_chains)
                joint_chains.append(lower_leg)

    # Digit chains
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
                    chain_ids[f"hand_{side}_{digit}"] = len(joint_chains)
                    joint_chains.append(hand_chain)

                foot_chain: list[tuple[str, SceneNode]] = []
                for seg in ("mt", "prox", "mid", "dist"):
                    p = jp.get(f"toe_{side}_{digit}_{seg}")
                    if p is not None:
                        foot_chain.append((f"toe_{side}_{digit}_{seg}", p))
                if foot_chain:
                    chain_ids[f"foot_{side}_{digit}"] = len(joint_chains)
                    joint_chains.append(foot_chain)

    # Rib chain
    if body_anim is not None and body_anim._rib_pivots:
        rib_chain: list[tuple[str, SceneNode]] = []
        for i, pivot in enumerate(body_anim._rib_pivots):
            rib_chain.append((f"rib_{i}", pivot))
        if rib_chain:
            chain_ids["ribs"] = len(joint_chains)
            joint_chains.append(rib_chain)

    if joint_chains:
        scene.update()
        skinning.build_skin_joints(joint_chains)
        logger.info("Skin joints built: %d joints in %d chains",
                     len(skinning.joints), len(joint_chains))

    return HeadlessScene(
        scene=scene,
        named_nodes=named_nodes,
        skinning=skinning,
        body_animation=body_anim,
        body_constraints=body_constraints,
        joint_chains=joint_chains,
        chain_ids=chain_ids,
        pipeline=pipeline,
        skeleton=pipeline.skeleton,
        assets=assets,
    )


def _resolve_chain_set(chain_names: list[str], chain_ids: dict[str, int]) -> set[int] | None:
    """Convert chain name list to chain ID set."""
    chains = set()
    for cn in chain_names:
        cid = chain_ids.get(cn)
        if cid is not None:
            chains.add(cid)
    return chains if chains else None


def _resolve_sided_chains(
    chain_names: list[str], muscle_name: str, chain_ids: dict[str, int],
) -> set[int] | None:
    """Resolve side-neutral chain tokens to muscle's side."""
    side = None
    if muscle_name.endswith(" R"):
        side = "R"
    elif muscle_name.endswith(" L"):
        side = "L"

    resolved: list[str] = []
    for cn in chain_names:
        if cn in ("arm", "leg", "forearm", "lower_leg"):
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
    return _resolve_chain_set(resolved, chain_ids)


def load_layer(hs: HeadlessScene, layer_name: str) -> list[MeshInstance]:
    """Load a tissue layer's STL meshes and add to scene graph.

    Parameters
    ----------
    hs : HeadlessScene
        The headless scene from :func:`load_headless_scene`.
    layer_name : str
        One of: ``"skin"``, ``"back_muscles"``, ``"shoulder_muscles"``,
        ``"arm_muscles"``, ``"torso_muscles"``, ``"hip_muscles"``,
        ``"leg_muscles"``, ``"organs"``, ``"vasculature"``.

    Returns
    -------
    list[MeshInstance]
        The loaded meshes (also added to scene graph under bodyRoot).
    """
    info = _LAYER_INFO.get(layer_name)
    if info is None:
        raise ValueError(f"Unknown layer: {layer_name!r}. "
                         f"Known layers: {sorted(_LAYER_INFO)}")

    body_root = hs.named_nodes.get("bodyRoot")
    if body_root is None:
        raise RuntimeError("No bodyRoot in scene")

    loader_name = info["loader"]
    if loader_name == "load_body_muscles":
        result = hs.assets.load_body_muscles(info["config"])
    else:
        result = getattr(hs.assets, loader_name)()

    body_root.add(result.group)
    hs._loaded_layers.add(layer_name)
    logger.info("Loaded layer %s: %d meshes", layer_name, len(result.meshes))
    return result.meshes


def register_layer(
    hs: HeadlessScene,
    meshes: list[MeshInstance],
    layer_name: str,
    params: dict | None = None,
) -> None:
    """Register meshes with skinning using given params.

    Parameters
    ----------
    hs : HeadlessScene
        The headless scene.
    meshes : list[MeshInstance]
        Meshes returned by :func:`load_layer`.
    layer_name : str
        Layer name (same as passed to :func:`load_layer`).
    params : dict, optional
        Override skinning parameters. Keys can include ``spatial_limit``,
        ``chain_z_margin``, and any ``SoftTissueSkinning`` instance attribute
        (``min_spatial``, ``spatial_factor``, ``min_z_pad``,
        ``lateral_threshold``, ``midline_tolerance``, ``CROSS_CHAIN_RADIUS``,
        ``MAX_CROSS_WEIGHT_MUSCLE``, ``MAX_CROSS_WEIGHT_OTHER``,
        ``BLEND_ZONE``).
    """
    params = params or {}
    info = _LAYER_INFO.get(layer_name)
    if info is None:
        raise ValueError(f"Unknown layer: {layer_name!r}")

    skinning = hs.skinning
    chain_ids = hs.chain_ids
    layer_type = info["type"]

    # Apply instance-level parameter overrides
    _INSTANCE_ATTRS = (
        "min_spatial", "spatial_factor", "min_z_pad",
        "lateral_threshold", "midline_tolerance",
    )
    _CLASS_ATTRS = (
        "CROSS_CHAIN_RADIUS", "MAX_CROSS_WEIGHT_MUSCLE",
        "MAX_CROSS_WEIGHT_OTHER", "BLEND_ZONE",
        "DIVERGENCE_MIN", "DIVERGENCE_MAX",
    )
    for attr in _INSTANCE_ATTRS:
        if attr in params:
            setattr(skinning, attr, params[attr])
    for attr in _CLASS_ATTRS:
        if attr in params:
            setattr(skinning, attr, params[attr])

    if layer_type == "skin":
        all_chains = set(chain_ids.values()) if chain_ids else None
        sl = params.get("spatial_limit", 25.0)
        zm = params.get("chain_z_margin", 15.0)
        for mesh in meshes:
            skinning.register_skin_mesh(
                mesh, is_muscle=False, allowed_chains=all_chains,
                chain_z_margin=zm, spatial_limit=sl,
            )

    elif layer_type == "muscle":
        config_name = info["config"]
        defs = load_muscle_config(config_name)
        default_chain_names = _MUSCLE_CHAIN_MAP.get(layer_name, ["spine"])

        for mesh, defn in zip(meshes, defs):
            muscle_name = defn.get("name", mesh.name)
            override = _MUSCLE_CHAIN_OVERRIDES.get(muscle_name)
            chain_names = override if override else default_chain_names
            ac = _resolve_sided_chains(chain_names, muscle_name, chain_ids)
            skinning.register_skin_mesh(mesh, is_muscle=True, allowed_chains=ac)

        # Remove digit/limb cross-chain blending for arm and leg muscles
        if layer_name in ("arm_muscles", "leg_muscles"):
            digit_cids: set[int] = set()
            prefix = "hand" if layer_name == "arm_muscles" else "foot"
            for side in ("R", "L"):
                for digit in range(1, 6):
                    cid = chain_ids.get(f"{prefix}_{side}_{digit}")
                    if cid is not None:
                        digit_cids.add(cid)
            if digit_cids:
                skinning.snap_hierarchy_blends(digit_cids)
                skinning.reassign_orphan_vertices(digit_cids)

    elif layer_type == "organ":
        spine_id = chain_ids.get("spine")
        ac = {spine_id} if spine_id is not None else None
        for mesh in meshes:
            skinning.register_skin_mesh(mesh, is_muscle=False, allowed_chains=ac)

    elif layer_type == "vascular":
        spine_id = chain_ids.get("spine")
        ac = {spine_id} if spine_id is not None else None
        for mesh in meshes:
            skinning.register_skin_mesh(mesh, is_muscle=False, allowed_chains=ac)


def apply_pose(hs: HeadlessScene, body_state: BodyState) -> None:
    """Apply a pose: constraints -> animation -> scene update -> skinning update.

    Parameters
    ----------
    hs : HeadlessScene
        The headless scene.
    body_state : BodyState
        The desired body pose.
    """
    hs.body_state = body_state

    # Clamp to joint limits
    hs.body_constraints.clamp(body_state)

    # Apply body animation (spine, limbs, breathing)
    if hs.body_animation is not None:
        hs.body_animation.apply(body_state, dt=0.016)

    # Update scene graph matrices
    hs.scene.update()

    # Force skinning signature invalidation so update() actually runs
    hs.skinning._last_signature = ""

    # Run soft tissue skinning
    hs.skinning.update(body_state)


def apply_head_rotation(hs: HeadlessScene, face_state: FaceState) -> Quat:
    """Apply head rotation and update neck muscles.

    Parameters
    ----------
    hs : HeadlessScene
        The headless scene.
    face_state : FaceState
        Face state with ``head_yaw``, ``head_pitch``, ``head_roll``.

    Returns
    -------
    Quat
        The computed head rotation quaternion.
    """
    hs.face_state = face_state
    pipeline = hs.pipeline
    head_rot = pipeline.head_rotation

    if head_rot is None:
        logger.warning("Head rotation system not loaded")
        return quat_identity()

    skull_group = hs.named_nodes.get("skullGroup")
    face_group = hs.named_nodes.get("faceGroup")
    if skull_group is None or face_group is None:
        logger.warning("Missing skull/face group nodes")
        return quat_identity()

    head_q = head_rot.apply(
        face_state,
        skull_group,
        face_group,
        pipeline.vertebrae_pivots or None,
        hs.constraint_state,
        brain_group=hs.named_nodes.get("brainGroup"),
        stl_muscle_group=hs.named_nodes.get("stlMuscleGroup"),
        expr_muscle_group=hs.named_nodes.get("exprMuscleGroup"),
        face_feature_group=hs.named_nodes.get("faceFeatureGroup"),
    )
    hs._head_quat = head_q

    # Update neck muscles
    if pipeline.neck_muscles is not None:
        pipeline.neck_muscles.update(head_q, face_state=face_state, body_state=hs.body_state)

    # Platysma correction (after head rotation)
    if pipeline.platysma is not None and pipeline.platysma.registered:
        plat_bone_cur = None
        plat_bone_rest = None
        if pipeline.bone_anchors is not None:
            plat_bone_cur = pipeline.bone_anchors.get_muscle_anchor_current(
                "Platysma", ["Right Clavicle", "Left Clavicle"],
            )
            plat_bone_rest = pipeline.bone_anchors.get_muscle_anchor(
                "Platysma", ["Right Clavicle", "Left Clavicle"],
            )
        pipeline.platysma.update(
            head_q,
            bone_anchor_current=plat_bone_cur,
            bone_anchor_rest=plat_bone_rest,
        )

    # Update neck constraints
    if pipeline.neck_constraints is not None:
        pipeline.neck_constraints.solve(
            hs.constraint_state,
            pipeline.neck_muscles.muscle_data if pipeline.neck_muscles else [],
            head_q,
        )

    # Apply head rotation to skin meshes
    _apply_head_to_skin_bindings(hs, head_q)

    hs.scene.update()
    return head_q


def _apply_head_to_skin_bindings(hs: HeadlessScene, head_q: Quat) -> None:
    """Apply head rotation to all skin bindings' mesh vertices."""
    head_rot = hs.pipeline.head_rotation
    if head_rot is None:
        return
    for binding in hs.skinning.bindings:
        mesh = binding.mesh
        if mesh.rest_positions is None:
            continue
        head_rot.apply_to_skin(
            mesh.geometry.positions,
            mesh.rest_positions,
            head_q,
        )
        mesh.needs_update = True


def apply_full_pose(
    hs: HeadlessScene,
    body_state: BodyState,
    face_state: FaceState,
) -> Quat:
    """Apply combined body pose and head rotation in correct order.

    Order:
    1. Body constraints (clamp)
    2. Body animation (spine, limbs, breathing)
    3. Head rotation (skull, face, brain, +3 groups)
    4. Neck muscles (uses head quaternion + body state)
    5. Neck constraints
    6. Soft tissue skinning
    7. Scene graph update

    Parameters
    ----------
    hs : HeadlessScene
        The headless scene.
    body_state : BodyState
        The desired body pose.
    face_state : FaceState
        Face state with head rotation values.

    Returns
    -------
    Quat
        The computed head rotation quaternion.
    """
    hs.body_state = body_state
    hs.face_state = face_state
    pipeline = hs.pipeline

    # 1. Clamp to joint limits
    hs.body_constraints.clamp(body_state)

    # 2. Body animation (spine, limbs, breathing)
    if hs.body_animation is not None:
        hs.body_animation.apply(body_state, dt=0.016)

    # 3. Head rotation
    head_rot = pipeline.head_rotation
    head_q = quat_identity()
    skull_group = hs.named_nodes.get("skullGroup")
    face_group = hs.named_nodes.get("faceGroup")

    if head_rot is not None and skull_group is not None:
        head_q = head_rot.apply(
            face_state,
            skull_group,
            face_group,
            pipeline.vertebrae_pivots or None,
            hs.constraint_state,
            brain_group=hs.named_nodes.get("brainGroup"),
            stl_muscle_group=hs.named_nodes.get("stlMuscleGroup"),
            expr_muscle_group=hs.named_nodes.get("exprMuscleGroup"),
            face_feature_group=hs.named_nodes.get("faceFeatureGroup"),
        )
    hs._head_quat = head_q

    # 4. Neck muscles
    if pipeline.neck_muscles is not None:
        pipeline.neck_muscles.update(head_q, face_state=face_state, body_state=body_state)

    # 4.5. Platysma correction (after head rotation + body animation)
    if pipeline.platysma is not None and pipeline.platysma.registered:
        plat_bone_cur = None
        plat_bone_rest = None
        if pipeline.bone_anchors is not None:
            plat_bone_cur = pipeline.bone_anchors.get_muscle_anchor_current(
                "Platysma", ["Right Clavicle", "Left Clavicle"],
            )
            plat_bone_rest = pipeline.bone_anchors.get_muscle_anchor(
                "Platysma", ["Right Clavicle", "Left Clavicle"],
            )
        pipeline.platysma.update(
            head_q,
            bone_anchor_current=plat_bone_cur,
            bone_anchor_rest=plat_bone_rest,
        )

    # 5. Neck constraints
    if pipeline.neck_constraints is not None:
        pipeline.neck_constraints.solve(
            hs.constraint_state,
            pipeline.neck_muscles.muscle_data if pipeline.neck_muscles else [],
            head_q,
        )

    # 6. Scene graph update (before skinning, so world matrices are current)
    hs.scene.update()

    # 7. Soft tissue skinning
    hs.skinning._last_signature = ""
    hs.skinning.update(body_state)

    # 8. Apply head rotation to skin meshes
    head_rot = pipeline.head_rotation
    if head_rot is not None:
        _apply_head_to_skin_bindings(hs, head_q)

    return head_q


def reset_skinning(hs: HeadlessScene) -> None:
    """Clear bindings, restore mesh rest positions, rebuild joints.

    After calling this, layers can be re-registered with different
    parameters for optimization sweeps.
    """
    # Restore rest positions for all bound meshes
    for binding in hs.skinning.bindings:
        mesh = binding.mesh
        if mesh.rest_positions is not None:
            mesh.geometry.positions = mesh.rest_positions.copy()
            if mesh.rest_normals is not None:
                mesh.geometry.normals = mesh.rest_normals.copy()
            mesh.needs_update = True

    # Clear bindings
    hs.skinning.bindings.clear()
    hs.skinning._last_signature = ""

    # Reset head rotation to identity
    if hs.pipeline.head_rotation is not None:
        hs.pipeline.head_rotation.reset()
    if hs.pipeline.neck_muscles is not None:
        hs.pipeline.neck_muscles.reset()
    hs.face_state = FaceState()
    hs._head_quat = quat_identity()

    # Reset body state to anatomical rest BEFORE rebuilding joints.
    # Without this, build_skin_joints snapshots the current (posed) joint
    # positions as rest matrices, producing wrong delta transforms.
    rest_state = BodyState()
    hs.body_constraints.clamp(rest_state)
    if hs.body_animation is not None:
        hs.body_animation.apply(rest_state, dt=0.0)
    hs.body_state = rest_state
    hs.scene.update()
    hs.skinning.build_skin_joints(hs.joint_chains)
