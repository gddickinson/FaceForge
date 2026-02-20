"""Sequential asset loading chain with progress reporting."""

import logging
from typing import Optional

from faceforge.core.scene_graph import SceneNode
from faceforge.core.mesh import MeshInstance
from faceforge.core.events import EventBus, EventType
from faceforge.core.config_loader import load_config, load_muscle_config, load_skeleton_config
from faceforge.loaders.asset_manager import AssetManager
from faceforge.loaders.stl_batch_loader import CoordinateTransform
from faceforge.anatomy.skull import build_skull, get_jaw_pivot_node
from faceforge.anatomy.face import build_face
from faceforge.anatomy.jaw_muscles import JawMuscleSystem
from faceforge.anatomy.expression_muscles import ExpressionMuscleSystem
from faceforge.anatomy.face_features import FaceFeatureSystem
from faceforge.anatomy.neck_muscles import NeckMuscleSystem
from faceforge.anatomy.vertebrae import VertebraeSystem
from faceforge.anatomy.neck_constraints import NeckConstraintSolver
from faceforge.anatomy.head_rotation import HeadRotationSystem
from faceforge.anatomy.facs import FACSEngine
from faceforge.body.skeleton import SkeletonBuilder
from faceforge.body.joint_pivots import JointPivotSetup
from faceforge.anatomy.bone_anchors import BoneAnchorRegistry
from faceforge.anatomy.platysma import PlatysmaHandler
from faceforge.anatomy.fascia import FasciaSystem, build_anatomical_fascia_regions
from faceforge.constants import STL_DIR, set_jaw_pivot

logger = logging.getLogger(__name__)


class LoadingPipeline:
    """Orchestrates the sequential loading of all assets.

    Loading order (mirrors JS):
    Phase 0: Skull mesh (from embedded data)
    Phase 1: Face mesh
    Phase 2: Jaw muscles → expression muscles → face features → neck muscles → vertebrae
    Phase 3: Body skeleton (thoracic, lumbar, ribs, pelvis, limbs, hands, feet)
    Phase 4+: On-demand (body muscles, organs, vasculature, brain)
    """

    def __init__(
        self,
        assets: AssetManager,
        event_bus: EventBus,
        named_nodes: dict[str, SceneNode],
    ):
        self.assets = assets
        self.event_bus = event_bus
        self.nodes = named_nodes
        self.transform = assets.transform

        # Skull mode and dynamic pivot
        self.skull_mode: str = "original"
        self.jaw_pivot: tuple[float, float, float] | None = None

        # Loaded systems
        self.skull_meshes: Optional[dict[str, MeshInstance]] = None
        self.face_mesh: Optional[MeshInstance] = None
        self.facs_engine: Optional[FACSEngine] = None
        self.jaw_muscles: Optional[JawMuscleSystem] = None
        self.expression_muscles: Optional[ExpressionMuscleSystem] = None
        self.face_features: Optional[FaceFeatureSystem] = None
        self.neck_muscles: Optional[NeckMuscleSystem] = None
        self.vertebrae: Optional[VertebraeSystem] = None
        self.vertebrae_pivots: list[dict] = []
        self.head_rotation: Optional[HeadRotationSystem] = None
        self.neck_constraints: Optional[NeckConstraintSolver] = None
        self.skeleton: Optional[SkeletonBuilder] = None
        self.joint_setup: Optional[JointPivotSetup] = None
        self.bone_anchors: Optional[BoneAnchorRegistry] = None
        self.platysma: Optional[PlatysmaHandler] = None
        self.fascia: Optional[FasciaSystem] = None

    def _report(self, phase: str, progress: float) -> None:
        self.event_bus.publish(EventType.LOADING_PHASE, phase=phase)
        self.event_bus.publish(EventType.LOADING_PROGRESS, progress=progress)

    def load_head(self, skull_mode: str = "original") -> None:
        """Load head assets (phases 0-2). Call from main thread.

        Args:
            skull_mode: ``"original"`` for embedded skull, ``"bp3d"`` for BP3D bones.
        """
        self.skull_mode = skull_mode
        self.event_bus.publish(EventType.LOADING_STARTED)
        stl_dir = self.assets.stl_dir

        # Phase 0: Skull
        self._report("Building skull...", 0.0)
        skull_group_node, self.skull_meshes, self.jaw_pivot = build_skull(
            self.assets, mode=skull_mode,
        )
        set_jaw_pivot(*self.jaw_pivot)

        # Reparent skull nodes into the scene's skullGroup
        skull_target = self.nodes["skullGroup"]
        for child in list(skull_group_node.children):
            skull_group_node.remove(child)
            skull_target.add(child)

        # Phase 1: Face
        self._report("Building face...", 0.15)
        face_group_node, self.face_mesh = build_face(self.assets)
        # Reparent children (faceAlignment → face) into the scene's faceGroup.
        # The faceGroup itself has no transform — head rotation sets it per frame.
        # Alignment lives on the faceAlignment child node.
        face_target = self.nodes["faceGroup"]
        for child in list(face_group_node.children):
            face_group_node.remove(child)
            face_target.add(child)

        # FACS engine
        try:
            regions = load_config("face_regions.json")
            self.facs_engine = FACSEngine(self.face_mesh, regions)
        except Exception as e:
            logger.warning("FACS engine failed: %s", e)

        # Phase 2a: Jaw muscles
        self._report("Loading jaw muscles...", 0.25)
        try:
            defs = load_muscle_config("jaw_muscles.json")
            self.jaw_muscles = JawMuscleSystem(defs, self.transform, jaw_pivot=self.jaw_pivot)
            jaw_group = self.jaw_muscles.load(stl_dir)
            self.nodes["stlMuscleGroup"].add(jaw_group)
        except Exception as e:
            logger.warning("Jaw muscles failed: %s", e)
            self.jaw_muscles = None

        # Phase 2b: Expression muscles
        self._report("Loading expression muscles...", 0.40)
        try:
            defs = load_muscle_config("expression_muscles.json")
            self.expression_muscles = ExpressionMuscleSystem(defs, self.transform)
            expr_group = self.expression_muscles.load(stl_dir)
            self.nodes["exprMuscleGroup"].add(expr_group)

            # Reparent Platysma R/L from exprMuscleGroup to platysmaGroup
            # (platysmaGroup has identity transform — no group rotation)
            platysma_group = self.nodes.get("platysmaGroup")
            if platysma_group is not None:
                n = PlatysmaHandler.reparent_to_group(
                    self.expression_muscles.muscle_data, platysma_group,
                )
                logger.info("Reparented %d Platysma muscle(s) to platysmaGroup", n)
        except Exception as e:
            logger.warning("Expression muscles failed: %s", e)
            self.expression_muscles = None

        # Phase 2c: Face features
        self._report("Loading face features...", 0.55)
        try:
            defs = load_config("face_features.json")
            self.face_features = FaceFeatureSystem(defs, self.transform)
            feat_group = self.face_features.load(stl_dir)
            self.nodes["faceFeatureGroup"].add(feat_group)
        except Exception as e:
            logger.warning("Face features failed: %s", e)
            self.face_features = None

        # Phase 2d: Neck muscles
        self._report("Loading neck muscles...", 0.70)
        try:
            defs = load_muscle_config("neck_muscles.json")
            self.neck_muscles = NeckMuscleSystem(defs, self.transform, jaw_pivot=self.jaw_pivot)
            neck_group = self.neck_muscles.load(stl_dir)
            self.nodes["neckMuscleGroup"].add(neck_group)
        except Exception as e:
            logger.warning("Neck muscles failed: %s", e)
            self.neck_muscles = None

        # Phase 2e: Vertebrae
        self._report("Loading vertebrae...", 0.85)
        try:
            vert_defs = load_skeleton_config("cervical_vertebrae.json")
            vert_fracs = load_skeleton_config("vertebra_fractions.json")
            self.vertebrae = VertebraeSystem(vert_defs, vert_fracs, self.transform)
            vert_group, self.vertebrae_pivots = self.vertebrae.load(stl_dir)
            self.nodes["vertebraeGroup"].add(vert_group)
        except Exception as e:
            logger.warning("Vertebrae failed: %s", e)
            self.vertebrae = None

        # Head rotation system
        self.head_rotation = HeadRotationSystem(jaw_pivot=self.jaw_pivot)

        # Platysma body-spanning handler (registers from expression muscles)
        if self.expression_muscles is not None:
            self.platysma = PlatysmaHandler(head_pivot=self.jaw_pivot)
            self.platysma.register(self.expression_muscles.muscle_data)
            if self.platysma.registered:
                logger.info("Platysma handler registered (%d muscles)",
                            len(self.platysma._platysma))

        # Neck constraint solver
        try:
            limits_data = load_config("joint_limits.json")
            self.neck_constraints = NeckConstraintSolver(limits_data)
        except Exception as e:
            logger.warning("Neck constraints failed: %s", e)

        self._report("Head complete", 1.0)

    def load_body_skeleton(self) -> None:
        """Load body skeleton (phase 3)."""
        self._report("Loading body skeleton...", 0.0)

        self.skeleton = SkeletonBuilder(self.assets)
        body_root = self.nodes["bodyRoot"]

        try:
            self.skeleton.load_all(body_root)
        except Exception as e:
            logger.warning("Body skeleton failed: %s", e)

        # Build bone_nodes dict mapping bone name → SceneNode for joint computation
        bone_nodes = self._collect_bone_nodes()

        # Setup joint pivots from bone geometry (dynamic position computation)
        self.joint_setup = JointPivotSetup()
        self.joint_setup.setup_from_skeleton(
            bone_nodes,
            body_root,
            self.skeleton.groups.get("upper_limb"),
            self.skeleton.groups.get("lower_limb"),
            self.skeleton.groups.get("hand"),
            self.skeleton.groups.get("foot"),
        )

        logger.info("Body skeleton loaded: %d groups, %d pivots",
                     len(self.skeleton.groups), len(self.joint_setup.pivots))

        # Build BoneAnchorRegistry from skeleton nodes + thoracic pivots
        self.bone_anchors = BoneAnchorRegistry()
        self.bone_anchors.register_bones(bone_nodes)

        # Also register thoracic pivot nodes (used by deep prevertebral muscles)
        thoracic_group = self.skeleton.groups.get("thoracic")
        if thoracic_group is not None:
            for child in thoracic_group.children:
                if child.name and child.name not in bone_nodes:
                    self.bone_anchors.register_bones({child.name: child})

        # Register rib nodes for scalene attachment
        rib_group = self.skeleton.groups.get("ribs")
        if rib_group is not None:
            for child in rib_group.children:
                if child.name and child.name not in bone_nodes:
                    self.bone_anchors.register_bones({child.name: child})

        # Force world matrix update so bone positions are correct for snapshot.
        # The pipeline doesn't hold a Scene reference, but body_root is the
        # common ancestor of all skeleton nodes.
        body_root.update_world_matrix(force=True)

        # Snapshot rest positions (now that world matrices are current)
        self.bone_anchors.snapshot_rest_positions()
        logger.info("Bone anchor registry: %d bones registered",
                     len(self.bone_anchors.bone_names))

        # Build fascia constraint system from skeleton bones
        self.fascia = FasciaSystem(build_anatomical_fascia_regions(), self.bone_anchors)
        self.fascia.snapshot_rest()
        if self.platysma is not None:
            self.platysma.set_fascia_system(self.fascia)
            logger.info("Fascia system wired to Platysma handler")

        # Create fascia visualization markers
        from faceforge.anatomy.fascia import create_fascia_markers
        fascia_group = self.nodes.get("fasciaGroup")
        if fascia_group is not None:
            markers = create_fascia_markers(self.fascia)
            fascia_group.add(markers)
            logger.info("Fascia markers created (%d regions)", len(self.fascia.region_names))

        # Wire bone registry to neck muscles for per-muscle pinning
        if self.neck_muscles is not None:
            self.neck_muscles.set_bone_registry(self.bone_anchors)

        self._report("Skeleton complete", 1.0)
        self.event_bus.publish(EventType.LOADING_COMPLETE)

    def _collect_bone_nodes(self) -> dict[str, SceneNode]:
        """Build a dict mapping bone names to SceneNodes from all skeleton groups.

        Scans upper_limb, lower_limb, hand, foot, and pelvis groups for
        child nodes whose names match expected bone names.
        """
        bone_nodes: dict[str, SceneNode] = {}
        if self.skeleton is None:
            return bone_nodes

        # Collect from all relevant skeleton groups
        for group_key in ("upper_limb", "lower_limb", "hand", "foot", "pelvis"):
            group = self.skeleton.groups.get(group_key)
            if group is None:
                continue
            for child in group.children:
                if child.name:
                    bone_nodes[child.name] = child

        return bone_nodes
