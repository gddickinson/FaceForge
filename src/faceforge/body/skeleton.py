"""Build full-body skeleton from STL batches."""

import logging
from typing import Optional

import numpy as np

from faceforge.core.scene_graph import SceneNode
from faceforge.core.config_loader import load_skeleton_config
from faceforge.loaders.stl_batch_loader import load_stl_batch, CoordinateTransform, STLBatchResult
from faceforge.loaders.asset_manager import AssetManager

logger = logging.getLogger(__name__)


def nest_spine_pivots(pivots: list[dict]) -> int:
    """Chain vertebral pivots parent-to-child, preserving the rest pose.

    The loader parents every pivot to the batch group, so they are siblings.
    ``BodyAnimationSystem`` then rotates each one by ``fraction * total``, and
    the fraction tables sum to exactly 1.0 -- which only produces the intended
    total bend if the rotations ACCUMULATE down a chain.  As siblings they do
    not: each vertebra tilts a couple of degrees about its own centroid and the
    spine as a whole never curves.  Measured before this function existed, the
    top of the thoracic spine moved 0.000 units at full flexion.

    The chain must run caudal -> cranial so that flexion carries the head and
    shoulders forward over a fixed pelvis.  The caudal end is found from the
    geometry, NOT from list order: in this dataset ``pivots[0]`` is the most
    cranial vertebra, so nesting in list order builds the chain upside down and
    inverts it: the chain's caudal end swings and its cranial end stays put
    (measured on the thoracic chain: caudal end 7.0 units, cranial 0.0).

    List order is left untouched -- ``BodyAnimationSystem`` pairs
    ``thoracic_fracs[i]`` with ``pivots[i]`` positionally, so reordering the
    list would silently give every vertebra a different fraction.  Only the
    parenting changes.

    The reparent must not move anything in the rest pose.  Pivots carry a
    translation only -- no rotation, unit scale -- so re-expressing a child's
    position relative to its new parent is a subtraction rather than a matrix
    solve.  The drift is asserted below regardless.

    Returns the number of pivots reparented.
    """
    if len(pivots) < 2:
        return 0

    nodes = [p["group"] for p in pivots]
    for node in nodes:
        node.update_world_matrix(force=True)

    # +Z is superior in this coordinate frame, so ascending Z is caudal ->
    # cranial.  Sorting a COPY keeps the caller's list order intact.
    ordered = sorted(nodes, key=lambda n: float(n.world_matrix[2, 3]))
    world = [n.world_matrix[:3, 3].copy() for n in ordered]

    moved = 0
    for i in range(1, len(ordered)):
        child, parent = ordered[i], ordered[i - 1]
        parent.add(child)                     # add() detaches from the old parent
        offset = world[i] - world[i - 1]
        child.set_position(float(offset[0]), float(offset[1]), float(offset[2]))
        moved += 1

    for node in ordered:
        node.update_world_matrix(force=True)
    drift = max(float(np.linalg.norm(n.world_matrix[:3, 3] - w))
                for n, w in zip(ordered, world))
    if drift > 1e-6:
        logger.error("Spine nesting moved the rest pose by %.6g units; the "
                     "hierarchy change was not transform-preserving", drift)
    return moved


class SkeletonBuilder:
    """Loads all body skeleton groups (thoracic, lumbar, ribs, pelvis, limbs, hands, feet)."""

    def __init__(self, asset_manager: AssetManager):
        self.assets = asset_manager
        self.groups: dict[str, SceneNode] = {}
        self.pivots: dict[str, list[dict]] = {}
        self.loaded: dict[str, bool] = {}
        self.rib_nodes: list[SceneNode] = []  # individual rib SceneNodes for breathing
        #: group key -> "ExcType: message" for every batch that failed to load.
        self.failed_batches: dict[str, str] = {}

    def load_thoracic_spine(self) -> STLBatchResult:
        defs = load_skeleton_config("thoracic_spine.json")
        result = load_stl_batch(
            defs, label="thoracic_spine",
            transform=self.assets.transform,
            create_pivots=True, pivot_key="level",
            stl_dir=self.assets.stl_dir,
        )
        self.groups["thoracic"] = result.group
        self.loaded["thoracic"] = True

        # Build pivot info with fractions
        fracs = load_skeleton_config("thoracic_fractions.json")
        thoracic_pivots = []
        for lv, pivot_node in result.pivot_groups.items():
            thoracic_pivots.append({
                "group": pivot_node,
                "level": lv,
                "fraction": fracs[lv] if lv < len(fracs) else 0.0,
            })
        self.pivots["thoracic"] = thoracic_pivots
        logger.info("Thoracic spine: nested %d of %d pivots into a chain",
                    nest_spine_pivots(thoracic_pivots), len(thoracic_pivots))
        return result

    def load_lumbar_spine(self) -> STLBatchResult:
        defs = load_skeleton_config("lumbar_spine.json")
        result = load_stl_batch(
            defs, label="lumbar_spine",
            transform=self.assets.transform,
            create_pivots=True, pivot_key="level",
            stl_dir=self.assets.stl_dir,
        )
        self.groups["lumbar"] = result.group
        self.loaded["lumbar"] = True

        fracs = load_skeleton_config("lumbar_fractions.json")
        lumbar_pivots = []
        for lv, pivot_node in result.pivot_groups.items():
            lumbar_pivots.append({
                "group": pivot_node,
                "level": lv,
                "fraction": fracs[lv] if lv < len(fracs) else 0.0,
            })
        self.pivots["lumbar"] = lumbar_pivots
        logger.info("Lumbar spine: nested %d of %d pivots into a chain",
                    nest_spine_pivots(lumbar_pivots), len(lumbar_pivots))
        return result

    def load_rib_cage(self) -> STLBatchResult:
        result = self.assets.load_skeleton_batch("rib_cage.json", label="rib_cage")
        self.groups["ribs"] = result.group
        self.loaded["ribs"] = True
        # Save individual rib nodes for breathing animation
        self.rib_nodes = list(result.nodes)
        return result

    def load_pelvis(self) -> STLBatchResult:
        result = self.assets.load_skeleton_batch("pelvis.json", label="pelvis")
        self.groups["pelvis"] = result.group
        self.loaded["pelvis"] = True
        return result

    def load_upper_limbs(self) -> STLBatchResult:
        result = self.assets.load_skeleton_batch("upper_limb.json", label="upper_limb")
        self.groups["upper_limb"] = result.group
        self.loaded["upper_limb"] = True
        return result

    def load_hands(self) -> STLBatchResult:
        result = self.assets.load_skeleton_batch("hand.json", label="hand")
        self.groups["hand"] = result.group
        self.loaded["hand"] = True
        return result

    def load_lower_limbs(self) -> STLBatchResult:
        result = self.assets.load_skeleton_batch("lower_limb.json", label="lower_limb")
        self.groups["lower_limb"] = result.group
        self.loaded["lower_limb"] = True
        return result

    def load_feet(self) -> STLBatchResult:
        result = self.assets.load_skeleton_batch("foot.json", label="foot")
        self.groups["foot"] = result.group
        self.loaded["foot"] = True
        return result

    @property
    def load_failed(self) -> bool:
        """True if any skeleton batch failed on the last :meth:`load_all`."""
        return bool(self.failed_batches)

    def load_all(self, body_root: SceneNode) -> None:
        """Load all skeleton groups and parent them to body_root.

        A batch that fails is recorded in :attr:`failed_batches` and logged with
        a traceback; ``self.loaded[key]`` is forced False so the group is never
        reported present when it is not. Partial success is intentional -- the
        rest of the skeleton is still usable.
        """
        # Keys match the self.groups / self.loaded keys each loader writes.
        loaders = [
            ("thoracic", self.load_thoracic_spine),
            ("lumbar", self.load_lumbar_spine),
            ("ribs", self.load_rib_cage),
            ("pelvis", self.load_pelvis),
            ("upper_limb", self.load_upper_limbs),
            ("hand", self.load_hands),
            ("lower_limb", self.load_lower_limbs),
            ("foot", self.load_feet),
        ]
        for key, loader in loaders:
            try:
                result = loader()
                body_root.add(result.group)
            except (OSError, ValueError, KeyError) as exc:
                # Missing/unreadable STL, malformed config, missing config key.
                # Anything else (AttributeError, TypeError) is a bug and must
                # propagate rather than become a silently missing bone group.
                logger.exception("Skeleton batch %r failed to load", key)
                self.failed_batches[key] = f"{type(exc).__name__}: {exc}"
                # The loader may have set loaded[...] before raising.
                self.loaded[key] = False
            else:
                self.failed_batches.pop(key, None)
