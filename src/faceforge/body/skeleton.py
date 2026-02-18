"""Build full-body skeleton from STL batches."""

from typing import Optional

from faceforge.core.scene_graph import SceneNode
from faceforge.core.config_loader import load_skeleton_config
from faceforge.loaders.stl_batch_loader import load_stl_batch, CoordinateTransform, STLBatchResult
from faceforge.loaders.asset_manager import AssetManager


class SkeletonBuilder:
    """Loads all body skeleton groups (thoracic, lumbar, ribs, pelvis, limbs, hands, feet)."""

    def __init__(self, asset_manager: AssetManager):
        self.assets = asset_manager
        self.groups: dict[str, SceneNode] = {}
        self.pivots: dict[str, list[dict]] = {}
        self.loaded: dict[str, bool] = {}
        self.rib_nodes: list[SceneNode] = []  # individual rib SceneNodes for breathing

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

    def load_all(self, body_root: SceneNode) -> None:
        """Load all skeleton groups and parent them to body_root."""
        loaders = [
            self.load_thoracic_spine,
            self.load_lumbar_spine,
            self.load_rib_cage,
            self.load_pelvis,
            self.load_upper_limbs,
            self.load_hands,
            self.load_lower_limbs,
            self.load_feet,
        ]
        for loader in loaders:
            try:
                result = loader()
                body_root.add(result.group)
            except Exception:
                pass  # Skip failed batches gracefully
