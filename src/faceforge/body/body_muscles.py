"""On-demand body muscle loading and management."""

from faceforge.core.scene_graph import SceneNode
from faceforge.core.mesh import MeshInstance
from faceforge.loaders.asset_manager import AssetManager
from faceforge.loaders.stl_batch_loader import STLBatchResult
from faceforge.body.soft_tissue import SoftTissueSkinning


# Muscle config file names for each body region
MUSCLE_CONFIGS = [
    "back_muscles.json",
    "shoulder_muscles.json",
    "arm_muscles.json",
    "torso_muscles.json",
    "hip_muscles.json",
    "leg_muscles.json",
]


class BodyMuscleManager:
    """Manages on-demand loading of body muscle groups."""

    def __init__(self, asset_manager: AssetManager):
        self.assets = asset_manager
        self.groups: dict[str, SceneNode] = {}
        self.meshes: dict[str, list[MeshInstance]] = {}
        self.loaded: bool = False

    def load_all(
        self,
        parent: SceneNode,
        skinning: SoftTissueSkinning | None = None,
        allowed_chains: set[int] | None = None,
    ) -> None:
        """Load all body muscle groups and parent them."""
        if self.loaded:
            return

        for config_name in MUSCLE_CONFIGS:
            region = config_name.replace("_muscles.json", "").replace(".json", "")
            try:
                result = self.assets.load_body_muscles(config_name)
                self.groups[region] = result.group
                self.meshes[region] = result.meshes
                parent.add(result.group)

                # Register with soft tissue skinning
                if skinning is not None:
                    for mesh in result.meshes:
                        skinning.register_skin_mesh(mesh, is_muscle=True, allowed_chains=allowed_chains)
            except Exception:
                pass  # Skip failed loads

        self.loaded = True

    def set_visibility(self, visible: bool) -> None:
        for group in self.groups.values():
            group.visible = visible
