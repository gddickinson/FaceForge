"""On-demand body muscle loading and management."""

import logging

from faceforge.core.scene_graph import SceneNode
from faceforge.core.mesh import MeshInstance
from faceforge.loaders.asset_manager import AssetManager
from faceforge.loaders.stl_batch_loader import STLBatchResult
from faceforge.body.soft_tissue import SoftTissueSkinning

logger = logging.getLogger(__name__)


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
        #: region -> "ExcType: message" for every region that failed to load.
        #: ``loaded`` stays False while this is non-empty so the failed regions
        #: can be retried; regions that did load are not re-added.
        self.failed_regions: dict[str, str] = {}

    @property
    def load_failed(self) -> bool:
        """True if any muscle region failed to load on the last attempt."""
        return bool(self.failed_regions)

    def load_all(
        self,
        parent: SceneNode,
        skinning: SoftTissueSkinning | None = None,
        allowed_chains: set[int] | None = None,
    ) -> None:
        """Load all body muscle groups and parent them.

        Partial success is normal here (six independent configs), so a failed
        region is recorded in :attr:`failed_regions` and logged rather than
        silently skipped. ``loaded`` is only set once every region is present.
        """
        if self.loaded:
            return

        for config_name in MUSCLE_CONFIGS:
            region = config_name.replace("_muscles.json", "").replace(".json", "")
            if region in self.groups:
                continue  # already loaded on an earlier attempt
            try:
                result = self.assets.load_body_muscles(config_name)
                self.groups[region] = result.group
                self.meshes[region] = result.meshes
                parent.add(result.group)

                # Register with soft tissue skinning
                if skinning is not None:
                    for mesh in result.meshes:
                        skinning.register_skin_mesh(mesh, is_muscle=True, allowed_chains=allowed_chains)
            except (OSError, ValueError, KeyError) as exc:
                # Missing/unreadable STL, malformed config, missing config key.
                # Anything else (AttributeError, TypeError) is a bug and must
                # propagate rather than become a silently missing region.
                logger.exception("Body muscle region %r failed to load", region)
                self.failed_regions[region] = f"{type(exc).__name__}: {exc}"
            else:
                self.failed_regions.pop(region, None)

        self.loaded = not self.failed_regions

    def set_visibility(self, visible: bool) -> None:
        for group in self.groups.values():
            group.visible = visible
