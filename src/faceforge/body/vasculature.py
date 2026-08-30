"""On-demand vascular system loading."""

import logging

from faceforge.core.scene_graph import SceneNode
from faceforge.core.mesh import MeshInstance
from faceforge.loaders.asset_manager import AssetManager
from faceforge.body.soft_tissue import SoftTissueSkinning

logger = logging.getLogger(__name__)


class VasculatureManager:
    """Manages on-demand loading of vasculature."""

    def __init__(self, asset_manager: AssetManager):
        self.assets = asset_manager
        self.group: SceneNode | None = None
        self.meshes: list[MeshInstance] = []
        self.loaded: bool = False
        #: Set when the last load attempt failed. ``loaded`` stays False so the
        #: layer can be retried once the asset problem is fixed.
        self.load_failed: bool = False
        self.load_error: str | None = None

    def load(
        self,
        parent: SceneNode,
        skinning: SoftTissueSkinning | None = None,
    ) -> None:
        if self.loaded:
            return
        try:
            result = self.assets.load_vasculature()
            self.group = result.group
            self.meshes = result.meshes
            parent.add(result.group)
            if skinning is not None:
                for mesh in result.meshes:
                    skinning.register_skin_mesh(mesh)
        except (OSError, ValueError, KeyError) as exc:
            # Missing/unreadable STL, malformed config, missing config key.
            # Anything else (AttributeError, TypeError) is a bug and must
            # propagate rather than become a silently empty layer.
            logger.exception("Vasculature load failed; vascular layer will be empty")
            self.load_failed = True
            self.load_error = f"{type(exc).__name__}: {exc}"
            return
        self.loaded = True
        self.load_failed = False
        self.load_error = None

    def set_visibility(self, visible: bool) -> None:
        if self.group is not None:
            self.group.visible = visible
