"""On-demand vascular system loading."""

from faceforge.core.scene_graph import SceneNode
from faceforge.core.mesh import MeshInstance
from faceforge.loaders.asset_manager import AssetManager
from faceforge.body.soft_tissue import SoftTissueSkinning


class VasculatureManager:
    """Manages on-demand loading of vasculature."""

    def __init__(self, asset_manager: AssetManager):
        self.assets = asset_manager
        self.group: SceneNode | None = None
        self.meshes: list[MeshInstance] = []
        self.loaded: bool = False

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
        except Exception:
            pass
        self.loaded = True

    def set_visibility(self, visible: bool) -> None:
        if self.group is not None:
            self.group.visible = visible
