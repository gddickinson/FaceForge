"""On-demand brain loading.

Brain meshes are parented to skullGroup (not bodyRoot) so they follow
skull movement via the scene graph, without needing soft tissue skinning.
"""

from faceforge.core.scene_graph import SceneNode
from faceforge.core.mesh import MeshInstance
from faceforge.loaders.asset_manager import AssetManager


class BrainManager:
    """Manages on-demand loading of brain structures."""

    def __init__(self, asset_manager: AssetManager):
        self.assets = asset_manager
        self.group: SceneNode | None = None
        self.meshes: list[MeshInstance] = []
        self.loaded: bool = False

    def load(self, skull_group: SceneNode) -> None:
        """Load brain structures and parent to skull group (not bodyRoot).

        Brain follows skull via scene graph hierarchy, so no skinning needed.
        """
        if self.loaded:
            return
        try:
            result = self.assets.load_brain()
            self.group = result.group
            self.meshes = result.meshes
            skull_group.add(result.group)
        except Exception:
            pass
        self.loaded = True

    def set_visibility(self, visible: bool) -> None:
        if self.group is not None:
            self.group.visible = visible
