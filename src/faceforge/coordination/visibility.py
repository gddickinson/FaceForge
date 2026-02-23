"""Layer toggle → node visibility mapping."""

from faceforge.core.scene_graph import SceneNode


class VisibilityManager:
    """Maps layer toggle names to scene nodes for visibility control."""

    def __init__(self):
        self._toggles: dict[str, list[SceneNode]] = {}

    def register(self, toggle_name: str, node: SceneNode) -> None:
        """Register a scene node to be controlled by a toggle."""
        if toggle_name not in self._toggles:
            self._toggles[toggle_name] = []
        self._toggles[toggle_name].append(node)

    def set_visible(self, toggle_name: str, visible: bool) -> None:
        """Set visibility for all nodes registered to a toggle."""
        nodes = self._toggles.get(toggle_name, [])
        for node in nodes:
            node.visible = visible

    def is_visible(self, toggle_name: str) -> bool:
        """Check if the first node for a toggle is visible."""
        nodes = self._toggles.get(toggle_name, [])
        return nodes[0].visible if nodes else True

    def get_toggle_names(self) -> list[str]:
        return list(self._toggles.keys())

    def get_all_nodes(self) -> list[SceneNode]:
        """Return a flat list of all registered nodes."""
        result = []
        for nodes in self._toggles.values():
            result.extend(nodes)
        return result

    def dim_all_except(self, names: set[str], dim_opacity: float = 0.15) -> None:
        """Dim all nodes except those whose mesh name is in *names*.

        Sets opacity on matching nodes to 1.0, all others to *dim_opacity*.
        Used for search highlighting.
        """
        for toggle_name, nodes in self._toggles.items():
            for node in nodes:
                for mesh, _ in node.collect_meshes() if hasattr(node, 'collect_meshes') else []:
                    if mesh.name in names:
                        mesh.material.opacity = 1.0
                    else:
                        mesh.material.opacity = dim_opacity

    def restore_opacity(self) -> None:
        """Restore all nodes to full opacity after search highlighting."""
        for toggle_name, nodes in self._toggles.items():
            for node in nodes:
                for mesh, _ in node.collect_meshes() if hasattr(node, 'collect_meshes') else []:
                    mesh.material.opacity = 1.0
