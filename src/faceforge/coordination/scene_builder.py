"""Constructs the scene graph from loaded assets."""

from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.core.state import StateManager
from faceforge.loaders.asset_manager import AssetManager
from faceforge.coordination.visibility import VisibilityManager


class SceneBuilder:
    """Builds the scene graph hierarchy.

    Scene hierarchy (mirrors Three.js):
      scene
        └── bodyRoot
              ├── skullGroup
              │     ├── cranium
              │     └── jawPivot
              │           ├── jaw
              │           └── lower_teeth
              ├── faceGroup
              ├── stlMuscleGroup (jaw muscles)
              ├── exprMuscleGroup (expression muscles, excl. Platysma)
              ├── platysmaGroup (Platysma R/L — identity transform)
              ├── faceFeatureGroup
              ├── neckMuscleGroup
              ├── vertebraeGroup
              ├── thoracicSpineGroup
              ├── lumbarSpineGroup
              ├── ribCageGroup
              ├── pelvisGroup
              ├── bodyMeshGroup (MakeHuman body surface mesh)
              ├── brainGroup (independent of skull visibility)
              ├── fasciaGroup (debug: fascia constraint markers)
              ├── [limb groups...]
              └── [on-demand: body muscles, organs, vasculature]
    """

    def __init__(self, asset_manager: AssetManager, visibility: VisibilityManager):
        self.assets = asset_manager
        self.visibility = visibility

    def build(self) -> tuple[Scene, dict[str, SceneNode]]:
        """Build the initial scene graph (skull + face only).

        Returns (scene, named_nodes) where named_nodes is a dict
        of key nodes for other systems to reference.
        """
        scene = Scene()

        # Body root — top-level group for entire body
        body_root = SceneNode(name="bodyRoot")
        scene.add(body_root)

        # Skull group
        skull_group = SceneNode(name="skullGroup")
        body_root.add(skull_group)

        # Face group
        face_group = SceneNode(name="faceGroup")
        body_root.add(face_group)

        # Placeholder groups for later loading
        stl_muscle_group = SceneNode(name="stlMuscleGroup")
        body_root.add(stl_muscle_group)

        expr_muscle_group = SceneNode(name="exprMuscleGroup")
        body_root.add(expr_muscle_group)

        platysma_group = SceneNode(name="platysmaGroup")
        body_root.add(platysma_group)

        face_feature_group = SceneNode(name="faceFeatureGroup")
        body_root.add(face_feature_group)

        neck_muscle_group = SceneNode(name="neckMuscleGroup")
        body_root.add(neck_muscle_group)

        vertebrae_group = SceneNode(name="vertebraeGroup")
        body_root.add(vertebrae_group)

        body_mesh_group = SceneNode(name="bodyMeshGroup")
        body_root.add(body_mesh_group)
        body_mesh_group.visible = False

        brain_group = SceneNode(name="brainGroup")
        body_root.add(brain_group)
        brain_group.visible = False

        fascia_group = SceneNode(name="fasciaGroup")
        body_root.add(fascia_group)
        fascia_group.visible = False

        # Register visibility toggles
        # Head soft tissue hidden by default for performance (skeleton-only view)
        self.visibility.register("skull", skull_group)
        self.visibility.register("face", face_group)
        face_group.visible = False
        self.visibility.register("jaw_muscles", stl_muscle_group)
        stl_muscle_group.visible = False
        self.visibility.register("expression_muscles", expr_muscle_group)
        self.visibility.register("expression_muscles", platysma_group)
        expr_muscle_group.visible = False
        platysma_group.visible = False
        self.visibility.register("face_features", face_feature_group)
        face_feature_group.visible = False
        self.visibility.register("neck_muscles", neck_muscle_group)
        neck_muscle_group.visible = False
        self.visibility.register("vertebrae", vertebrae_group)
        self.visibility.register("body_mesh", body_mesh_group)
        self.visibility.register("brain", brain_group)
        self.visibility.register("fascia", fascia_group)

        named = {
            "bodyRoot": body_root,
            "skullGroup": skull_group,
            "faceGroup": face_group,
            "stlMuscleGroup": stl_muscle_group,
            "exprMuscleGroup": expr_muscle_group,
            "platysmaGroup": platysma_group,
            "faceFeatureGroup": face_feature_group,
            "neckMuscleGroup": neck_muscle_group,
            "vertebraeGroup": vertebrae_group,
            "bodyMeshGroup": body_mesh_group,
            "brainGroup": brain_group,
            "fasciaGroup": fascia_group,
        }

        return scene, named
