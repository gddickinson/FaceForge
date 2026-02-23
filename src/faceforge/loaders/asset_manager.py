"""Central asset cache and lazy loading manager."""

from pathlib import Path
from typing import Optional

from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import SceneNode
from faceforge.loaders.stl_parser import load_stl_file
from faceforge.loaders.stl_batch_loader import CoordinateTransform, load_stl_batch, STLBatchResult
from faceforge.loaders.mesh_data_loader import load_skull_meshes, load_face_mesh
from faceforge.core.config_loader import (
    load_config, load_muscle_config, load_skeleton_config,
)
from faceforge.constants import STL_DIR, BODY_MESHES_DIR


class AssetManager:
    """Central cache for loaded assets.

    Provides lazy loading of STL meshes and config data.
    All loaded geometry is cached to avoid duplicate file I/O.
    """

    def __init__(self, stl_dir: Optional[Path] = None):
        self.stl_dir = stl_dir or STL_DIR
        self.transform = CoordinateTransform()

        # Caches
        self._stl_cache: dict[str, BufferGeometry] = {}
        self._config_cache: dict[str, object] = {}

        # Loaded results
        self.skull_meshes: Optional[dict[str, MeshInstance]] = None
        self.face_mesh: Optional[MeshInstance] = None

    def init_transform(self) -> None:
        """Load coordinate transform from config."""
        try:
            self.transform = CoordinateTransform.from_config()
        except FileNotFoundError:
            pass  # Use defaults

    def get_stl(self, stl_name: str, indexed: bool = True) -> BufferGeometry:
        """Load and cache an STL file by name (without extension)."""
        cache_key = f"{stl_name}:{'idx' if indexed else 'raw'}"
        if cache_key not in self._stl_cache:
            path = self.stl_dir / f"{stl_name}.stl"
            self._stl_cache[cache_key] = load_stl_file(path, indexed=indexed)
        return self._stl_cache[cache_key]

    def load_skull(self) -> dict[str, MeshInstance]:
        """Load skull meshes (cached)."""
        if self.skull_meshes is None:
            self.skull_meshes = load_skull_meshes()
        return self.skull_meshes

    def load_face(self) -> MeshInstance:
        """Load face mesh (cached)."""
        if self.face_mesh is None:
            self.face_mesh = load_face_mesh()
        return self.face_mesh

    def load_batch(self, config_name: str, config_loader, **kwargs) -> STLBatchResult:
        """Load a batch of STLs from a config file."""
        defs = config_loader(config_name)
        return load_stl_batch(
            defs,
            transform=self.transform,
            stl_dir=self.stl_dir,
            **kwargs,
        )

    def load_jaw_muscles(self) -> STLBatchResult:
        return self.load_batch(
            "jaw_muscles.json", load_muscle_config,
            label="jaw_muscles", indexed=True,
        )

    def load_expression_muscles(self) -> STLBatchResult:
        return self.load_batch(
            "expression_muscles.json", load_muscle_config,
            label="expression_muscles", indexed=True,
        )

    def load_neck_muscles(self) -> STLBatchResult:
        return self.load_batch(
            "neck_muscles.json", load_muscle_config,
            label="neck_muscles", indexed=True,
        )

    def load_vertebrae(self) -> STLBatchResult:
        defs = load_skeleton_config("cervical_vertebrae.json")
        return load_stl_batch(
            defs,
            label="vertebrae",
            transform=self.transform,
            create_pivots=True,
            pivot_key="level",
            stl_dir=self.stl_dir,
        )

    def load_skeleton_batch(self, config_name: str, **kwargs) -> STLBatchResult:
        defs = load_skeleton_config(config_name)
        return load_stl_batch(
            defs,
            transform=self.transform,
            stl_dir=self.stl_dir,
            **kwargs,
        )

    def load_body_muscles(self, config_name: str) -> STLBatchResult:
        return self.load_batch(
            config_name, load_muscle_config,
            label=config_name.replace(".json", ""),
        )

    def load_organs(self) -> STLBatchResult:
        defs = load_config("organs.json")
        return load_stl_batch(
            defs, label="organs",
            transform=self.transform,
            stl_dir=self.stl_dir,
        )

    def load_vasculature(self) -> STLBatchResult:
        defs = load_config("vascular.json")
        return load_stl_batch(
            defs, label="vasculature",
            transform=self.transform,
            stl_dir=self.stl_dir,
        )

    def load_brain(self) -> STLBatchResult:
        defs = load_config("brain.json")
        return load_stl_batch(
            defs, label="brain",
            transform=self.transform,
            stl_dir=self.stl_dir,
        )

    def load_skull_bones(self) -> STLBatchResult:
        defs = load_config("skull_bones.json")
        return load_stl_batch(
            defs, label="skull_bones",
            transform=self.transform,
            stl_dir=self.stl_dir,
            indexed=True,
        )

    def load_teeth(self) -> STLBatchResult:
        defs = load_config("teeth.json")
        return load_stl_batch(
            defs, label="teeth",
            transform=self.transform,
            stl_dir=self.stl_dir,
            indexed=True,
        )

    def load_hand_muscles(self) -> STLBatchResult:
        return self.load_batch(
            "hand_muscles.json", load_muscle_config,
            label="hand_muscles",
        )

    def load_foot_muscles(self) -> STLBatchResult:
        return self.load_batch(
            "foot_muscles.json", load_muscle_config,
            label="foot_muscles",
        )

    def load_pelvic_floor(self) -> STLBatchResult:
        defs = load_config("pelvic_floor.json")
        return load_stl_batch(
            defs, label="pelvic_floor",
            transform=self.transform,
            stl_dir=self.stl_dir,
        )

    def load_ligaments(self) -> STLBatchResult:
        defs = load_config("ligaments.json")
        return load_stl_batch(
            defs, label="ligaments",
            transform=self.transform,
            stl_dir=self.stl_dir,
        )

    def load_oral(self) -> STLBatchResult:
        defs = load_config("oral.json")
        return load_stl_batch(
            defs, label="oral",
            transform=self.transform,
            stl_dir=self.stl_dir,
        )

    def load_cardiac_additional(self) -> STLBatchResult:
        defs = load_config("cardiac_additional.json")
        return load_stl_batch(
            defs, label="cardiac_additional",
            transform=self.transform,
            stl_dir=self.stl_dir,
        )

    def load_intestinal(self) -> STLBatchResult:
        defs = load_config("intestinal.json")
        return load_stl_batch(
            defs, label="intestinal",
            transform=self.transform,
            stl_dir=self.stl_dir,
        )

    def load_cns_additional(self) -> STLBatchResult:
        defs = load_config("cns_additional.json")
        return load_stl_batch(
            defs, label="cns_additional",
            transform=self.transform,
            stl_dir=self.stl_dir,
        )

    def load_skin(self) -> STLBatchResult:
        defs = load_config("skin.json")
        return load_stl_batch(
            defs, label="skin",
            transform=self.transform,
            stl_dir=self.stl_dir,
        )

    def load_body_mesh(self) -> tuple[BufferGeometry, BufferGeometry]:
        """Load male and female body surface meshes.

        Returns (male_geometry, female_geometry) with identical topology.
        """
        from faceforge.loaders.obj_parser import load_obj_file

        male_geom = load_obj_file(BODY_MESHES_DIR / "body_male.obj")
        female_geom = load_obj_file(BODY_MESHES_DIR / "body_female.obj")
        assert male_geom.vertex_count == female_geom.vertex_count, (
            f"Topology mismatch: male={male_geom.vertex_count} "
            f"vs female={female_geom.vertex_count}"
        )
        return male_geom, female_geom

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._stl_cache.clear()
        self._config_cache.clear()
