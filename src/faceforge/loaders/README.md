# loaders/ -- Asset Loading

This package handles all file I/O for mesh data: parsing binary STL files, batch-loading anatomical structures with coordinate transforms, caching loaded geometry, and loading embedded mesh data from JSON.

## Modules

### `stl_parser.py` -- Binary STL Parser

Parses binary STL files into `BufferGeometry` objects.

**Key functions:**
- `parse_binary_stl(data: bytes) -> BufferGeometry` -- Parses raw STL binary data (80-byte header + 4-byte triangle count + 50 bytes per triangle) into flat position and normal arrays
- `load_stl_file(path, indexed=True) -> BufferGeometry` -- Loads from disk with optional index-based deduplication

Binary STL format per triangle (50 bytes):
- 12 bytes: face normal (3x float32)
- 36 bytes: 3 vertices (3x 3x float32)
- 2 bytes: attribute byte count

When `indexed=True`, duplicate vertices are merged using spatial hashing, reducing memory (e.g., ~957k triangle vertices to ~160k unique vertices for jaw muscles).

### `stl_batch_loader.py` -- Batch STL Loader

Loads groups of STL files defined by JSON config arrays, applying the BodyParts3D coordinate transform.

**Key classes:**
- `CoordinateTransform` -- BP3D-to-skull/world coordinate mapping. Transform: `result = (bp3d_pos - center) * scale + skull_center`. Note: X is negated (BP3D left=+X, skull left=-X). Can be loaded from `coordinate_transform.json` via `from_config()`.
- `STLBatchResult` -- Container for loaded batch: group SceneNode, list of MeshInstances, list of SceneNodes, dict of pivot groups

**Key function:** `load_stl_batch(defs, label, transform, create_pivots, pivot_key, stl_dir)`
- Takes a list of definition dicts (each with `stl`, `color`, optional `level`, `category`, etc.)
- Loads each STL, applies coordinate transform, creates MeshInstance with material
- Optionally creates pivot groups per level for articulation (used by vertebrae and spine)
- Returns `STLBatchResult` with all loaded data

### `asset_manager.py` -- Central Asset Cache

Lazy-loading manager that caches all loaded geometry to avoid duplicate file I/O.

**Key class:** `AssetManager`

Provides high-level loading methods:
- `get_stl(stl_name, indexed)` -- Load and cache a single STL file
- `load_skull()` -- Load skull meshes from embedded JSON data
- `load_face()` -- Load face mesh from embedded JSON data
- `load_jaw_muscles(config)` -- Load jaw muscle STL batch
- `load_expression_muscles(config)` -- Load expression muscle STL batch
- `load_face_features(config)` -- Load face feature STL batch
- `load_neck_muscles(config)` -- Load neck muscle STL batch
- `load_vertebrae(config)` -- Load vertebrae STL batch
- `load_skeleton_batch(config, label)` -- Load skeleton region STL batch
- `load_body_muscles(config)` -- Load body muscle STL batch
- `load_organs()` / `load_vasculature()` / `load_brain()` -- Load on-demand tissue

All methods use the internal STL cache (`_stl_cache`) keyed by filename and indexing mode.

### `mesh_data_loader.py` -- Embedded Mesh Data Loader

Loads skull and face mesh data from extracted JSON files (not STL).

**Key functions:**
- `load_skull_meshes()` -- Loads skull vertex positions and face indices from `skull.json`. Returns dict of named MeshInstances: jaw, upper_teeth, lower_teeth, cranium. Builds compact per-group vertex arrays with remapped indices.
- `load_face_mesh()` -- Loads the 468-vertex MediaPipe face mesh from `face.json` with positions and triangle indices.

## Data Flow

```
JSON config files  -->  AssetManager  -->  Anatomy/Body builders
                         |
STL files on disk  -->  stl_parser  -->  stl_batch_loader  -->  MeshInstance + SceneNode
                                          |
                                    CoordinateTransform (BP3D -> skull coords)
```

## External Dependencies

- `struct` -- Binary STL parsing
- `numpy` -- Vertex array operations
- `pathlib` -- File path handling

## Internal Dependencies

- `core.mesh` -- BufferGeometry, MeshInstance
- `core.material` -- Material
- `core.scene_graph` -- SceneNode
- `core.config_loader` -- JSON loading utilities
- `constants` -- STL_DIR path
