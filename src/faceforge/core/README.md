# core/ -- Core Infrastructure

This package provides the foundational data structures, math utilities, and communication primitives used by every other package in FaceForge. It has no dependencies on any other FaceForge subpackage (except `constants.py` at the package root).

## Modules

### `scene_graph.py` -- Hierarchical Scene Graph

Mirrors Three.js Object3D and Scene classes.

**Key classes:**
- `SceneNode` -- A node with position, quaternion, scale, optional mesh, children, and visibility. Computes local and world matrices via `update_matrix()` / `update_world_matrix()`.
- `Scene` -- Root node that traverses the full hierarchy to update world matrices and collect visible meshes for rendering.

SceneNode methods: `add()`, `remove()`, `set_position()`, `set_quaternion()`, `set_scale()`, `mark_dirty()`, `get_world_position()`.

### `state.py` -- Application State Management

Centralized mutable state for face and body animation parameters.

**Key classes:**
- `FaceState` -- 12 Action Units (AU1-AU26), eye look, blink, head yaw/pitch/roll, ear wiggle, pupil dilation, eye color, auto-animation toggles, internal timers
- `BodyState` -- 26 joint DOFs (spine flex/bend/twist, shoulder/elbow/wrist/hip/knee/ankle for both sides, breathing depth)
- `TargetAU` -- Target values for AU interpolation
- `TargetHead` -- Target values for head rotation interpolation
- `ConstraintState` -- Neck constraint solver outputs (total excess, shoulder compensation)
- `StateManager` -- Aggregates face, body, targets, and constraint state

**Constant:** `AU_IDS` -- List of all 12 AU identifiers.

### `math_utils.py` -- NumPy Math Utilities

Lightweight 3D math operations backed by NumPy arrays.

**Type aliases:** `Vec3`, `Vec4`, `Mat3`, `Mat4`, `Quat` (all NDArray[float64])

**Key functions:**
- Vector: `vec3()`, `vec4()`, `normalize()`, `cross()`, `dot()`, `clamp()`
- Matrix: `mat4_identity()`, `mat4_translation()`, `mat4_rotation_x/y/z()`, `mat4_scale()`, `mat4_compose()`, `mat4_inverse()`, `mat4_look_at()`, `mat4_perspective()`
- Quaternion: `quat_identity()`, `quat_from_euler()`, `quat_from_axis_angle()`, `quat_multiply()`, `quat_slerp()`, `quat_rotate_vec3()`, `quat_normalize()`
- Normal matrix: `mat3_normal()` (upper-left 3x3 of inverse-transpose)
- Batch operations: `batch_mat4_to_dual_quat()`, `batch_quat_multiply()`, `batch_quat_rotate()` for vectorized skinning
- Coordinate: `transform_point()`, `deg_to_rad()`

Quaternion convention: `[x, y, z, w]`.

### `mesh.py` -- Mesh Data Structures

GL-independent geometry storage.

**Key classes:**
- `BufferGeometry` -- Stores vertex positions (float32), normals (float32), optional indices (uint32), and vertex count. Methods: `compute_normals()`, `triangle_count`, `has_indices`.
- `MeshInstance` -- Named mesh with geometry, material, and rest-pose snapshot. Methods: `store_rest_pose()` (snapshots positions/normals for animation reset).

### `material.py` -- Material Definitions

Rendering material properties.

**Key classes:**
- `RenderMode` -- Enum: SOLID, WIREFRAME, XRAY, POINTS, OPAQUE
- `Material` -- Dataclass with color, opacity, shininess, emissive, render mode, double-sided, transparent, depth-write, wireframe color. Factory: `Material.from_hex()`.

### `events.py` -- Event Bus

Decoupled publish/subscribe communication system.

**Key classes:**
- `EventType` -- Enum of all application events (AU changes, loading progress, layer toggles, camera, alignment, render mode, etc.)
- `EventBus` -- `subscribe(event_type, handler)`, `publish(event_type, **data)`, `unsubscribe()`, `clear()`

### `clock.py` -- Frame Timing

Delta clock for consistent animation timing.

**Key class:** `DeltaClock`
- `get_delta()` -- Returns seconds elapsed since last call, clamped to `MAX_DELTA_TIME` (0.1s)
- Uses `time.perf_counter()` for high-resolution timing

### `config_loader.py` -- JSON Config Loading

Utilities for loading JSON config files from well-known asset directories.

**Key functions:**
- `load_json(path)` -- Generic JSON file loader
- `load_config(name)` -- From `assets/config/`
- `load_meshdata(name)` -- From `assets/meshdata/`
- `load_muscle_config(name)` -- From `assets/config/muscles/`
- `load_skeleton_config(name)` -- From `assets/config/skeleton/`

## Design Principles

- Zero GL/Qt dependencies -- core types are usable in headless testing and tools
- NumPy-backed math for performance and compatibility with the rendering pipeline
- Dataclass-based state for easy serialization and inspection
- Event bus for loose coupling between UI and simulation
