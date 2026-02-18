# body/ -- Full-Body Systems

This package handles everything below the neck: skeleton loading, joint articulation, body animation, soft tissue skinning, and on-demand loading of muscles, organs, vasculature, and brain structures.

## Modules

### `soft_tissue.py` -- Delta-Matrix Soft Tissue Skinning

The core skinning system that deforms body meshes (muscles, organs, skin) as joints move.

**Key classes:**
- `SoftTissueSkinning` -- Main skinning engine
- `SkinJoint` -- Skeletal joint with rest-pose matrix and bone segment
- `SkinBinding` -- Per-mesh vertex-to-joint assignments and blend weights

**Algorithm:**
1. `build_skin_joints()` -- Collects spine and limb joints, snapshots rest-pose world matrices, builds bone segments for nearest-bone assignment
2. `register_skin_mesh()` -- Assigns each vertex to its nearest bone segment, computes blend weights (muscles: full-range along segment; non-muscles: 15% endpoint blend zones)
3. `update()` -- Per-frame delta transform: `restWorldInv * currentWorld` applied per vertex with dual quaternion blending

**Features:**
- Cross-chain blending with `CROSS_CHAIN_RADIUS` for smooth transitions between kinematic chains
- Divergence clamping to prevent distortion at extreme poses
- Neighbor-stretch clamping to avoid spiky artifacts
- Boundary smoothing for vertices at chain boundaries
- Muscle activation coloring (contraction = red tint, stretch = darker)
- Early-exit optimization when joint state is unchanged

### `body_animation.py` -- Body Animation System

Per-frame body animation: spine flex/bend/rotation, limb articulation, rib breathing.

**Key class:** `BodyAnimationSystem`
- Distributes spine rotation across thoracic and lumbar vertebrae using fraction tables
- Applies limb joint rotations (shoulder, elbow, wrist, hip, knee, ankle)
- Cycles rib rotation for breathing animation via `breathPhaseBody` and `breathDepth`
- Finger articulation support

### `joint_pivots.py` -- Joint Pivot Setup

Dynamically computes joint positions from loaded bone geometry.

**Key class:** `JointPivotSetup`
- `find_joint_center()` -- Finds midpoint between closest vertices of adjacent bones
- `compute_bone_endpoint()` -- Fallback: centroid of top/bottom 5% of vertices
- Creates chained pivot hierarchies: shoulder -> elbow -> wrist, hip -> knee -> ankle

### `skeleton.py` -- Skeleton Builder

Loads all body skeleton groups from STL definition configs.

**Key class:** `SkeletonBuilder`
- Loads 8 skeleton regions: thoracic spine, lumbar spine, rib cage, pelvis, upper limbs, hands, lower limbs, feet
- Creates pivot groups at each vertebra level for articulation
- Loads spine distribution fractions from JSON config

### `body_constraints.py` -- Body Joint Limits

Clamps body DOF values to physiological limits loaded from `body_joint_limits.json`.

**Key class:** `BodyConstraints`
- Pure clamp-based (unlike the iterative neck constraint solver)
- Supports bilateral template expansion (`{s}` -> `r`/`l`)

### `body_muscles.py` -- Body Muscle Manager

On-demand loading of body muscle groups (back, shoulder, arm, torso, hip, leg).

**Key class:** `BodyMuscleManager`
- Loads from 6 muscle config files
- Registers each mesh with the soft tissue skinning system

### `organs.py`, `vasculature.py`, `brain.py` -- On-Demand Tissue Managers

Each manages lazy loading of its respective tissue category.

**Key classes:** `OrganManager`, `VasculatureManager`, `BrainManager`
- Organs and vasculature register with the skinning system for deformation
- Brain meshes are parented to `skullGroup` (not `bodyRoot`) and follow the skull via the scene graph hierarchy, requiring no skinning

### `diagnostics.py` -- Skinning Diagnostics

Analysis tools for debugging and validating skinning quality.

**Key class:** `SkinningDiagnostic`
- `analyze_bindings()` -- Per-mesh chain assignment statistics at registration time
- `check_displacements()` -- Detects vertices displaced beyond threshold after update
- `check_mesh_distortion()` -- Topological checks: edge stretch, triangle inversion, area collapse
- `format_report()` -- Human-readable diagnostic output

## Internal Dependencies

- `core.scene_graph` -- SceneNode for hierarchy
- `core.mesh` -- MeshInstance, BufferGeometry
- `core.state` -- BodyState for joint DOF values
- `core.math_utils` -- Mat4, Quat, vector operations, dual quaternion functions
- `loaders.asset_manager` -- AssetManager for STL loading
- `loaders.stl_batch_loader` -- Batch STL loading with coordinate transforms
