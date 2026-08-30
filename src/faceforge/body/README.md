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

**Key function:** `nest_spine_pivots()`

The STL batch loader parents every vertebral pivot to the batch group, so they
arrive as *siblings*. `BodyAnimationSystem` rotates each by
`fraction * total_angle`, and the fraction tables sum to exactly 1.0 — which
only produces the intended bend if those rotations **accumulate down a chain**.
As siblings they did not: each vertebra tilted a couple of degrees about its own
centroid and the spine never curved. Measured on the real skeleton, the cranial
end of the thoracic spine moved **0.000 units at full flexion**, and the skin
displaced 2.48 units in total.

`nest_spine_pivots()` chains each region parent-to-child after load. Two
properties it must hold, both covered by `tests/body/test_spine_nesting.py`:

- **Rooted at the caudal end**, so flexion carries the shoulders forward over a
  fixed pelvis. The caudal end is found from the geometry, not from list order —
  `pivots[0]` is the most *cranial* vertebra in this dataset, so nesting in list
  order builds the chain upside down and swings the sacrum instead.
- **List order untouched.** `BodyAnimationSystem` pairs `thoracic_fracs[i]` with
  `pivots[i]` positionally, so reordering the list would silently hand every
  vertebra a different fraction. Only the parenting changes.

Note that `joint_pivots.py` already chains the limbs (shoulder → elbow → wrist);
the spine was the one region built through the batch loader instead, which is
how it ended up flat.

**This nesting depends on a fix in `soft_tissue.update()`.** `SceneNode.
update_world_matrix` propagates *downward* only — it multiplies by the parent's
current world matrix without first ensuring the parent is up to date. Refreshing
joints one at a time in list order is therefore correct only while every joint's
ancestors are static, which was true by accident while the pivots were siblings
of a group that never moves. Once chained, a joint whose ancestor appears later
in `self.joints` reads a stale parent transform: measured, `spine_flex=0.3`
displaced the skin *further* than `spine_flex=1.0`, and re-applying the rest pose
left it 10.3 units adrift. `update()` now refreshes from the joint hierarchy's
root before reading any joint.

Current range: spinal articulation is correct and reversible, but the thoracic
and lumbar chains are still independent, so the shoulders receive the thoracic
share (40%) and not the lumbar (60%). Linking the thoracic root to the topmost
lumbar vertebra would complete the curve.

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
