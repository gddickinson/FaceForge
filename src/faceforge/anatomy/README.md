# anatomy/ -- Head and Neck Anatomical Systems

This package contains all anatomical modeling for the head and neck region: skull construction, face mesh, FACS-based animation, muscle systems, facial features, vertebrae, and head rotation with constraint solving.

All modules in this package have zero OpenGL imports -- vertex math is done entirely with NumPy.

## Modules

### `skull.py` -- Skull Builder

Constructs the skull mesh hierarchy (cranium, jaw with TMJ pivot, upper/lower teeth).

**Key function:** `build_skull(asset_manager, jaw_pivot, mode)`
- `"original"` mode: embedded skull data from `skull.json` (cranium + jaw + teeth)
- `"bp3d"` mode: individual BodyParts3D skull bone STLs with dynamic TMJ pivot computation

Returns `(skullGroup SceneNode, meshes_dict, jaw_pivot_tuple)`.

### `face.py` -- Face Mesh Builder

Loads the 468-vertex MediaPipe face mesh and positions it relative to the skull.

**Key function:** `build_face(asset_manager)`

Scene-graph structure:
```
faceGroup  (head rotation applied here)
  +-- faceAlignment  (scale=1.14, pos=(-0.2,-10.6,9.5), rotX=88.5deg)
       +-- face  (mesh)
```

The alignment lives on an intermediate node so head rotation does not disturb it.

### `facs.py` -- FACS Engine

Applies Facial Action Coding System displacements to face mesh vertices.

**Key class:** `FACSEngine`
- 12 Action Units (AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU20, AU22, AU25, AU26)
- Each AU affects specific vertex regions loaded from `face_regions.json`
- Displacement vectors are accumulated on top of rest positions
- All region loops are vectorized with NumPy fancy indexing

### `jaw_muscles.py` -- Jaw Muscle System

22 STL jaw muscles (16 closers + 6 suprahyoid openers) with per-frame jaw-angle deformation.

**Key class:** `JawMuscleSystem`
- Per-vertex jaw weights determine skull-fixed vs jaw-attached behavior
- Closers (masseters, temporalis, pterygoids) stretch when jaw opens
- Openers (suprahyoids: stylohyoid, mylohyoid, geniohyoid) contract when jaw opens
- Indexed geometry reduces ~957k triangle vertices to ~160k unique vertices

### `expression_muscles.py` -- Expression Muscle System

38 STL expression muscles (34 original + 4 scalp) with AU-driven contraction.

**Key class:** `ExpressionMuscleSystem`
- Each muscle has an `auMap` mapping AU IDs to contraction weights
- Activation = max weighted AU value across the map
- Active muscles contract vertices toward the muscle centroid along fiber direction
- Color tint shifts toward red when activated

### `face_features.py` -- Face Feature System

8 STL meshes: eyeballs, ears, nasal cartilages, eyebrows, throat structures.

**Key class:** `FaceFeatureSystem`
- Bilateral meshes split by X coordinate (X>=0 left, X<0 right)
- Eyeballs use pivot groups for rotation (+/-15deg horizontal, +/-10deg vertical)
- Procedural iris, pupil, cornea, and limbal ring geometry
- Eyebrow deformation via AU1 (inner raise), AU2 (outer raise), AU4 (lower)
- Nasal cartilage deformation via AU9 (scrunch up + pull back)
- Ear wiggle displacement
- Throat structures: hyoid bone and thyroid cartilage

### `neck_muscles.py` -- Neck Muscle System

36 STL neck muscles with head-follow deformation.

**Key class:** `NeckMuscleSystem`
- Groups: SCM (2), Infrahyoid (8), Scalenes (6), Levator Scapulae (2), Deep Prevertebral (12), Suboccipital (8)
- Parented to `bodyRoot` (not `skullGroup`) for stable world reference frame
- Forward rotation math: slerp(identity, headQ, effectiveF) per vertex around head pivot
- `lowerAttach` field determines body-follow fraction (shoulder, ribcage, thoracic)

### `vertebrae.py` -- Cervical Vertebrae System

15 STLs: 8 vertebrae (C1-C7 + T1) + 7 intervertebral discs.

**Key class:** `VertebraeSystem`
- Each vertebra level gets a pivot group at its centroid for articulation
- `VERTEBRA_FRACTIONS`: C1 gets 100% of head rotation, T1 gets 0% (anchor)
- Driven by the `HeadRotationSystem`

### `head_rotation.py` -- Head Rotation System

Applies head yaw/pitch/roll rotation distributed across cervical vertebrae.

**Key class:** `HeadRotationSystem`
- Composes rotation quaternion from `headYaw`, `headPitch`, `headRoll` (each -1 to 1)
- Yaw +/-45deg, Pitch +/-30deg, Roll +/-30deg
- Distributes rotation to vertebrae pivot groups per fraction table
- Includes soft-clamping via neck constraint solver when tension exceeds threshold

### `neck_constraints.py` -- Neck Constraint Solver

Tension monitoring, soft-clamping, and dynamic spine compensation.

**Key class:** `NeckConstraintSolver`
- Computes upper/lower attachment points per muscle from top/bottom 15% by spine fraction
- Per-frame solver computes stretch ratios, tension values, spine compensation
- Dynamic thoracic compensation: THORACIC_HEAD_FRACTIONS boosted by up to 40% per level
- Dynamic shoulder compensation: 3% base + up to 12% from constraint violations
- Tension visualization: muscle color lerps toward red based on tension

## Internal Dependencies

- `core.mesh` -- MeshInstance, BufferGeometry
- `core.scene_graph` -- SceneNode hierarchy
- `core.state` -- FaceState, BodyState, ConstraintState
- `core.math_utils` -- Vec3, Quat, quaternion operations
- `core.material` -- Material for mesh coloring
- `loaders.stl_batch_loader` -- CoordinateTransform, load_stl_batch
- `constants` -- JAW_PIVOT, head rotation limits, vertex counts
