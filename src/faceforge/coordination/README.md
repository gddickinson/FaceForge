# coordination/ -- Loading and Simulation Orchestration

This package contains the high-level orchestration modules that wire together the anatomy, body, animation, and rendering systems. It manages the asset loading sequence, scene graph construction, per-frame simulation loop, and visibility control.

## Modules

### `loading_pipeline.py` -- Sequential Asset Loading

Orchestrates the phased loading of all anatomical assets with progress reporting.

**Key class:** `LoadingPipeline`

Loading order (mirrors the original JS implementation):

| Phase | Assets |
|-------|--------|
| 0 | Skull mesh (from embedded JSON data) |
| 1 | Face mesh (468-vertex MediaPipe landmarks) |
| 2 | Jaw muscles (22 STLs) -> Expression muscles (38 STLs) -> Face features (8 STLs) -> Neck muscles (36 STLs) -> Vertebrae (15 STLs) |
| 3 | Body skeleton: thoracic spine, lumbar spine, rib cage, pelvis, upper/lower limbs, hands, feet (~200 STLs) |
| 4+ | On-demand: body muscles, organs, vasculature, brain (toggled by layer UI) |

The pipeline emits `EventType.LOADING_PROGRESS` and `EventType.LOADING_PHASE` events so the UI loading overlay can display status.

After loading, the pipeline constructs and returns references to all anatomy systems (FACS engine, jaw/expression/neck muscle systems, face features, vertebrae, head rotation, neck constraints) that the simulation loop needs.

### `scene_builder.py` -- Scene Graph Construction

Builds the hierarchical scene graph from loaded assets.

**Key class:** `SceneBuilder`

Scene hierarchy (mirrors Three.js):
```
scene
  +-- bodyRoot
        +-- skullGroup
        |     +-- cranium
        |     +-- jawPivot
        |           +-- jaw
        |           +-- lower_teeth
        +-- faceGroup
        +-- stlMuscleGroup (jaw muscles)
        +-- exprMuscleGroup (expression muscles)
        +-- faceFeatureGroup
        +-- neckMuscleGroup
        +-- vertebraeGroup
        +-- thoracicSpineGroup
        +-- lumbarSpineGroup
        +-- ribCageGroup
        +-- pelvisGroup
        +-- [limb groups...]
        +-- [on-demand: body muscles, organs, vasculature, brain]
```

The `build()` method returns `(Scene, named_nodes_dict)` for other systems to reference specific nodes.

### `simulation.py` -- Per-Frame Simulation Loop

Drives all animation and deformation systems each frame.

**Key class:** `Simulation`

Call order per frame (mirrors the JS `animate()` function):

1. State interpolation (smooth lerp toward targets)
2. Auto-animations (blink, breathing, micro-expressions, eye tracking)
3. FACS displacement + jaw pivot rotation
4. Jaw muscle deformation
5. Expression muscle deformation
6. Face feature updates (eyes, eyebrows, nose, ears)
7. Head rotation + vertebra distribution
8. Neck muscle deformation
9. Neck constraint solving
10. Body animation (spine, limbs, breathing)
11. Soft tissue skinning update
12. Scene graph matrix propagation

Systems whose parent group is invisible are skipped for performance.

### `visibility.py` -- Layer Toggle Management

Maps UI toggle names (e.g., `"tog-skull"`, `"tog-neck"`, `"tog-vertebrae"`) to scene nodes for visibility control.

**Key class:** `VisibilityManager`
- `register(toggle_name, node)` -- Associate a SceneNode with a toggle
- `set_visible(toggle_name, visible)` -- Set visibility for all nodes under a toggle
- `is_visible(toggle_name)` -- Query current visibility state
- `get_toggle_names()` -- List all registered toggles

## Internal Dependencies

- `core.scene_graph` -- Scene, SceneNode
- `core.state` -- StateManager
- `core.events` -- EventBus, EventType for progress reporting
- `anatomy.*` -- All head/neck anatomy systems
- `body.*` -- Skeleton, soft tissue skinning, body animation
- `animation.*` -- Interpolation, auto-behaviors, presets
- `loaders.asset_manager` -- AssetManager for file I/O
