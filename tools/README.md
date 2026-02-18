# tools/ -- Diagnostic and Debugging Utilities

Standalone scripts for analyzing, debugging, and validating the soft tissue skinning system. These tools operate headlessly (no Qt or OpenGL required) by replicating the asset loading pipeline and running diagnostics on the resulting mesh bindings and deformations.

Most tools depend on `headless_loader.py` to bootstrap a scene, and produce results as console output, JSON files, or PNG images in the `results/` directory.

## Usage

All tools are run from the project root as Python modules:

```bash
# From the project root:
python -m tools.run_skinning_diagnostic --diagnose
python -m tools.visual_skinning_test --quick
python -m tools.bone_specificity
```

## Core Tools

### `headless_loader.py` -- Headless Scene Loader

Replicates the `app.py` loading sequence (skeleton, joint chains, soft tissue skinning) without any Qt or OpenGL imports. Provides the foundation for all other tools.

**Key functions:**
- `load_headless_scene()` -- Returns a `HeadlessScene` dataclass with scene, body root, skeleton, joints, animation, skinning, and constraints
- `load_layer(hs, name)` -- Load a specific tissue layer (skin, muscles by region, organs, vasculature)
- `register_layer(hs, meshes, name)` -- Register meshes with the skinning system
- `apply_pose(hs, body_state)` -- Apply a body pose and run the animation/skinning pipeline

### `run_skinning_diagnostic.py` -- Skinning Diagnostic CLI

Full-featured CLI for diagnosing and optimizing skinning parameters.

**Modes:**
- `--diagnose` -- Run displacement/distortion checks across poses and layers
- `--optimize` -- Parameter optimization using scipy (requires `[tools]` dependency)
- `--test` -- Test specific parameter values
- `--output results.json` -- Save results to file

### `skinning_scorer.py` -- Skinning Quality Scorer

Evaluates skinning quality by applying multiple body poses and running diagnostic checks, producing a composite score suitable for automated optimization.

**Key class:** `SkinningScorer`
- Loads 6 preset poses plus custom extreme poses
- Runs `SkinningDiagnostic` on each
- Produces a single composite score

### `visual_skinning_test.py` -- Visual Skinning Tester

Renders 3D meshes with edge-stretch coloring to PNG images for visual inspection.

```bash
python -m tools.visual_skinning_test              # Full test
python -m tools.visual_skinning_test --quick       # Quick: 2 poses, 2 views
python -m tools.visual_skinning_test --poses sitting arm_raise
python -m tools.visual_skinning_test --all-views
```

### `mesh_renderer.py` -- Headless Mesh Renderer

Software rasterizer for generating PNG images without OpenGL. Uses PIL for triangle rasterization with orthographic projection, backface culling, painter's algorithm depth sorting, and per-triangle edge-stretch coloring.

**Key function:** `render_mesh(positions, rest_positions, triangles, ...)`

## Debugging Tools

### `bone_specificity.py` -- Binding Specificity Checker

Verifies that vertices only respond to their associated bone chains. For each DOF, isolates the movement and checks that only the expected chains' vertices are displaced.

### `detect_stuck.py` -- Stuck Vertex Detection

Finds vertices that do not move when they should, using 4 strategies:
1. ASSIGNED-JOINT: vertex vs its assigned joint
2. NEAREST-JOINT: vertex vs nearest moving joint
3. REGION-BASED: groups by body region, flags low-movement regions
4. NEIGHBOR-CONTRAST: compares displacement to mesh neighbors

### `check_static.py` -- Static Vertex Finder

Quick investigation tool to find vertices that remain static when the skeleton moves. Tests multiple poses and reports per-layer statistics.

### `debug_arm_binding.py` -- Arm Binding Debugger

Detailed analysis of arm mesh vertex bindings: chain assignments, distances, spatial relationships to arm vs leg chains.

### `debug_arm_misbinding.py` -- Arm Misbinding Debugger

Investigates specific cases where arm vertices are incorrectly bound to leg or spine chains.

### `debug_arm_hip_overlap.py` -- Arm/Hip Overlap Debugger

Analyzes spatial overlap between arm and hip chain vertices that could cause cross-binding artifacts.

### `debug_hip_misbinding.py` -- Hip Misbinding Debugger

Investigates hip region vertices bound to incorrect chains (arm, foot, hand).

### `debug_pelvis.py` -- Pelvis Binding Debugger

Analyzes pelvis mesh vertex assignments and their relationship to nearby joints.

### `debug_thigh_spikes.py` -- Thigh Spike Debugger

Investigates spiking artifacts in thigh region during leg poses.

### `debug_zero_disp.py` -- Zero Displacement Debugger

Finds vertices with zero displacement despite being bound to joints that have moved.

### `test_boundary_smoothing.py` -- Boundary Smoothing Tester

Tests the boundary smoothing system that blends deformation at chain boundaries.

### `test_neighbor_clamp.py` -- Neighbor Clamp Tester

Tests the neighbor-stretch clamping system that prevents vertices from stretching beyond a threshold relative to their neighbors.

## External Dependencies

- `numpy` -- Array operations (all tools)
- `PIL/Pillow` -- PNG rendering (`mesh_renderer.py`, `visual_skinning_test.py`)
- `scipy` -- Parameter optimization (`run_skinning_diagnostic.py --optimize`)

## Internal Dependencies

- `faceforge.body.*` -- Skinning, diagnostics, skeleton, animation
- `faceforge.core.*` -- State, scene graph, mesh, math
- `faceforge.loaders.*` -- Asset loading
- `faceforge.coordination.*` -- Loading pipeline
