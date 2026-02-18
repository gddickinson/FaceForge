# src/faceforge -- Source Package

This is the main source package for FaceForge. All application code lives here, organized into focused subpackages.

## Entry Point

- **`app.py`** -- Application entry point (`main()` function). Wires together all systems: creates the Qt application, event bus, state manager, asset loader, scene builder, simulation loop, renderer, and UI. Registered as the `faceforge` console script via `pyproject.toml`.

- **`constants.py`** -- Shared constants and path definitions used across all subpackages. Includes face mesh vertex counts, skin thickness, skull/face alignment defaults, jaw pivot coordinates, camera defaults, head rotation limits, animation timing, and STL loading tier definitions.

## Subpackages

| Package | Purpose |
|---------|---------|
| `anatomy/` | Head and neck anatomical systems: skull, face mesh, FACS engine, jaw muscles, expression muscles, face features, neck muscles, vertebrae, neck constraints, head rotation |
| `animation/` | Animation subsystems: state interpolation, auto-blink, auto-breathing, eye tracking, micro-expressions, expression/pose presets |
| `body/` | Full-body systems: skeleton builder, joint pivots, body animation, soft tissue skinning, body constraints, muscles, organs, vasculature, brain |
| `coordination/` | High-level orchestration: loading pipeline, scene graph builder, per-frame simulation loop, visibility management |
| `core/` | Foundational infrastructure: scene graph, state management, mesh/material data structures, math utilities, event bus, clock, config loading |
| `loaders/` | Asset I/O: binary STL parser, batch STL loader with coordinate transforms, asset manager cache, mesh data JSON loader |
| `rendering/` | OpenGL 3.3 rendering: renderer, camera, shader programs, GL mesh (VAO/VBO), materials, lights, orbit controls, PySide6 GL widget |
| `ui/` | Qt GUI: main window, control panel, info panel, dark theme stylesheet, tab panels (animate, body, layers, align, display, debug), reusable widgets |

## Data Flow

```
app.py
  |-- creates EventBus, StateManager
  |-- creates AssetManager, SceneBuilder
  |-- LoadingPipeline loads assets in phases
  |-- Simulation.step(dt) called each frame:
  |     |-- StateInterpolator lerps toward targets
  |     |-- AutoBlink, AutoBreathing, MicroExpressions, EyeTracking
  |     |-- FACSEngine applies AU displacements
  |     |-- JawMuscles, ExpressionMuscles, FaceFeatures deform
  |     |-- HeadRotation distributes to vertebrae
  |     |-- NeckMuscles, NeckConstraints
  |     |-- BodyAnimation (spine, limbs, breathing)
  |     |-- SoftTissueSkinning updates all registered meshes
  |     |-- Scene.update_matrices() propagates transforms
  |-- GLRenderer.render(scene, camera) draws everything
  |-- UI tabs publish events, Simulation subscribes
```

## Dependencies

All subpackages depend on `core/` for fundamental types. The dependency graph flows:

```
core  <--  loaders  <--  anatomy
                    <--  body
      <--  animation
      <--  coordination (depends on anatomy, body, animation, loaders)
      <--  rendering
      <--  ui (depends on core, rendering)
```

`app.py` imports from all packages to wire them together.
