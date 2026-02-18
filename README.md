# FaceForge

FACS-based facial animation with full-body anatomy visualization.

FaceForge renders anatomically accurate 3D models of the human head and body using BodyParts3D (BP3D) STL meshes. Facial expressions are driven by the Facial Action Coding System (FACS), with 12 Action Units controlling muscle contraction, jaw movement, eye tracking, and skin deformation. The full-body extension adds a complete skeleton, muscles, organs, vasculature, and brain -- all deformable via a delta-matrix soft tissue skinning system.

## Architecture

The application is built in Python with:

- **PySide6 (Qt 6)** for the desktop GUI
- **PyOpenGL** (OpenGL 3.3 core profile) for 3D rendering
- **NumPy** for all vertex math and matrix operations
- **BodyParts3D STL files** for anatomical mesh data

The codebase is organized into a clean modular structure under `src/faceforge/`, with each subsystem in its own package (anatomy, animation, body, coordination, core, loaders, rendering, ui).

## Requirements

- Python 3.11+
- numpy >= 1.24
- PyOpenGL >= 3.1.7
- PySide6 >= 6.6
- glfw >= 2.6
- BodyParts3D STL files (symlinked at `assets/stl/`)

Optional (dev): pytest, pytest-qt. Optional (tools): scipy.

## Installation

```bash
pip install -e .
# Or with dev dependencies:
pip install -e ".[dev]"
```

## Running

```bash
faceforge
# Or:
python -m faceforge.app
```

## Project Structure

```
faceforge/
  src/faceforge/           # Main source package
    app.py                 # Application entry point, wires all systems together
    constants.py           # Shared constants, paths, and project configuration
    anatomy/               # Head anatomy: skull, face, muscles, vertebrae, FACS
    animation/             # Auto-blink, breathing, interpolation, presets
    body/                  # Full-body: skeleton, soft tissue skinning, organs
    coordination/          # Loading pipeline, scene building, simulation loop
    core/                  # Scene graph, state management, math, events, mesh
    loaders/               # STL parser, batch loader, asset manager
    rendering/             # OpenGL renderer, camera, shaders, orbit controls
    ui/                    # Qt GUI: main window, tabs, widgets
  assets/
    config/                # JSON configs for muscles, skeleton, expressions
    meshdata/              # Embedded mesh data (skull, face landmarks)
    stl -> ../../bodyparts3D/stl  # Symlink to BP3D STL files
  tests/                   # pytest test suite
  tools/                   # Diagnostic and debugging utilities
  pyproject.toml           # Build configuration
```

## Loading Pipeline

Assets are loaded in tiered phases to keep startup fast:

| Tier | Phase | Contents | When |
|------|-------|----------|------|
| 0 | Skull | Cranium, jaw, teeth from embedded data | Startup |
| 1 | Head | Jaw muscles, expression muscles, face features, neck, vertebrae | Startup |
| 2 | Skeleton | Thoracic/lumbar spine, ribs, pelvis, limbs, hands, feet (~200 STLs) | Startup |
| 3 | Muscles | Back, shoulder, arm, torso, hip, leg muscles | On demand |
| 4 | Organs | 52 organ structures | On demand |
| 5 | Vascular/Brain | 50 vascular + 80 brain structures | On demand |

## Key Systems

- **FACS Engine**: Applies Action Unit displacements to the 468-vertex MediaPipe face mesh
- **Head Rotation**: Yaw/pitch/roll distributed across cervical vertebrae with constraint solving
- **Neck Constraints**: Tension monitoring, soft-clamping, and dynamic thoracic compensation
- **Soft Tissue Skinning**: Delta-matrix system with dual quaternion blending for body deformation
- **Body Animation**: Spine flex/bend/rotation, limb articulation, breathing cycle
- **Expression Presets**: 12 expressions (neutral, happy, sad, angry, surprised, fear, disgust, contempt, pout, kiss, pain, thinking)
- **Body Poses**: 6 pose presets (anatomical, relaxed, walking, sitting, reaching, crouching)

## License

See project files for license information.
