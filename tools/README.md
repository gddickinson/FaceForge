# `tools/`

39 Python scripts, of which 21 are maintained infrastructure,
5 are visual runners kept but unverified, and 13 are one-off
investigations that outlived their bug. This file says which is which, so the
next person does not have to guess.

**Nothing has been deleted.** The retirement list below is a recommendation;
every file named in it is still present and still imports cleanly.

## How to regenerate the evidence

```
python tools/inventory_tools.py                       # table
python tools/inventory_tools.py --json inventory.json  # machine-readable
```

`inventory_tools.py` measures four things per script: whether it imports in a
clean subprocess, whether it has a `__main__` entry point, which other files
import it, and how many ruff findings it has. It does **not** assign verdicts —
those are editorial and live here.

### What was and was not verified

| Claim | How |
| --- | --- |
| All 39 scripts import cleanly | `inventory_tools.py`, subprocess import of each, 0 failures |
| Import counts ("used by N files") | tree scan for `tools.<name>` across `tools/`, `tests/`, `src/`, `.github/` |
| ruff findings per file | `ruff check --output-format json tools` under the project's own config |
| **Whether each script still produces correct output** | **not verified** — most need the full 1.2 GB asset set, a GPU, or minutes of compute; only the two ported test scripts were actually executed |

The verdicts below therefore rest on: import health (measured), importer count
(measured), and overlap of stated purpose (read from the docstrings and code).

## Maintained set

### Golden-image and render validation

| Script | What it does |
| --- | --- |
| `glcontext.py` | Acquires a headless GL 4.1 core context on macOS. Imported by 6 files; the foundation of all headless rendering here. |
| `capture_golden.py` | Renders the fixed 16-mesh scene in all 16 modes to PNG through an FBO. Imported by 9 files. |
| `compare_golden.py` | Diffs two capture directories. Measured noise floor between identical captures is exactly zero, so any non-zero difference is signal. |
| `render_agent.py` | Watcher that renders golden-image jobs on this machine's GPU. |
| `verify_state_pixels.py` | Proves a saved `SceneState` reproduces identical pixels. |
| `bench_render_real.py` | Real-GPU frame-time benchmark. |
| `glsl_cpu.py` | numpy transliteration of the fragment shaders, for shader-semantics tests. Library (no entry point), imported by 2. |

### Assets and generated data

| Script | What it does |
| --- | --- |
| `fetch_assets.py` | Verifies the BodyParts3D asset set and builds the welded-mesh disk cache. |
| `generate_fma_taxonomy.py` | Generates `assets/config/fma_taxonomy.json` (FMA is-a, part-of and composite-of edges) from the BodyParts3D `FMA.csv` and its two relation tables. Run it when the upstream dataset changes; the 5.9 MB source stays outside this repo. |
| `generate_readme_images.py`, `generate_scanner_images.py`, `capture_gui_screenshots.py` | Documentation image generators. |

### Shared libraries (no entry point — imported, not run)

| Script | Importers |
| --- | --- |
| `headless_loader.py` | 30 — the headless scene/skeleton/tissue loader everything else builds on |
| `skinning_scorer.py` | 16 — pose set and scoring |
| `mesh_renderer.py` | 8 — headless mesh renderer with edge-stretch colouring |
| `head_renderer.py` | 2 |
| `head_rotation_diagnostic.py` | 2 |
| `bone_specificity.py` | 1 |
| `glsl_cpu.py` | 2 (listed above) |

### Diagnostics with real CLIs

| Script | What it does |
| --- | --- |
| `run_skinning_diagnostic.py` | Four modes (diagnose / optimise / test / bone-test) with ~15 arguments. **This is the maintained entry point for skinning work** and is what supersedes most of the retirement list. |
| `detect_stuck.py` | Stuck-vertex detection with four selectable strategies (`assigned_joint`, `nearest_joint`, `region_based`, `neighbor_contrast`). |
| `inventory_tools.py` | This file's evidence generator. |

### Visual runners (kept, but unverified)

`head_rotation_test.py`, `visual_skinning_test.py`, `gender_scaling_test.py`,
`scene_mode_diagnostic.py`, `analyze_mesh_topology.py`.

These are thin visual/one-shot front ends over the libraries above. They import
cleanly and have entry points; whether their output is still correct was not
checked. Keep unless someone confirms they are dead.

## Moved into `tests/`

Two files named `test_*.py` lived in `tools/`, where pytest never collected
them, and neither asserted anything — they printed a table for a human to read.
Both have been ported to real tests; the originals are still in `tools/` and
are recommended for removal.

| Was | Now | Note |
| --- | --- | --- |
| `tools/test_boundary_smoothing.py` | `tests/tools/test_boundary_smoothing_effect.py` | 4 tests, 6.6 s. The port found the real switch (`BOUNDARY_SMOOTH_PASSES` on the skinning manager class) and now asserts that smoothing reduces cross-chain discontinuity, which the script only printed. |
| `tools/test_neighbor_clamp.py` | `tests/tools/test_neighbor_stretch_clamp.py` | 6 tests. The port surfaced a finding: the clamp lowers the *tail* of edge stretch (99.9th percentile 10.12 vs 10.44; edges over 5× 7064 vs 7156) but leaves the **single worst edge higher** (778× vs 517× on `extreme_arm_raise`). See the test's docstring. |

Both are marked `slow`.

## Recommended for retirement

Listed for the user to remove in one action. None has any importer, all print
to stdout only, and each is superseded by a maintained tool or by a test.

### Skin mis-binding investigations → `run_skinning_diagnostic.py`, `detect_stuck.py`

`debug_arm_binding.py`, `debug_arm_hip_overlap.py`, `debug_arm_misbinding.py`,
`debug_hip_misbinding.py`, `debug_pelvis.py`, `debug_thigh_spikes.py`,
`debug_zero_disp.py`

Each was written to investigate one specific mis-binding bug (arm chain
claiming torso vertices, hip chain claiming groin vertices, thigh spikes) and
hardcodes the coordinate ranges of that bug. The general capability — per-joint
displacement, per-region binding, stuck-vertex detection — is in
`run_skinning_diagnostic.py` and `detect_stuck.py` with arguments.

### `check_static.py` → `detect_stuck.py`

**Ruff does not supersede this, and the name is the reason to think it might.**
`check_static.py` is not a static-analysis linter: it finds *stationary*
vertices — skin that does not move when the skeleton does. Ruff and it have
nothing in common. It is superseded by `detect_stuck.py`, whose
`assigned_joint` and `neighbor_contrast` strategies cover the same question
with a threshold argument and four strategies instead of one hardcoded pose.

(Separately: ruff runs over `tools/` in CI and currently reports 16 findings
here under the project's configuration.)

### Platysma fascia investigations → `tests/tools/test_fascia_integration.py`

`debug_platysma_fascia.py`, `debug_platysma_fascia_comparison.py`,
`debug_platysma_minimal.py`

Three scripts on one question — whether fascia pinning reaches the Platysma's
body-end vertices. That behaviour is now asserted in
`tests/tools/test_fascia_integration.py`, which runs in CI; the scripts only
print.

### Superseded by the ports above

`test_boundary_smoothing.py`, `test_neighbor_clamp.py` — see the table above.
Removing them also stops two files named `test_*` sitting where pytest cannot
reach them, which is its own trap.

## Summary

| | Count |
| --- | --- |
| Maintained (render/asset/library/CLI) | 21 |
| Visual runners kept but unverified | 5 |
| Recommended for retirement | 13 |
| **Total** | **39** (`__init__.py` excluded) |

Of the 13 recommended for retirement, two (`test_boundary_smoothing.py`,
`test_neighbor_clamp.py`) have been ported into `tests/` and are counted here,
not twice.

---

# Per-tool reference

The section below is the previous contents of this file, retained verbatim: it
documents each script's arguments and key functions, which the inventory above
deliberately does not repeat. Note that it predates the verdicts above and
describes several scripts that are now recommended for retirement — the
inventory is authoritative on status, this section on usage.

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
