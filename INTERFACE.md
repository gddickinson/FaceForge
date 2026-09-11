# INTERFACE.md — FaceForge navigation map

**Read this first.** What each package and module contains, the key classes,
and how they connect. Update it whenever the structure changes
(`python -m tools.export_exercise_docs` regenerates the exercise list;
this file is maintained by hand). Rig facts that were *measured*, not
assumed, live in `docs/exercise_animation.md`; the sex morph's measurements
live in `docs/sex_morph.md`.

## Running and testing

```bash
# Launchers (pick the interpreter with FaceForge's dependencies; run from anywhere):
./start_faceforge.sh                                          # the full GUI
./start_exercise_viewer.sh --exercise pull_up                 # the standalone exercise viewer

# The editable install lives in the `flika` conda environment (Python 3.11).
/opt/anaconda3/envs/flika/bin/python -m faceforge.app        # GUI
/opt/anaconda3/envs/flika/bin/python -m faceforge.cli --help # headless render/scan/export
PYTHONPATH=src /opt/anaconda3/envs/flika/bin/python -m faceforge.exercise_viewer --exercise conventional_deadlift  # exercise viewer only
export QT_QPA_PLATFORM=offscreen
/opt/anaconda3/envs/flika/bin/python -m pytest -m "not slow"  # fast tier, no assets needed (~20 s)
/opt/anaconda3/envs/flika/bin/python -m pytest                # full tier (needs assets/stl)
python -m tools.fetch_assets verify                           # BodyParts3D asset check
python -m tools.render_exercise_demo --exercise bodyweight_squat   # exercise demo frames/MP4/GIF
python -m tools.render_exercise_demo --probe --all                 # placement check, no GL
```

## Top-level layout

```
FaceForge/
├── src/faceforge/        the package (see the map below)
├── assets/config/        JSON: structures, muscles/, skeleton/, joint limits, poses, DOF→muscle map
├── assets/stl            symlink to the BodyParts3D STL set (outside the repo)
├── tools/                headless loaders, renderers, diagnostics, README image generators
├── tests/                pytest; `slow` marks asset-heavy and whole-app tests
├── docs/                 headless_cli.md, exercise_animation.md, sex_morph.md, exercises.md (generated), research/
├── results/              generated outputs (exercise demo frames live in results/exercise_demo/)
├── start_faceforge.sh    launcher: the full GUI
├── start_exercise_viewer.sh  launcher: the standalone exercise viewer (--exercise ID, --list)
├── INTERFACE.md          ← you are here
├── SESSION_LOG.md        work log
└── CLAUDE.md             project instructions (refs @INTERFACE.md)
```

## How the application is assembled

1. `appcontext.build_app_context()` constructs every collaborator (event bus,
   `StateManager`, scene graph, `Simulation`, GL viewport, main window,
   `AnimationPlayer`, `MuscleActivationSystem`, …) into an `AppContext`.
2. `controllers.build_controllers(ctx)` constructs one controller per concern
   and subscribes its handlers to the `EventBus` in a fixed order
   (`tests/app/test_app_wiring.py` asserts the exact subscription map).
3. `coordination.asset_load_sequence.AssetLoadSequence` loads the anatomy in
   stages (head → body skeleton → joint chains → skinning → finalise) and
   `coordination.demand_loaders.DemandLoaders` loads layers on first enable.
4. `controllers.frame_loop.FrameLoop` wraps `paintGL`: speech → `Simulation.step(dt)`
   → draw → transport/exercise/label updates.
5. `coordination.simulation.Simulation.step()` runs the animation player, state
   interpolation, FACS, head rotation, body animation, neck/skin/muscle
   deformation, the heatmap, then updates scene matrices. Two hook lists,
   `after_animation_hooks` and `after_scene_update_hooks`, let the exercise
   runtime sample its activation track and re-anchor the feet.

State: `core/state.py` (`FaceState`, `BodyState`, targets); UI writes targets
through events, `animation/interpolation.py` lerps live state toward them.
Poses are normalised DOF values; `body/dof_ranges.py` is the one table that
converts them to degrees and names them anatomically.

## The exercise system (`faceforge/exercise/`)

```
ExerciseDefinition (model.py)  ── catalog/*.py author them in degrees via pose_library.py
        │  phases, muscles by role, equipment, cues, sources
        ▼
clip_builder.build_exercise_clip() ──▶ ExerciseClip
        │   AnimationClip (full pose per keyframe, wrapper quaternion/position per phase)
        │   PhaseSpan timeline with JointMotion descriptions (motion_description.py)
        │   ActivationTrack (activation.py: role × phase-kind → per-muscle level over time)
        ▼
runtime.ExerciseRuntime  (no Qt; also driven by tools/render_exercise_demo.py)
        │   loads the clip into the shared AnimationPlayer
        │   after_animation: track.sample(t) → MuscleActivationSystem.set_levels()
        │   after_scene_update: body/ground_contact.GroundLock re-anchors feet/hands,
        │                       equipment_rig.EquipmentRig places equipment.py nodes
        ▼
controllers/exercise.ExerciseController  ◀── EXERCISE_* events ◀── ui/tabs/exercise_tab.py
        enters the gym scene, loads muscle regions (muscle_groups.regions_for_groups),
        publishes EXERCISE_STATUS (phase, cue, moving joints, group levels) per frame
```

Key facts: body frame +Z up / −Y anterior / +X right; abduction is a rotation
about Y and axial rotation about Z (fixed in `body_animation._apply_limbs`);
the arms hang off the pelvis root, so trunk lean is a whole-body pitch
(`Phase.pitch`, about `Phase.pivot`) plus matching hip flexion; the sole is
flat when ankle dorsiflexion = pitch − hip + knee.

## Module map

### `faceforge/` (root)

| module | purpose |
|---|---|
| `app.py` | FaceForge application entry point. |
| `appcontext.py` | The collaborators an assembled FaceForge application is made of. |
| `cli.py` | ``faceforge-cli`` -- the scriptable half of FaceForge. |
| `exercise_viewer.py` | `python -m faceforge.exercise_viewer [--exercise ID]`: the application opened straight into the exercise-viewer mode (skeleton preset, then every muscle layer). `watch_load_sequence` enters the mode when the load *sequence* reaches `COMPLETE` (not on `LOADING_COMPLETE`, which the skeleton pipeline publishes before body animation, rib pivots and skinning are wired). |
| `constants.py` | Shared constants and paths for FaceForge. |
| `session.py` | A headless, scriptable FaceForge render session. |

### `faceforge/anatomy/`

| module | purpose |
|---|---|
| `answer_explanations.py` | Why a wrong quiz answer was wrong, in real anatomical terms. |
| `back_neck_muscles.py` | Body-end pinning for back-of-neck muscles. |
| `bone_anchors.py` | Registry mapping muscle names to bone SceneNode references. |
| `bone_collision.py` | Capsule-based bone collision: capsules live in each bone's local frame, follow it every frame (`refresh`), and resolve only penetration beyond a vertex's rest depth. |
| `curricula.py` | Named, ordered study sets ("curricula") derived from the project's own data. |
| `distractors.py` | Distractor selection: anatomically adjacent wrong answers, deterministically. |
| `exam_items.py` | Exam item schema, provenance, and the gate that refuses ungrounded items. |
| `exam_session.py` | Exam sessions: assemble items, run a format, record the outcome. |
| `expression_muscles.py` | 38 STL expression muscles with AU-driven contraction. |
| `face.py` | Build face mesh with positioning for skull alignment. |
| `face_features.py` | Face features: eyes, ears, nose cartilage, eyebrows, throat structures. |
| `facs.py` | FACS (Facial Action Coding System) engine -- applies AU displacements to face mesh vertices. |
| `fascia.py` | Virtual fascia constraint surfaces for muscle attachment. |
| `fibre_field.py` | Harmonic fibre interpolation: a footprinted muscle's belly stretches between the rigid images of its two attachments instead of bowing; `build_fibre_field`, `trim_footprints`, disk-cached via `cached_fibre_field`. |
| `fma_taxonomy.py` | Read-only access to the FMA relation graph shipped in assets/config. |
| `head_rotation.py` | Head yaw/pitch/roll rotation with cervical vertebra distribution. |
| `item_generators.py` | Exam item generators. Every fact comes from data; none is authored here. |
| `jaw_muscles.py` | 22 STL jaw muscles with jaw-angle deformation. |
| `muscle_attachments.py` | Attachment placement for body muscles: authored footprints (trimmed to a geodesic gap) drive the fibre field; stretch is measured, never clamped. |
| `neck_constraints.py` | Neck constraint solver: tension monitoring, soft-clamping, spine compensation. |
| `neck_muscles.py` | 36 STL neck muscles with head-follow deformation. |
| `pathology.py` | Pathology visualization system. |
| `platysma.py` | Body-spanning deformation for Platysma muscles. |
| `quiz_engine.py` | Interactive anatomy quiz engine. |
| `quiz_progress.py` | Per-user quiz progress: attempt history plus SM-2 card state, on disk. |
| `radiology_items.py` | L4: identify a tagged structure on a simulated cross-section. |
| `skull.py` | Build skull mesh hierarchy from loaded mesh data. |
| `spaced_repetition.py` | SM-2 spaced-repetition scheduling for the anatomy quiz. |
| `structure_search.py` | Natural language anatomical structure search. |
| `vertebrae.py` | Cervical vertebrae (15 STLs) with articulation pivots. |

### `faceforge/animation/`

| module | purpose |
|---|---|
| `auto_blink.py` | Automatic blinking system. |
| `auto_breathing.py` | Automatic breathing cycle for face (nostril flare + subtle jaw). |
| `eye_tracking.py` | Mouse-follow eye positioning. |
| `interpolation.py` | State interpolation for smooth transitions. |
| `micro_expressions.py` | Micro-expression generator: random subtle AU flickers. |
| `preset_manager.py` | Expression and body pose preset management. |
| `speech.py` | Speech & Phoneme Animation system. |

### `faceforge/body/`

| module | purpose |
|---|---|
| `blood_flow.py` | Blood flow particle effect module. |
| `body_animation.py` | Body animation: spine flex/bend/rotation, limb articulation, breathing; scapulohumeral rhythm as a glide on the thorax with clavicle elevation (`_apply_girdle`); pronation is the outermost wrist rotation. |
| `body_constraints.py` | Body joint limit enforcement via simple clamping. |
| `body_muscles.py` | On-demand body muscle loading and management. |
| `bone_scaling.py` | Female/male scale factor per bone, from `assets/config/gender_dimorphism.json`; matches a bone name only on whole words, so "Tibialis" is not a tibia. |
| `brain.py` | On-demand brain loading. |
| `centres_of_rotation.py` | Per-vertex centres of rotation for skin deformation. |
| `chain_overrides.py` | JSON persistence for vertex chain reassignment overrides. |
| `chain_reassignment.py` | Reassign selected vertices to a different kinematic chain. |
| `diagnostics.py` | Skinning diagnostics: detect mesh vertices displaced beyond expected limits. |
| `dof_ranges.py` | The body's joint degrees of freedom: range, sign convention and anatomical name. |
| `edge_relaxation.py` | Distance-constraint relaxation: `relax_edges` (one-sided, for poses) and `enforce_edge_range` (two-sided, so a projection cannot flatten a limb). |
| `gender_morph.py` | The sex morph's front door: the male/female surface pair, the skeleton morph, the soft-tissue field, and the warp of the surface mesh onto the skeleton. |
| `ground_contact.py` | Keep the feet (or hands) on the floor while the joints move; hands anchored to a point (a bar) are measured at the closed-finger ring. |
| `hand_points.py` | `finger_ring_centre`: the centroid of the closed finger joints, where a held bar's axis passes (shared by the equipment rig and the ground lock). |
| `joint_pivots.py` | Joint pivot setup for limb articulation; digit pivots sit at each phalanx's proximal end (`proximal_end`). |
| `skinning_ops.py` | The gathered-einsum `transform_points` / `rotate_vectors` every skinning pass uses (measured against grouped alternatives), `used_joints` (bincount), `accumulate_rows` (bincount face-normal sums replacing `np.add.at`). |
| `muscle_activation.py` | Muscle activation heatmap: colour each muscle by how hard it is working. |
| `neural_impulse.py` | Neural impulse particle effect module. |
| `organs.py` | On-demand organ loading. |
| `physiology.py` | Physiological simulation systems: heartbeat, blood flow, breathing, digestion, fasciculation. |
| `region_labels.py` | Anatomical region labeling for body mesh segmentation. |
| `skeleton.py` | Build full-body skeleton from STL batches. |
| `skeleton_field.py` | Turns a skeleton change into a smooth spatial warp: joint displacements as a thin-plate spline (`displacement_warp`), sampled on a lattice (`sampled_warp`). |
| `skeleton_joints.py` | Shuts an articulation whose two bones scale apart (the acromioclavicular joint), from a contact patch measured on the unscaled skeleton. |
| `skeleton_morph.py` | `SkeletonMorph`: scales the skeleton as an articulated hierarchy -- bones about the joint they hang from, joints moved to the end of the scaled bone -- so proportions change without the joints coming apart. |
| `skin_morph.py` | `SkinShapeMorph`: the female-minus-male soft-tissue field (breast, gluteal and thigh fat, waist), measured from the surface pair and transferred to the model's own skin. |
| `soft_tissue_morph.py` | `SoftTissueMorph`: composes the skeleton warp, the muscle bulk change and the soft-tissue field onto every mesh's rest pose, always from a captured original. |
| `muscle_morph.py` | `MuscleMorph`: thins a muscle belly perpendicular to its own long axis, tapered to nothing at the attachments. |
| `skinning_cache.py` | Disk cache for the soft-tissue binding solve. |
| `surface_fit.py` | The refinement that pulls the registered surface mesh the last few units onto the reference skin, under edge-length constraints. |
| `surface_landmarks.py` | Joint landmarks from the skeleton's own bones, and the same joints found on a body mesh by shape (the wrist and ankle are where a limb is narrowest). |
| `surface_register.py` | `register_onto`: limb-by-limb registration of the body-surface mesh onto the skeleton, held inside an edge-length band; `fit_head_to_skull`. |
| `surface_projection.py` | The geometry the fit is built from: closest point on a triangle, region-constrained projection, edges, Laplacian smoothing, normals. |
| `soft_tissue.py` | Delta-matrix soft tissue skinning for body muscles/organs/vasculature; `resnapshot_rest` re-snapshots the rest pose after the skeleton itself moves, keeping the binding. |
| `stretch_viz.py` | Stretch heatmap and chain assignment visualization for soft tissue skinning. |
| `vasculature.py` | On-demand vascular system loading. |

### `faceforge/controllers/`

| module | purpose |
|---|---|
| `__init__.py` | Application event handlers, grouped by concern. |
| `alignment.py` | Alignment of the scanned face mesh onto the skull. |
| `animation.py` | Animation clip playback: transport controls and the player's callbacks. |
| `body.py` | Body: joint pose, pose presets, and sexual dimorphism morphing. |
| `diagnostics.py` | The debug tab: skinning diagnostics, vertex selection and chain overrides. |
| `display.py` | Display: render mode, background, camera presets, clip plane, skull mode. |
| `exercise.py` | Exercise demonstrations: the app-side glue around :class:`ExerciseRuntime`. |
| `expression.py` | Face: Action Units, expression presets, head rotation, speech. |
| `frame_loop.py` | What happens on every frame, and in what order. |
| `labels.py` | Anatomical name labels drawn over the viewport. |
| `layers.py` | Layer and per-structure visibility, including load-on-first-enable. |
| `overlays.py` | Overlays that answer "where is it?": search highlight, heatmap, pathology. |
| `scene_view.py` | Themed scene mode: reposing the whole body inside an environment. |
| `tools.py` | The auxiliary windows: scanner, export, quiz, timeline, comparison. |

### `faceforge/coordination/`

| module | purpose |
|---|---|
| `asset_load_sequence.py` | The startup load, as an explicit ordered sequence of named stages. |
| `joint_chains.py` | The kinematic chains the skinning binds to (arm chain from the clavicle); shared by the app and the headless tools |
| `demand_loaders.py` | Load anatomy groups the first time the user asks to see them. |
| `loading_pipeline.py` | Sequential asset loading chain with progress reporting. |
| `render_mode_sync.py` | Keep newly loaded meshes in step with the render mode already on screen. |
| `scene_builder.py` | Constructs the scene graph from loaded assets. |
| `simulation.py` | Per-frame simulation orchestrator — mirrors the JS animate() function. |
| `visibility.py` | Layer toggle → node visibility mapping. |

### `faceforge/core/`

| module | purpose |
|---|---|
| `clock.py` | Delta clock for frame timing. |
| `config_loader.py` | JSON config file loading utilities. |
| `events.py` | EventBus for decoupled publish/subscribe communication. |
| `material.py` | Material definitions for rendering. |
| `math_utils.py` | NumPy-backed math utilities: Vec3, Quaternion, Mat4 operations. |
| `mesh.py` | Mesh data structures for geometry storage (no GL dependencies). |
| `scene_graph.py` | Scene graph with hierarchical transforms, mirroring Three.js group structure. |
| `state.py` | Application state management for face and body parameters. |

### `faceforge/core/scene_state/`

| module | purpose |
|---|---|
| `__init__.py` | Versioned, hash-stamped serialisation of everything a render depends on. |
| `binding.py` | Capturing a SceneState from live objects, and applying one back onto them. |
| `codec.py` | Reading and writing SceneState files. |
| `confighash.py` | Fingerprinting the anatomy config set. |
| `model.py` | The SceneState data model: everything needed to reproduce a render. |

### `faceforge/exercise/`

| module | purpose |
|---|---|
| `__init__.py` | Exercise demonstrations: technique, moving joints and working muscles. |
| `activation.py` | From muscle roles and phase kind to a per-frame activation level. |
| `clip_builder.py` | ExerciseDefinition -> a playable clip with phase spans and an activation track. |
| `equipment.py` | Procedural gym equipment, built from the scene's own primitives. |
| `equipment_rig.py` | Keep hand-held equipment in the hands, every frame: the bar's axis passes through the ring of closed finger joints (`grip_point`). |
| `grip_lock.py` | `GripWidthLock`: hands anchored to a fixed bar stop sliding along it; per frame, shoulder abduction is solved (2x2 finite-difference Newton) so each hand's offset from the trunk holds its calibrated value. |
| `model.py` | The data model for an exercise demonstration; `EquipmentSpec.hang` overrides how far below the hands a held item's origin sits (a goblet-held kettlebell, a front-racked bar). |
| `motion_description.py` | Turn a change of pose into the words a physiotherapist would use. |
| `muscle_groups.py` | Functional muscle groups -> the muscle mesh names in ``assets/config/muscles``; hand/foot intrinsics (side-prefixed names) and `ALL_MUSCLE_REGIONS`. |
| `pose_library.py` | Pose authoring for exercises: degrees in, normalised BodyState DOFs out. |
| `runtime.py` | The exercise runtime: drives a built clip through the existing animation player; `equipment_tuning(spec)` merges the kind's grip/hang/spin defaults with the spec's override. |
| `stabilisers.py` | Implied stabilisers: grip, carry, brace and stance muscles derived from the equipment, anchor and orientation, appended at clip-build time so a deadlift colours the hands, arms and back. |

### `faceforge/exercise/catalog/`

| module | purpose |
|---|---|
| `__init__.py` | The exercise catalogue, by category. |
| `_helpers.py` | Constructors and shared source citations for the catalogue modules. |
| `athletic.py` | Power and athletic movements: kettlebell swing, jumps, power clean, slam, burpee. |
| `conditioning.py` | Cyclic conditioning: bike, rower, walking, running, jump rope, jacks, climbers, ropes. |
| `core_stability.py` | Trunk: planks, sit-up, crunch, dead bug, bird dog, twist, knee raise, Pallof press. |
| `lower_body.py` | Squats, deadlifts, lunges and split squats. |
| `lower_body_accessory.py` | Hip thrust, step-up, calf raise, wall sit, machine knee work and the physio staples. |
| `upper_pull.py` | Pulling: pull-up, chin-up, pulldown, rows, face pull, curls and rotator cuff. |
| `upper_push.py` | Pressing: bench press, incline press, push-up, overhead presses, dips, flyes, triceps. |

### `faceforge/export/`

| module | purpose |
|---|---|
| `baking.py` | World-space baking shared by every geometry exporter. |
| `dicom.py` | DICOM export of virtual-scanner volumes. |
| `glb_exporter.py` | Export visible scene meshes to GLB (binary glTF 2.0) for Blender import. |
| `hounsfield.py` | What the virtual scanner's numbers are, and what they are not. |
| `mesh_export.py` | OBJ, PLY and STL export of scene geometry, alongside the GLB exporter. |
| `nifti.py` | NIfTI-1 export of virtual-scanner volumes. |
| `provenance.py` | Attribution and per-structure provenance for everything FaceForge exports. |
| `still.py` | True-resolution offscreen stills, and the evidence that they are one. |
| `video_export.py` | Video/GIF export from the GL viewport. |
| `volume.py` | Stack virtual-scanner slices into a volume with defensible geometry. |

### `faceforge/loaders/`

| module | purpose |
|---|---|
| `asset_manager.py` | Central asset cache and lazy loading manager. |
| `mesh_data_loader.py` | Load skull and face mesh data from extracted JSON files. |
| `obj_parser.py` | Wavefront OBJ parser → BufferGeometry. |
| `stl_batch_loader.py` | Batch STL loader with BodyParts3D coordinate transform. |
| `stl_parser.py` | Binary STL parser with indexed geometry support. |
| `target_parser.py` | MakeHuman .target file parser → dense delta array. |

### `faceforge/rendering/`

| module | purpose |
|---|---|
| `__init__.py` | Rendering subsystem -- OpenGL 3.3 core profile with PySide6 integration. |
| `camera.py` | Perspective camera with view and projection matrices. |
| `gl_material.py` | Apply Material properties to a shader program and configure GL state. |
| `gl_mesh.py` | VAO / VBO management for uploading and drawing mesh geometry. |
| `gl_widget.py` | PySide6 QOpenGLWidget subclass bridging Qt and OpenGL rendering. |
| `lights.py` | Ambient, directional, and point light setup for Phong shading. |
| `orbit_controls.py` | Mouse-driven orbit, pan, and zoom controls using spherical coordinates. |
| `particle_system.py` | Lightweight particle system for physiological effects. |
| `renderer.py` | Main OpenGL renderer -- traverses the scene graph and issues draw calls. |
| `selection_tool.py` | Interactive vertex selection tool using screen-space nearest-vertex projection. |
| `shader_program.py` | Compile and link GLSL shader programs for OpenGL 3.3 core profile. |

### `faceforge/scanner/`

| module | purpose |
|---|---|
| `__init__.py` | Virtual scanner: cross-section imaging through the 3D body. |
| `engine.py` | Scanner engine: cross-section imaging via ray-triangle intersection. |
| `scan_plane.py` | 3D scan plane visualization in the viewport. |
| `scanner_window.py` | Scanner window: QDialog popup with cross-section image display and controls. |
| `tissue_map.py` | Tissue classification and density/intensity tables for scan modes. |

### `faceforge/scene/`

| module | purpose |
|---|---|
| `__init__.py` | Scene environment package: room, table, lamp, and scene mode control. |
| `builtin_animations.py` | Built-in animation clips for scene mode. |
| `procedural_geometry.py` | Procedural mesh builders for scene environment objects. |
| `scene_animation.py` | Keyframe animation engine for scene-mode sequences. |
| `scene_description.py` | Serializable scene description for future Blender export. |
| `scene_environment.py` | Assembles the room environment from procedural primitives. |
| `scene_mode_controller.py` | Manages scene mode state and body repositioning. |

### `faceforge/ui/`

| module | purpose |
|---|---|
| `comparison_dialog.py` | Comparison mode dialog for side-by-side anatomy views. |
| `control_panel.py` | Right control panel with QTabWidget containing 7 tabs. |
| `exam_dialog.py` | Examination dialog: the exam tier, reachable from the UI. |
| `export_dialog.py` | Export dialog for video/GIF/screenshot settings. |
| `illustration_presets.py` | Grey's Anatomy-style illustration presets. |
| `info_panel.py` | Left info panel showing active AUs and expression name. |
| `load_status.py` | Non-modal surface for asset-load failures. |
| `exercise_viewer.py` | `ExerciseViewerPanel`: gym camera views, whole-body display toggles, and the adopted exercise tab; the main window's viewer mode swaps it in for the control panel. |
| `main_window.py` | Main window: assembles all UI components around GL viewport; `set_viewer_mode()` (View menu, Ctrl+Shift+V) moves the exercise tab into the viewer panel and enters the gym. |
| `quiz_dialog.py` | Interactive anatomy quiz dialog. |
| `startup_dialog.py` | Startup dialog for choosing an initial layer configuration preset. |
| `style.py` | QSS dark theme stylesheet matching the HTML version's design. |
| `timeline_editor.py` | Custom Animation Timeline Editor dialog. |

### `faceforge/ui/tabs/`

| module | purpose |
|---|---|
| `align_tab.py` | Alignment tab: face-to-skull alignment sliders. |
| `animate_tab.py` | Animate tab: expressions grid, AU sliders, eye controls, head rotation, auto-animation toggles, speech. |
| `body_tab.py` | Body tab: pose presets, spine/joint sliders, breathing. |
| `debug_tab.py` | Debug tab: debug visualization toggles, stats, vertex selection, overrides. |
| `display_tab.py` | Display tab: render mode, camera presets, colors, labels, clip plane. |
| `exercise_tab.py` | Exercise tab: pick a demonstration, read the technique, watch the muscles work. |
| `layers_tab.py` | Layers tab: visibility toggles for all anatomical groups. |

### `faceforge/ui/widgets/`

| module | purpose |
|---|---|
| `collapsible_section.py` | Collapsible section with master toggle and individual item toggles. |
| `color_picker.py` | Color picker button + QColorDialog. |
| `expression_grid.py` | Grid of expression preset buttons. |
| `eye_color_grid.py` | Grid of eye color preset buttons with color swatches. |
| `label_overlay.py` | QPainter 2D overlay for structure labels projected from 3D. |
| `loading_overlay.py` | Semi-transparent loading overlay with progress bar. |
| `muscle_activation_list.py` | A list of muscle groups with a role tag and a live activation bar each. |
| `section_label.py` | Section label with accent underline. |
| `slider_row.py` | Label + QSlider + value display widget. |
| `timeline_canvas.py` | Timeline canvas widget for keyframe visualisation and interaction. |
| `toggle_row.py` | Label + QCheckBox toggle widget. |
| `transport_controls.py` | Animation transport controls: play/pause, stop, speed, timeline slider. |

### `tools/` (selected)

| script | purpose |
|---|---|
| `headless_loader.py` | Skeleton + joint chains + skinning without Qt/GL; `load_layer`, `register_layer`, `apply_pose` |
| `render_exercise_demo.py` | Exercise demo through the real GL renderer; `--probe` measures placement without GL |
| `author_footprints.py` | Mirror authored attachment footprints to the other side, or seed them from bone proximity (measure before keeping) |
| `export_exercise_docs.py` | Renders the catalogue to `docs/exercises.md` (`--check` for CI) |
| `glcontext.py` | Offscreen CGL context (software renderer in a sandbox) |
| `capture_golden.py`, `compare_golden.py` | Golden-image capture and diff |
| `generate_readme_images.py`, `generate_scanner_images.py` | README figures via the PIL renderers |
| `mesh_renderer.py`, `head_renderer.py` | PIL orthographic renderers with stretch/group colouring |
| `fetch_assets.py` | Verify / cache the BodyParts3D asset set |

### `tests/` layout

`anatomy` (attachments, fibre field, bone collision, bone anchors), `animation`, `app` (wiring), `body` (skinning, skinning under the
scene wrapper, ground lock, DOF axes, hand grip, shoulder-girdle rhythm, heatmap), `controllers` (handlers on a
stub `AppContext`; `fakes.py`),
`core`, `exercise` (catalogue, activation, implied stabilisers, equipment, runtime, grip lock), `export`,
`integration`, `loaders`, `rendering`, `scanner`, `session`, `tools`, `ui`
(the viewer panel headless; whole-app smoke test and the viewer mode end to end, slow).
