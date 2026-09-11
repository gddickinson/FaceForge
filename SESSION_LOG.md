# SESSION_LOG.md — FaceForge

## 2026-09-09 / 10 — Exercise demonstrations

**Goal.** Animate the anatomical figure through weight-training, conditioning
and athletic movements; show the technique, which joints move, and which
muscles work (heatmap), for teaching and physiotherapy use.

**Measured first.** Probed the real skeleton (`tools/headless_loader`): the
"abduct" DOF spun the limb about its length and "rotate" swung it sideways
(JS Y-up axes kept in a Z-up model); the arms hang off the pelvis root so
spinal flexion does not carry the shoulders. Both recorded in
`docs/exercise_animation.md`.

**Built.**
- `body/dof_ranges.py` (one DOF table) and the axis fix in
  `body_animation._apply_limbs`; joint limits widened for overhead/deep-squat.
- `body/muscle_activation.py` rewritten: the heatmap set a flag nothing read
  and never marked colours dirty, so it never showed; now external levels,
  palettes, DiGiovine bands; on-demand muscle regions are registered.
- `faceforge/exercise/`: model, muscle groups → mesh names, pose library in
  degrees with the foot-flat rule, activation model, motion description, clip
  builder, equipment + rig, runtime; 63 exercises in six catalogue modules.
- `body/ground_contact.py`: the ground lock (feet/hands re-anchored per frame).
- Gym scene + cameras, `ExerciseController`, `ExerciseTab`, simulation hooks.
- `tools/render_exercise_demo.py` (GL demo + no-GL probe),
  `tools/export_exercise_docs.py` → `docs/exercises.md`.
- Tests: `tests/exercise/*`, `tests/body/test_ground_contact.py`,
  `test_dof_ranges_and_limb_axes.py`, `test_muscle_activation_heatmap.py`,
  `test_skinning_under_scene_wrapper.py` (with a negative control),
  `tests/controllers/test_exercise_controller.py`; wiring map and the GUI
  smoke test's tab list updated. Fast tier: 1718 passed.

**Found and fixed on the way.** The heatmap never rendered (wrong flag,
colours never marked dirty); the abduction/rotation axis swap; the skinning's
correction passes ignoring the scene wrapper (muscles flew off the body in
every scene mode, not only the gym); a `merge()`-as-partial mistake in the
first catalogue draft that the placement probe caught (legs reset to neutral
under an arm bundle). Every placement number in the catalogue was checked
against the real skeleton with `tools/render_exercise_demo.py --probe`.

**Research.** Six web searches and six fetched papers grounded the squat,
deadlift, bench, pull-up, pulldown and eccentric/concentric numbers; the
delegated research agents were stopped by a session rate limit and the
search budget then ran out, so the rest is standard references, marked as
such in `docs/research/exercise_sources_2026-09.md`.

**Next.** Tune placements from the probe report and the rendered demo;
consider reparenting the shoulder girdle under T1; a "length change" heatmap
mode from the attachment system's stretch ratio; scapular DOFs for shrugs.

## 2026-09-10 (later) — Shoulder muscles torn by arm movement

**Reported.** Lats, pecs and shoulder muscles pulled away from the torso into
peaks in the pull-up and barbell back squat renders.

**Found.** The headless render path was not the app's (private, stale chain
builder without the clavicle/scapula joints; bare muscle registration with no
attachments, so the shipped footprints never resolved); latissimus dorsi's
config origin was the scapula; footprints existed for three right-side
muscles only.

**Fixed.** Shared chain builder (`coordination/joint_chains.py`) and shared
muscle registration (`demand_loaders.register_muscle_layer`); lat origin →
T7–T12 / lower ribs / hip bone, with vertebra meshes registered as attachment
bones; footprints mirrored to the left and seeded from bone proximity for 16
shoulder-girdle muscles (`tools/author_footprints.py`). Per-muscle stretch
measured before/after (table in `docs/exercise_animation.md`); every listed
muscle improved, none regressed. Fast tier 1722 passed.

## 2026-09-10 (evening) — Sagging arms, bowed lats, spikes, and bars through hands

**Reported.** At the pull-up dead hang the lats hung away from the body as if
unattached at the back; other muscles spiked; in the back squat the upper-arm
muscles sagged below the humerus; bars passed through the hands instead of
being gripped. Asked to detect, diagnose and fix all of it and re-render
every demo.

**Found, each by a measurement before the fix.**

1. *Arm sag.* `MuscleAttachmentSystem.apply_stretch_clamp` measured a
   muscle's "length" along the mesh's anterior-posterior extent and, when the
   meaningless ratio exceeded 1.35, blended the whole mesh halfway back to its
   rest position in space. At the back-squat rack pose the biceps sat a median
   7.2 units below the humerus with it, 0.4 without. Legacy Y-extent pinning
   toward a bone centroid's translation tore the non-footprinted arm muscles
   too (triceps medial head stretch p99 1.30× → 5.90×).
2. *Bowed lats.* Footprinted muscles were placed by blending the rigid images
   of their two joints; a belly vertex 40 units from the shoulder has a 40-unit
   arc as its humerus image, and half of that arc is the loop the user saw.
3. *Spikes at rest.* Bone-collision capsules were built from pivot-local bone
   vertices, so all twelve sat near the neck; a phantom humerus displaced
   ~3,000 vertices of every deep neck muscle up to 3 units at the neutral pose
   (semispinalis stretch p99 9.3× at rest; 26 of 169 muscles non-unity).
4. *Bars through hands.* Digit pivots sat at bone centroids (a phalanx
   rotating about its middle opens the joint) and 90° of curl was shared over
   four joints, so a full curl barely bent the fingers; the bar was placed at
   the wrist plus an offset rather than inside the fingers.

**Fixed.** Stretch clamp is measurement-only and legacy pinning is gone;
`anatomy/fibre_field.py` interpolates a footprinted muscle's belly
harmonically between the rigid images of its footprints (eight bind-time
solves, disk-cached; footprints trimmed to a geodesic gap and to their far
ends, isolated specks pruned), and the passes that pull toward a rigid image
(bone-offset projection, superior envelope, hull bound, balloon, collision)
are skipped for those muscles; capsules live in the bone's local frame,
follow it every frame and only resolve penetration beyond each vertex's rest
depth; digit pivots at the phalanges' proximal ends, per-joint curl maxima
(MCP 90°, PIP 100°, DIP 60°), and the equipment rig puts a bar's axis through
the centroid of the closed finger joints (`EquipmentRig.grip_point`); a body
anchored by its hands to a point (the pull-up bar) now hangs by that ring
instead of by its wrists (`body/hand_points.py`).

**Measured after.** Neutral pose: 0 of 169 muscles deviate. Rack pose:
biceps residual from the humerus median 7.2 → 0.0 units. Deformation gate on
the app path: seam p99 37.65 → 0.18, seam max 665 → 257, containment 0; bulk
p99 0.090 → 0.25 because bellies now stretch instead of tearing, thresholds
re-ratcheted from that measurement. Per-pose shoulder table in
`docs/exercise_animation.md`. Fast tier 1741 passed (new: fibre field,
collision placement/allowance, measure-only clamp, hand grip, grip ring).

**Still open.** The deltoids sit at ~3–4× p99 at a 165° dead hang: the
scapula pivot's placement moves the acromion too little, so the deltoid is
asked to lengthen where it should shorten. Proximity-seeded footprints remain
far larger than real attachments; hand-authored ones would tighten every
shoulder number further.

## 2026-09-10 (night) — Grips, the winged scapula, hands sliding

**Reported.** Back-squat bar between the fingers (hands not rotated to the
bar); bench press hands supinated; something still pointing out of the back
at the pull-up hang; hands sliding along the bar during the pull. Committed
and pushed the previous work first (582ee0f).

**Found.** The "wing" was teres major / infraspinatus / subscapularis
following a scapula rotated about a fixed centroid axis (inferior angle 7
units outside the ribcage at 165°); the lats were not involved. The wrist
composed pronation innermost, so a pronated wrist could not extend toward a
bar. Grip poses had been authored without measuring the hand frame. Nothing
held the hands' lateral position while the shoulder angles interpolated.

**Fixed.** Scapular upward rotation about the thorax's surface normal plus
clavicle elevation carrying the acromion (`_apply_girdle`, inferior angle
stays at r = 21, wing gone: 0 vertices displaced outward); pronation
outermost at the wrist; footprints for the trapezius parts and rhomboids
(config origins corrected to the spine); grips re-authored from a wrist-frame
probe (back squat 0.9°, bench 13–15°, pull-up 10° between the flexion axis
and the bar); `exercise/grip_lock.py` holds each hand's offset along the bar
by solving shoulder abduction per frame (grip width 54.5–68.7 → 55.1–55.1).
Fast tier 1748 passed; gate passing (seam p99 0.14).

**Open.** Rhomboid major 5.5× at the dead hang (the medial border glides far
from T2–T5); C7/T1 unregistered; the pull-up top's flexion axis 18° off
after the width correction.

## 2026-09-11 — Exercise viewer, whole-body display, implied stabilisers

**Asked.** A viewer, reachable from the main GUI or standalone, showing the
model in the 3D gym from any angle with a menu of exercises to play; every
muscle and the skeleton in every exercise; colour by exertion including the
muscles that stabilise or grip (a deadlift lights the hands, arms and back).

**Built.** `ui/exercise_viewer.py` + `MainWindow.set_viewer_mode` (View menu,
Ctrl+Shift+V): the control panel's exercise tab is moved into a viewer panel
with eight gym camera views (four new presets: back, two back quarters, low
front) and whole-body toggles; `faceforge/exercise_viewer.py` opens the app
straight into that mode (`--exercise ID`, `--list`). `exercise/stabilisers.py`
derives grip / carry / brace / stance / racked-bar stabilisers from the
equipment, anchor and orientation and appends them at clip-build time;
hand and foot intrinsic groups added (side-prefixed names); the exercise
controller's `all_muscles` option loads every body layer. Tests: implied
stabilisers, group expansion, controller option, the panel headless, and a
slow end-to-end viewer-mode test on the real application.

Launchers added at the project root: `start_faceforge.sh` (the full GUI) and
`start_exercise_viewer.sh` (the standalone viewer; forwards `--exercise` /
`--list`), both choosing `$FACEFORGE_PYTHON`, then the flika environment, then
`python3`, and setting `PYTHONPATH=src`.

## 2026-09-11 — Bones outside the body in the exercise viewer

**Reported.** Skeleton rendering wrong in the exercise viewer: bones outside
the model's body.

**Found.** Headless renders were fine and every world matrix and vertex
array in the running app agreed with them; the *drawn* frame did not.
Projecting the scene's own centroids onto a grab of the real window showed
the ribs drawn ~45 units below and outside their centroids (10th rib) while
every muscle sat where its matrix said. Cause: the standalone viewer entered
its mode on `LOADING_COMPLETE`, which the skeleton pipeline publishes
several stages before body animation, rib pivots, skinning and attachment
systems are wired. The exercise therefore started without body animation
(no grip lock) and the skeleton was painted — vertices uploaded to the GPU —
before `reparent_under_pivot` re-based each rib under its breathing pivot by
subtracting the centroid in place. Nothing set `needs_update`, the renderer
re-streams only flagged meshes, so each rib was drawn at pivot + original
vertices: twice its distance from the body origin.

**Fixed.** `reparent_under_pivot` flags the mesh for re-upload (any in-place
edit of `geometry.positions` after load must); `faceforge/exercise_viewer.py`
enters the mode from the load sequence's `COMPLETE` stage
(`watch_load_sequence`, deferred to the event loop). Also found in the same
grabs: the view buttons applied presets at the standing height, so a hanging
body was cut off at the hips; `SceneModeController.set_camera_target_override`
now carries the running exercise's `camera_target` (set at start, cleared at
stop) into every preset. Verified on the real window: ribs and pelvis on
their projected centroids, full body framed in the front view at mid-rep.
Tests: pivot re-basing flag, viewer entry timing, camera target override
(controller and scene controller).

**Per-frame cost.** With every muscle loaded (317 muscles, 7.9 M vertices)
one `Simulation.step` cost 9.7 s. Profiled and cut, output unchanged
(deformation gate re-run): the bone collision pass now measures only capsules
whose box overlaps the mesh, and only the vertices in that box (3372
distance passes → 220; 3.4 s → 0.75 s); face normals accumulate with
`np.bincount` instead of `np.add.at` (0.92 → 0.31 s); a muscle none of whose
driving joints moved since its last frame is skipped (per-binding driver
deltas, 1e-9); the skinning signature no longer includes the wrapper (the
output is body-frame, and the ground lock moved the wrapper every frame).
Full-body recompute 9.7 → 6.3 s; a pull-up frame 9.7 → 1.25 s. A grouped
per-joint matmul replacing the `(V, 4, 4)` gather measured slower from two
joints up and was not kept (`body/skinning_ops.py` records the numbers).
Fast tier 1774 passed.

**Open.** The viewer is still ~1 frame/s with everything loaded; the floor
is memory bandwidth over 4–8 M vertices per pass. A display level of detail
(decimated muscle meshes with the vertex-indexed data remapped by position)
is the next step, and a project of its own.
