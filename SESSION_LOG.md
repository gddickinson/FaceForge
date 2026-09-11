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

**Duplicate mesh names.** The hand and foot configs both named
"R/L Abductor Digiti Minimi", "Flexor Digiti Minimi Brevis", "Opponens
Digiti Minimi" and "Dorsal Interossei". The heatmap registry and the
activation track are keyed by name, so the foot's registration replaced the
hand's: a deadlift's grip never coloured those hand muscles and the foot's
took the hand's level. The foot entries now carry "(Foot)" (config and
`foot_intrinsics`), the registry warns on a second mesh under one name, and
a test asserts name uniqueness across every muscle config.

**Open.** The viewer is still ~1 frame/s with everything loaded; the floor
is memory bandwidth over 4–8 M vertices per pass. A display level of detail
(decimated muscle meshes with the vertex-indexed data remapped by position)
is the next step, and a project of its own.

## 2026-09-11 — Every exercise through the viewer

**Asked.** Look at the viewer, fix what comes up, and test that every
exercise renders and plays.

**Sweep.** A script drove the real viewer window through all 63 catalogue
entries (four points per clip, three-quarter and side grabs, per-frame
measurements): no NaN, no muscle centroid outside the skeleton's box, the
ground lock's residual below a unit everywhere, every definition's
equipment present, the grip lock armed for the hanging exercises. The
contact sheets confirmed each exercise plays its movement. The viewer panel
was exercised too: skin and whole-body toggles, leaving the mode (0.1 s),
re-entering (0.4 s), starting another exercise after re-entry, stopping.

**Fixed from the frames.** Goblet squat: the kettlebell sat inside the
chest — the hands were at the shoulders and the bell hung 28 below them.
Measured hand placement on the rig (closed-finger ring centres) and
re-authored the hold: hands 14 apart, 19 forward of and 19 above the
mid-sternum, the bell's origin 4 above the hands via a new per-spec
`EquipmentSpec.hang` override (`eq(..., hang=)`, `runtime.equipment_tuning`).
Front squat: the bar was 20 above the shoulders with the hands folded high;
the rack is now flex 65 / rotate −60 / elbow 145 with the bar hung 13 below
the hands, on the front deltoids. Bench press, incline press, lying triceps
extension, chest fly: the feet rested on the bench top; with the pelvis on a
58-high pad the thighs must slope 10 deg below the trunk and the knees bend
70 for the soles to reach the floor (hip −10, knee 70, abduct 22, measured
foot height 0–5). The incline press pitches the whole body 30 deg about the
hips, which carries the legs with it, so it has its own legs (hip 16, knee
70). Box jump verified standing on the box at the Stand phase (lowest bone
65.8 over a 60-high box). Hand and foot muscle names de-duplicated earlier
the same day.

## 2026-09-11 — The sex morph: skeleton, muscles and skin

**Reported.** The morph between sexes "doesn't work well": skeletal changes
leave gaps between bones and are not realistic, no change in musculature
works, and the skin flattens, especially in the arms.

**Diagnosed.** All three reproduced, and measured.

* *Gaps.* Every bone scaled about its own centroid, so the knee opened 0.07 →
  3.55 units at gender 1, the elbow 0.59 → 3.89, the patella 0.36 → 1.51.
  Scaling about the centroid also leaves the centroid where it is: the median
  bone-centroid displacement across the whole change was **0.00**, so the
  skeleton never changed proportion at all.
* *No musculature.* The release path scaled the bones and then *re-bound* the
  skinning, which makes the new skeleton the rest pose and sets every delta to
  the identity. Measured through the full path, muscle centroids, lengths,
  radii and every skin cross-section were bit-identical before and after.
* *Flattening.* The load-time warp projected every surface vertex onto the
  closest point of the BP3D skin. 1235 triangles ended below a tenth of their
  area, 3034 hand edges and 727 forearm edges below half their length, worst
  0.4 %: the hands and feet were flat blades.

**Built.** `body/skeleton_morph.py` scales the skeleton as an articulated
hierarchy (bones about the joint they hang from, joints moved to the end of
the scaled bone), with `skeleton_joints.py` shutting the one articulation the
hierarchy cannot (acromioclavicular) from a measured contact patch.
`skeleton_field.py` turns the joint movements into a thin-plate spline warp,
sampled on a lattice; `soft_tissue_morph.py` composes that warp with
`muscle_morph.py` (bellies thinned perpendicular to their own axis, tapered to
nothing at the attachments) and `skin_morph.py` (the female-minus-male
soft-tissue field, measured from the surface pair, size change and tangential
component removed). `surface_fit.py` and `surface_projection.py` carry the
warp machinery that `gender_morph.py` had grown to 1253 lines around.

**Two latent bugs found on the way.** `_rest_inv_cache` is keyed by joint index
and outlived the joints it described, so after any rebind the hull bound
clamped 554875 skin vertices by up to 10.4 units. And `build_skin_joints`
captured rest matrices in world space while `update` reads them
wrapper-cancelled, so rebuilding the joints under the gym wrapper dropped the
whole soft tissue on the floor, rotated 90 degrees. Both fixed, both with
regression tests.

**Measured after.** Every joint clearance within 0.12 of its male value;
gender 0 restores bit-for-bit; 33 of 2.37 M skin edges outside the band, 32 of
them in the toes. Biacromial F/M 0.89, femur 0.91, skin waist/hip 0.90 → 0.80,
biceps belly radius 0.58, rectus femoris 0.77 — all within the published
ranges. The anthropometry in `gender_dimorphism.json` was corrected: the female
pelvis keeps nearly the male's absolute breadth rather than being scaled up,
which had given a bi-iliac/biacromial ratio of 1.01 against a published
0.80-0.86.

A release costs 10 s, of which 5.9 s is the frame that follows. Replacing the
full re-binding with a rest re-snapshot (the binding does not change when the
body changes size) took 85 s off it. Fast tier green; `docs/sex_morph.md`
records the measurements.

## 2026-09-11 (later) — The body-surface mesh: flattening, and fitting the skull

**Reported.** The skin mesh layer is still deformed strangely at the back of
the head, the forearms and the feet, in both sexes; the skeleton should fit
better inside it, and the skull should be resized to match.

**Found.** The body-surface mesh is warped onto the skeleton at load by a
piecewise Z-remap blended against two arm rotations. Blending a rotation
against a translation is not a rigid motion: the forearm's depth fell from
24.7 to 17.4 while its width rose from 21.5 to 28.4, the foot lost a third of
its length, and the occiput was shaved flat. The mesh's "wrist" landmark was
the lowest tenth of the arm's vertices — the fingertips — so the forearm was
sheared to match, and there was no hand landmark at all, which left the
skeleton's fingertips 15.6 units outside the surface.

**Built.** `body/surface_register.py` registers the mesh limb by limb: each
segment rotated, scaled and moved onto its bone, the trunk and head matched to
the reference body level by level, the correspondences interpolated by a
spline, and the result held inside a band around the mesh's own edge lengths.
`body/surface_landmarks.py` finds the wrist and ankle at the narrowest station
of a limb and gives the hand a landmark from the middle finger's distal
phalanx. `fit_head_to_skull` moves and grows the head, per axis and never
below 1, by the least that clears the skull, blended to nothing by the
shoulders; the pipeline samples the loaded skeleton and hands it over.

**Measured.** The occiput is rounded again, the forearm keeps its
cross-section and the foot its length, in both sexes. Bone vertices outside
the surface: 53.5 % as shipped, 70.2 % if the mesh is only placed and not
deformed, 51.2 % now, with the 95th-percentile protrusion at 12.7 against
19.2; the skull alone is 16 % outside with a worst case of 2.1.

**Rejected, with the measurements that rejected them.** A spline through the
fourteen landmarks alone (a 2-unit cube in the forearm came out 1.36 x 1.64 x
2.00); blending the per-limb similarities by distance weight (59.9 % outside);
pairing the trunk's outline rather than its centre (imports the reference
cadaver's bulbous occiput and heavy abdomen); and inflating the surface
wherever any bone pokes through (self-intersecting spikes on a 10 500-vertex
mesh). The residual misfit is honest: the two bodies differ, and every method
that closes the gap further damages the mesh.

## 2026-09-11 (later still) — Neck muscles that distorted during exercise

**Reported.** The neck muscles distort during exercise; the other muscles
render correctly.

**Found.** Four defects, all in the neck muscles' own deformation path, which
is separate from the soft-tissue skinning.

1. *The wrong frame.* The rest body anchors are snapshotted at load, before
   scene mode exists; the current ones were read as world positions, which in
   the gym include the `scene_wrapper` (Rx(−90°) at Y = 203). One frame into
   a bodyweight squat the thoracic anchor read (0.225, 192.088, −5.661)
   against a rest of (0.225, 5.661, −10.912). Every neck muscle was dragged
   153 units with a 99th-percentile edge stretch of 63×, at the neutral pose,
   purely from entering the room. The other muscles were fine because the
   skinning cancels the wrapper.
2. *Attachment names that matched nothing.* Ten muscles named
   `"Thoracic Vertebra T1"` / `"T3"` where the scene nodes are `T1` / `T3`,
   and T1 was never registered at all — it hangs off the cervical pivot
   chain, and the registry walked only the thoracic and lumbar groups.
   Pinning silently did nothing for all ten.
3. *Region over bone.* Every muscle followed its coarse regional anchor even
   when it named the bone it attaches to. At full thoracic flexion the top
   thoracic pivot travels 4.58 units while T1 does not move at all, so 14 of
   the 38 muscles — the six suboccipitals among them, whose origins are on C1
   and C2 — were dragged 3.76 units by a thorax they are not attached to.
4. *A zero delta read as "nothing to do".* `update` skipped its work when the
   head quaternion was unchanged and the body delta was zero. Returning to
   rest *is* a zero delta, so the frame that should have straightened the
   neck was the frame that was skipped, and a neck bent by a sit-up stayed
   bent.

**Built.** `coordination/body_anchors.py` reads the anchors in the body frame
and owns the wrapper cancellation; `Simulation.frame_cancel()` settles it once
at step 9.5, before the neck muscles, the neck pinning and the platysma read a
pivot, instead of at step 12 where the skinning used to set it.
`anatomy/neck_body_follow.py` prefers the muscle's own attachment-bone
displacement over the regional average and holds the pinning pass;
`anatomy/neck_fibre_strain.py` takes the volume-preserving strain, which
brings `neck_muscles.py` back under the file-size limit. The loading pipeline
registers the cervical vertebrae, the twelve suboccipitals gained their real
C1/C2 attachments, and the early exit now compares the deltas the current
vertex buffers were built from.

**Measured.** `tools/neck_deformation_quality.py`, eight poses. In the gym:
worst 99th-percentile edge stretch 63.119 → 1.712, worst displacement 152.931
→ 3.515, and the gym rows are now identical to the clinical rows pose for
pose. At full thoracic flexion: 14 muscles displaced → 4, worst stretch 2.522
→ 1.712. The soft-tissue gate is unchanged (seam p99 0.163, bulk p99 0.2226).
Figures: `results/neck_frame_diagnosis.png`,
`results/neck_attachment_drag.png`.

**Left open, and why.** The cervical spine and skull hang off `bodyRoot`
rather than off the top of the thoracic chain, so thoracic flexion moves T3
under a stationary head and the four longus colli that originate there really
are stretched (1.712 at full flexion, about 1.1 at a sit-up's 30°). Fixing
that means reparenting the head onto the spine, which moves head rotation,
FACS, face alignment and the camera framing with it. Head rotation alone
still reaches 2.087 on the infrahyoids at full pitch; that path was not part
of this report and was not touched.

## 2026-09-11 (last) — Muscles loading onto the floor behind the skeleton

**Reported.** Many muscles load 90 degrees offset from the skeleton and lie on
the floor behind it in the exercise module.

**Found.** A one-line aliasing bug I introduced in 461ae0d.
`SceneNode.update_world_matrix` rewrites world matrices *in place*, and
`np.asarray` on an array that is already float64 returns the same object, so
`SoftTissueSkinning._joint_world` handed back a live view whenever there was
no wrapper to cancel — which is exactly the case at load time. Every joint's
*rest* matrix was therefore an alias of its node's world matrix, and all 152
of them silently became the current world matrices the moment the body stood
up in the gym (joints read 219 units from where they were snapshotted). The
exercise module enters the gym before it loads its muscle regions, so those
muscles were bound against world-space joint positions, `_joint_delta` cached
the inverse of a world matrix as a rest inverse, and the attachment pinning
carried each muscle to that image of its rest pose. The previous code was
`node.world_matrix.copy()`; the wrapper-cancel fix replaced it with a call
that only copies when a wrapper is present.

**Built.** `_joint_world` always returns a fresh array, documented with the
measurement; `build_skin_joints` copies explicitly where it stores the result.

**Measured.** Placement across four muscle regions, gym-entered-first against
muscles-loaded-first: before, all 138 meshes differed, median 160.6 units and
up to 203.0; after, 0.0000 for every mesh. Pronator Quadratus R held its
centroid of (37.7, -3.4, -79.4) through the skinning instead of jumping to
(37.7, 79.4, -206.4). Figure: `results/muscle_load_order_offset.png`, the arm
and thigh muscles flat on the floor behind the standing skeleton. The
soft-tissue gate is unchanged (seam p99 0.163, bulk
p99 0.2226). Two tests in `tests/body/test_skinning_under_scene_wrapper.py`
pin it and both fail on the old code. The two
`tests/ui/test_exercise_viewer_mode.py` failures carried in the previous entry
were partly this bug: they failed in both orders before and now pass when that
file runs first. They still fail when `tests/tools/test_deformation_quality.py`
runs before them, which is a test-order dependence in that fixture's fixed
one-second settle, not a fault in the application.

**Still open.** `tests/tools/test_deformation_quality.py::test_the_gate_is_sensitive_to_a_broken_engine`
fails: re-enabling the neighbour clamp with the hull bound and containment
corrections off no longer produces any containment drift, so the gate's
negative control passes a deliberately broken engine. It is not the
per-binding skip — measured with the skip disabled, containment is still
0.000 — so the control needs a mechanism that still misbehaves, which is its
own piece of work.
