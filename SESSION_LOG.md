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

## 2026-09-11 (last, later) — Skin that tore during exercises

**Reported.** The skin layer deforms during exercises, with individual
vertices either left behind or attached to the movements of different body
parts.

**Found.** The skin is intact at rest, in the clinical view and in the gym at
the neutral pose, and tears the instant any joint rotates; exercises are only
where it shows, because their poses are the most extreme. Containment is
0.000 throughout, so nothing is ever left behind by a joint that did not
move — "left behind" is the far side of a torn edge.

Decomposing the pipeline at a deadlift-style hip hinge, by edges stretched
past twice their rest length: rigid binding 9,900; two-joint linear blend
18,850; four-influence linear blend 53,485; four-influence dual-quaternion
blend 55,568; engine output after every correction pass 56,125. So neither
the dual-quaternion blend nor any correction was responsible — the
corrections moved the offending vertices by 0.000 units — and the count was
not the issue either, since three influences tore 58,841 edges, more than
four. The influence cutoff was rank-based: each vertex took its four nearest
bone segments cut off at the distance to the fifth, which is smooth but not
local, so a thigh vertex carried real weight on the ankle.

**Built.** `SoftTissueSkinning.INFLUENCE_CUTOFF_BAND`: compact support
measured from the vertex's nearest segment, three model units wide. A
departing segment's weight still reaches zero smoothly, but the set stays
local. The band is additive rather than a multiple of the nearest distance,
because a multiple collapses where the skin lies on the bone — with a 1.5x
ratio the worst edge went from 253x to 1982x. `SKIN_CHAIN_Z_MARGIN` and
`SKIN_SPATIAL_LIMIT` are named constants now so they can be measured, and
`tools/skin_deformation_quality.py` is the gate.

**Measured.** Over four poses, worst-pose 99th-percentile edge stretch and
total torn edges: rank-based 8.073 / 185,170; band 3.0 as shipped 1.954 /
70,092. At the hip hinge, vertices with a torn incident edge fall from 39,672
to 12,967 of 791,729 (5.01% to 1.64%). Through a real deadlift the torn
edges fall from 97,478 to 42,002 and the 99th percentile from 8.728 to 4.193;
the gain is smaller than on the gate's poses because the deadlift setup
combines deep hip flexion with knee flexion and the arm reach. The muscle
gate is byte-identical
(seam p99 0.163, bulk p99 0.2226) because muscles do not take the
multi-influence path. Figure: `results/skin_tearing.png`.

**Rejected, with the measurements that rejected them.** `DIFFUSE_WEIGHTS`,
the existing heat-diffusion pass, halves the seam tail but raises the bulk
tail 25-32%, torn edges 43% and the worst edge from 253x to 3179x, because it
rebuilds the influence set from the top four of a diffused field and so
destroys the compact support. Two influences instead of four fixes the legs
but loses the axilla (seam p99 70.2 against 56.1). Tightening the spatial
guard from 25 to 12 is worse than anything: seam p99 74/119/327 and a worst
edge of 4225x, because a harder eligibility cut adds partition boundaries.

**Still open.** The residual concentrates where the skin genuinely folds: the
hip crease, and the axilla at full abduction, where the worst edge is 439x.
Of the 998 edges past 50x there, 469 are lateral abdomen skin whose nearest
bone segment really is the arm hanging beside it — a limit of nearest-bone
binding rather than of the cutoff.

## 2026-09-11 (last, later still) — Torso skin that moved with the arms

**Reported.** When the arms move, pixels from the torso are incorrectly moved.
Asked whether it can be solved the way the muscle layer was, or by proximity
of skin to muscles.

**Found.** Holding the trunk still and abducting both arms: the lower trunk
and the chest do not move at all. The back does — 261 vertices in the midline
strip over the thoracic spinous processes, by up to 10.4 units, plus the
paraspinal and scapular region. The paraspinal motion is correct, and the
project's own muscle data says so: `MUSCLE_CHAIN_OVERRIDES` gives trapezius,
rhomboids and latissimus dorsi the arm chain, so skin over them follows the
shoulder girdle. So binding skin by proximity to muscle, as asked, would
endorse most of what was being reported rather than remove it.

The midline strip is a genuine fault. Those vertices had `clavicle_R` as
primary even though the spine is nearer by Euclidean distance (thoracic_1
8.78, clavicle_R 12.13, rib_1 13.52). The chain ranking is geodesic, and the
geodesic fields are seeded by Euclidean radius, which makes seeding a contest
between *superficial* bones: the clavicle and scapula are subcutaneous, the
vertebral bodies are not, so skin 8.78 units from its own vertebra was never a
spine seed and measured its distance to the spine the long way round.

**Built.** `SEED_FROM_OWNED_SKIN`: each chain also seeds from the skin whose
nearest bone segment belongs to it. Measured, every seed this adds falls on
skin no bone reaches within the radius, so it is exactly the deep-tissue skin
the old rule never saw. `skinning_cache.CACHE_VERSION` moved to 5 because the
rule changed rather than a number, and a stale entry had already served the
old seeding through one whole measurement.

**Measured.** Midline back skin under full abduction: 261 vertices moving up
to 10.4 units, now zero moving at all. The deep squat's worst edge halves,
212.80 to 100.57. The cost is 10% more moderately torn edges over the four
gate poses (70,092 to 77,434) and a quarter more seam stretch in the axilla
(56.054 to 69.974), which was already the worst region. Figure:
`results/skin_arm_follow.png`.

**Rejected, with the measurement.** Seeding from owned skin *instead of* the
radius, rather than as well as it, gives the same numbers — so the overlap the
radius provides is not what is buying the soft boundary. Restricting the new
seeds to vertices the radius rule leaves unseeded also changes nothing, for
the same reason: benefit and cost are the same seeds.

## 2026-09-11 (end) — The axilla, and 528 pieces of skin

**Asked.** Improve on the remaining skin problems.

**Found.** Every surviving worst edge had the same shape: two vertices a tenth
of a unit apart, one on the trunk and one flying off with the arm. The worst
in the mesh joined a pair on the left flank; one took the lumbar spine, the
other took elbow_L 0.32, shoulder_L 0.27 and wrist_L 0.22 and moved 60 units.
In the rest pose the arms hang against the trunk, so the forearm is ~4 units
from that skin and the lumbar spine ~19: straight-line distance treats the gap
between arm and waist as tissue.

Two causes, both about the surface rather than the bones. Skin in the
ambiguous band was seeding chains, so the geodesic field it was meant to be
corrected by was built from its own mistake. And the skin is not one surface:
791,729 vertices in 528 connected components, 454 under 100 vertices, 28,201
off the main one. Dijkstra never reaches an island, its geodesic distance is
infinite, and the solve falls back to the Euclidean measurement the geodesic
pass exists to overrule. The three worst edges at full abduction each joined
an island vertex to its intact neighbour.

**Built.** `SEED_CONFIDENCE_MARGIN` (1.5): a vertex seeds a chain only when
that chain is clearly nearest; ambiguous skin seeds nothing and the fields
reach it by propagation. `GEODESIC_BRIDGE` (5.0) with `BRIDGE_CONTACTS` (8):
island patches are joined into the Dijkstra graph at their few closest
contacts, never into `edge_pairs`, which the stretch metrics and edge
relaxation read as real topology. Joining at contacts rather than vertex by
vertex matters — gluing every island vertex to whatever is nearest attached a
patch on the lateral chest across the armpit to the arm.

**Measured.** Over the four gate poses, against the state committed before
this: torn edges 77,434 to 62,280; the axilla's seam tail 69.974 to 48.066;
the worst edge in the mesh 570.25 to 301.64; the hip hinge's worst edge 152.45
to 110.11. At the hip hinge, vertices with a torn incident edge are 11,535 of
791,729 (1.46%), against 5.01% before any of this session's skin work.
Containment stays 0.000 and the midline back skin still does not move when the
arms do. The margin was bracketed at 1.0/1.25/1.5/2.0/3.0 and the contact
count at 3 and 8, which agree to four significant figures.

**Still open.** The residual is where the skin genuinely folds: the hip crease
and buttock at deep flexion, and the axilla at full abduction, worst edge
301x. Linear and dual-quaternion skinning cannot represent a fold, so closing
that needs a different deformation model rather than a better binding.

## 2026-09-11 (end, later) — Confident seeding, bridged islands, and the spikes

**Asked.** Improve on the remaining skin problems; then, separately, whether I
can see the spikes of skin drawn off the front of the torso when the arms come
up to shoulder height.

**Built and measured.** Two further binding rules, both net wins on the gate:
`SEED_CONFIDENCE_MARGIN` (1.5), so a vertex seeds a chain only when that chain
is clearly the nearest and ambiguous skin seeds nothing; and `GEODESIC_BRIDGE`
(5.0) with `BRIDGE_CONTACTS` (8), which joins the skin's 528 disconnected
patches into the Dijkstra graph at their few closest contacts. Over the four
gate poses: torn edges 77,434 to 62,280, the worst edge at shoulder height
570.25 to 301.64, the axilla's seam tail 69.974 to 48.066, the hip hinge's
worst edge 152.45 to 110.11. The margin was bracketed at five values and the
contact count at two.

**The spikes, seen and diagnosed.** The gate now counts them directly, since
edge stretch does not: 676 at shoulder height, 225 at the hip hinge, none at
rest. They are a rim of trunk skin at the lateral silhouette, half a unit
wide, bound almost entirely to the arm; the worst holds elbow 0.39, shoulder
0.31, wrist 0.30 and travels 74 units while its neighbours travel 25. Below
the rib cage there is no trunk bone in reach — the ribs stop at z = -51, and
for a vertex at z = -67.8 the chain Z margin and the spatial limit leave 22 of
152 segments eligible, the nearest being the elbow at hybrid 36.6 against the
lumbar spine's 48.3. The spine loses geodesically, not by straight line,
because flank skin lying against the hanging forearm seeds the arm chain.

**Rejected, with the measurements.** Bootstrapping ownership from bone in
contact with skin (`SEED_CONTACT_RADIUS`, left in place, disabled) improves
every seam tail but raises torn edges 62,280 to 64,317 and the worst edge
301.64 to 469.48, and only moves the spikes from 88 to 70 above 10 units;
unioning the radius seeds back in is worse still at 68,309. A 2.5-unit contact
radius behaves the same. The island bridge is not implicated: with it off the
spike count is 549 against 552.

**Still open.** The spikes. The abdominal wall is what fills the space below
the ribs, so binding skin by proximity to muscle — the obliques are chained to
spine and ribs — is the shape of the answer, and nearest-bone binding is not.
Figure: `results/skin_arm_spikes.png`.

## 2026-09-11 (end, last) — Reach, flesh, and the gate that had gone quiet

**Asked.** Work through every suggestion in order, measuring and visualising
the defects before and after.

**Suggestion 1, by a different route.** The premise was wrong and the
measurement said so: the spikes were not short of a trunk bone. For 482 of the
700, a trunk segment was already nearer than any arm segment — a median 12.4
units against 14.6 — and had been masked out. A chain's spatial reach is
proportional to its size, and size was measured as vertical extent, so the rib
cage (41 units tall) got 10.35 floored to 12 while the arm chain (80 tall) got
20. The floor is 16 now, bracketed at 12/14/16/18/24. Measuring size by
bounding-box diagonal was tried and is redundant once the floor is right.

**Suggestion 2, built.** `body/muscle_field.py` holds the distance to each
body part's flesh, sampled from the muscle meshes by
`tools/build_muscle_field.py` into `assets/config/muscle_field.npz` (1.2 MB),
and the skin binding adds it to the bone distance. The premise was measured
first: for 78% of the surviving spikes the nearest muscle is a trunk muscle
while the nearest bone says arm. Weight bracketed at 0/0.6/1.0/2.0; 1.0 is
shipped, weighting flesh and bone equally. The field's digest is a public
scalar on the skinning, so the binding cache keys on it and a rebuilt field
cannot be served a stale binding.

**Measured, over the four gate poses.** Torn edges 62,280 to 55,971; spikes at
shoulder height 676 to 291; the axilla's seam tail 48.066 to 17.835; the worst
edge 301.64 to 203.89. Containment stays 0.000. Against where the skin work
started: 185,170 torn edges and a worst edge of 537.

**Visualised.** `tools/skin_defect_views.py` draws the skin from three
viewpoints coloured by stretch, spike or arm-weight, and takes a saved
baseline to draw before and after together. Figures:
`results/skin_spike_views.png`, `skin_stretch_views.png`,
`skin_armweight_views.png`, `skin_hiphinge_views.png`.

**Housekeeping.** The deformation gate's negative control was passing a
deliberately broken engine; the old mechanism (the neighbour clamp) no longer
breaks containment, nor does the per-binding skip, `CONTAIN_CORRECTIONS` or
`USE_BONE_OFFSET_PROJECTION`. Opting every muscle into the soft-body path does
— 6.000 units of drift on Triceps Long R — because the physics pass relaxes
edges and so couples vertices. The control uses that now and is live again.
`drain_deferred_startup` takes an `until` predicate, so a test waits for what
it needs rather than for a fixed settle.

**Still open.** The two exercise-viewer tests remain order-dependent, and a
second app context in one process is NOT the cause — measured, a second
context loads a demand layer perfectly well. The wait is capped at 60 seconds
now so the suite fails fast instead of pumping for five minutes. Suggestion 3,
solving the binding in a separated pose, was not built: the muscle field
addresses the same root cause — a limb lying against the trunk — more cheaply
and is measured to work, so the case for a much larger change is no longer
made. Skin asset repair is also open: 528 connected components and
inconsistent triangle winding, the latter of which blocks the inside-outside
test that would settle whether a bone is under the skin or across a gap.

## 2026-09-12 — Proof in pixels

**Asked.** Show the proof that the changes improved the rendering.

**Found.** Everything measured so far was arithmetic on vertex positions.
`tools/render_skin_proof.py` renders the app's own scene through `Session` --
the same GL renderer (OpenGL 4.1 on this machine), framebuffer and
blank-frame guard as the headless CLI -- with engine overrides so an earlier
state can be drawn from the same working tree.

**Measured, arms at shoulder height.** Stray skin removed: 24,988 pixels from
the front (16.2% of the lit area), 23,835 from the back, 18,075 from the
three-quarter. The side view is unchanged at 78 pixels, because the sheets of
stretched triangles are edge-on there.

**What the pixels correct.** The spikes are visibly reduced and visibly still
there. The count fell 676 to 291, but each spike vertex drags a fan of
triangles, so the count understates the screen area: the render shows two
dense wings from armpit to hip before and thinner wings after, not a clean
body. And at the hip hinge, where torn edges fell 22,638 to 8,817, the render
barely changes — 81 pixels of 110,000 from the front — because that tearing
is inside the silhouette and shows as shading, not as stray geometry.

**Control.** At the neutral pose, before and after are pixel-identical: 0 of
990,000 differing pixels, front and side. The changes touch deformation and
nothing else.

Figures: `results/skin_render_proof.png`, `results/skin_render_proof_hip.png`,
and the raw frames in `results/skin_render/`.

## 2026-09-12 (later) — Four more tries, judged by the renderer

**Asked.** Keep going, using the rendering to assess the changes.

**Kept: the inward test.** A chain may seed the geodesic field only where its
bone lies on the same side of the skin as the flesh that skin sits on. This is
the only thing that separates the flank from the forearm hanging beside it:
both bones are close and both have flesh close, so no distance decides it, and
direction does. The inward direction comes from the muscle field, not the mesh
normals, because this asset's winding is inconsistent and half its normals
point the wrong way. Torn edges 55,971 to 55,445 and spikes 669 to 587, and in
pixels 5,171 more stray skin off the front and 5,831 off the back.

**Rejected: re-binding in a separated pose.** The standard remedy, and it
fails for a reason worth keeping. The skin must be deformed into the separated
pose before it can be re-solved there, and the only thing available to deform
it is the rest-pose binding whose mistakes are the problem: round one drags
the flank out along the arm, round two finds those vertices beside the arm and
binds them harder. Spikes 291 to 1,237, seam tail 17.8 to 504.0. Disabling the
muscle field for the second solve changed nothing, so the rest-pose field was
not the cause.

**Rejected: discounting influences whose flesh is far.** Improves every seam
tail (17.835 to 11.039 at shoulder height) and moves the picture by 167 pixels
of 111,614. The vertices drawn into wings hold all four influences on arm
joints, so there is no trunk share to shift weight toward; the set has to
change, not the shares. That diagnosis is what led to the inward test.

**Rejected: flesh granting chain eligibility.** Inert on top of the inward
test, 587 pixels off the front and 334 back onto the three-quarter. With the
reach floor at 16 the trunk chain is already eligible where it needs to be.

**Where the rendering is.** At shoulder height, stray skin removed against the
state before any of this: 29,128 pixels from the front (20.1%), 28,978 from
the back (20.2%), 20,671 from the three-quarter (16.1%). The neutral-pose
control is pixel-identical throughout, 0 of 990,000 in three views. The wings
are thinner and still there.

**Next honest step for them.** An authored rest pose with the limbs
separated. Every fix that tries to manufacture one from the current binding
inherits the binding's mistake, which is what the separated-pose experiment
demonstrated.

## 2026-09-12 (last) — Chasing the wings to the asset

**Asked.** Keep going until the wings are removed.

**Kept, both judged on pixels.** Eligibility is now tested across the surface
rather than in a straight line: the limit and the ranking were measuring
different things and disagreed exactly where a limb lies against the trunk.
On a flank vertex the ribs were 17.27 away in a straight line and 17.78 across
the skin, the arm 17.16 and 34.29 — the ranking had it right, the ribs were
1.27 over their limit and masked, the arm kept. Limits scale by 1.9, bracketed
at five values; 6,813 pixels off the front. And a chain now seeds only skin
sitting on its own flesh, which the direction test could not settle: of the
123 vertices still drawn into flaps, 121 sat on trunk flesh 1.33 units away
with arm flesh 4.49 away. Worst edge at shoulder height 213.66 to 59.69, torn
edges 8,021 to 4,958.

**Rejected, with the measurements.** A wider influence band: 60% more torn
edges, no gain at the shoulder. A proportional support: the deep squat goes
from 22,505 torn edges to 53,020. A heavier flesh weight: spikes double.
Grouping the glenohumeral muscles with the arm: worst edge 59.69 to 184.79 for
135 pixels of difference. A flat price on crossing body parts: prices out the
real transition at the shoulder, worst edge 16,987. Cutting crossings below
the shoulder: catches the hip, wrist and neck too, spikes 398 to 1,131.

**Not removed, and the reason is in the asset.** The arm and chest are two
sheets facing each other across air below the armpit, and the surface path
between them should run up to the rim and back — a median 36.74 units against
a straight line of 8.47. For 1,443 lateral-chest vertices it does not: the
asset has the sheets touching, the shortest path being 0.11 units. Through
those welds the arm's field reaches the chest whatever the binding does.
Telling a weld from a genuine boundary needs to see the surface fold back on
itself, and the winding is too inconsistent to read the sign of a fold.

**Where it got to.** Rendered stray skin at shoulder height is 28.7% smaller
than at the start of this work (front 133,270 to 95,038 pixels, back 132,235
to 94,308), and what is left is barely stretched — the worst edge went 578 to
59.7 — so it is a coherent sheet rather than a fan of torn triangles. The
neutral-pose control stays pixel-identical. Removing the rest wants the asset
repaired: consistent winding, then cut the welds. That is the same repair the
surface fitting has been wanting.

## 2026-09-12 (after) — The skin load, without touching what it produces

**Reported.** The skin layer now takes a long time to load.

**Found.** True for the first load on a machine; every load after it reads the
binding from disk in 0.4 s. The first solve had gone to 92 s, of which 41.4 s
was the muscle field: three parts of the binding -- the term added to the bone
distance, the inward-direction test and the own-flesh test -- each walked
every body part themselves, 27 queries over 791,729 vertices where 9 will do.

**Built.** `_flesh_for` computes the per-body-part distances, the body part
each vertex sits on and the inward direction once per solve, and the three
consumers read it. The cache lives for one solve and is cleared around it.

**Measured.** First solve 92.0 s to 65.8 s; field queries 27 to 9 and 41.4 s
to 13.5 s. The output is unchanged and was checked both ways: the four gate
poses report identical numbers to every decimal, and the rendering is
pixel-identical in all four viewpoints, 0 of 990,000. Warm load stays 0.4 s.

**Left alone.** The geodesic pass, 43.4 s of Dijkstra over 2.4 M edges once
per chain. It is what the binding fix rests on, and the user's instruction was
not to trade rendering for speed.

## 2026-09-12 (end) — The body-surface meshes go back to as authored

**Asked.** Reinstate the original male and female skin meshes; the current
ones distort too much.

**Found, and it is visible rather than numerical.**
`results/morph_surface_proof.png` renders both meshes both ways from three
viewpoints through the app's own GL renderer. Warped onto the skeleton, the
hands splay, the face sinks and puckers, the feet twist and the torso loses
its symmetry. As authored, they are clean figures.

Worth being exact about what the warp was doing: it is computed from the male
mesh and applied to both, so it never distorted the *morph* — the difference
between the two ends is the authored one either way. What it distorted was the
base both ends sit on. It also inflated the surface by about 18%, median edge
length 1.336 against 1.131.

**Built.** `gender_morph.WARP_SURFACE_TO_SKELETON`, off. The meshes are scaled
and placed into the BP3D frame and otherwise untouched.

**Measured, and the cost is real.** Bone points to the nearest surface vertex
go from a median of 2.60 to 5.11, 95th percentile 10.02 to 15.33, worst 16.53
to 26.57. The skeleton sits about twice as far inside the skin and pokes
through in more places. That is the trade, taken deliberately.

## 2026-09-12 (later) — Fit the skeleton to the body mesh, not the mesh to the skeleton

**Asked.** A GUI option that fits the skeleton *into* the body-surface mesh —
moving and deforming the bones so they sit well inside it — applying to the
body-surface meshes, not to the skin layer that came with the skeleton.

**Measured first.** 65.7% of the male skeleton's vertices lie outside the
surface, a median of 2.42 units out and 11.38 at the 95th percentile; the
cranium is outside along its whole length (max 27.7), the scapula and humerus
by 12 to 19, and every bone of both feet. Female: 78.4%. A global scale about
the soles never got below 60% at any factor tried — shrinking laterally pulls
the humerus out of the mesh's sleeve into the gap beside the chest. The
disagreement is per-limb, in direction as well as length: the forearm axes
differ by 15 degrees, the shanks by 10.

**Built.**
- `body/fit_regions.py` — the skeleton as 17 regions, each a full 3×3 about
  the joint it hangs from, each anchor carried by its parent so no
  articulation can open. Trunk and head also carry a translation.
- `body/skeleton_fit.py` — `SkeletonFit`: capture, apply, exact reset, and
  the displacement field that carries the soft tissue.
- `tools/fit_skeleton_to_skin.py` — the offline solve (coordinate descent on
  squared protrusion, parents first) → `assets/config/skeleton_fit.json`,
  one table per sex, lerped at runtime.
- `tools/skeleton_containment.py` — signed distance to the surface, sign
  calibrated rather than read off the winding, which is inward on this mesh.
- `tools/render_skeleton_fit.py` — the picture, `--protrusion` colouring every
  bone vertex blue inside to red outside.
- GUI: **Body → Fit skeleton to body mesh**, `SKELETON_FIT_TOGGLED`, and
  `BodyController.on_skeleton_fit_toggled`.

**Measured after.** Outside 65.7% → 35.4% (male) and 78.4% → 40.5% (female);
median +2.42 → −0.78; p95 11.38 → 2.98; worst 27.7 → 7.2. Over every vertex
rather than the sample, 68.9% → 27.0%. Renders in `results/skeleton_fit/`.

**Two things the renders caught that the arithmetic did not.**

*The soft tissue tore.* A thin-plate spline extrapolates outside the hull of
its control points, and an eight-degree turn of the shoulder girdle made it
extrapolate hard — the trapezius and deltoid came away from the thorax in
wings. Replaced with an inverse-distance blend of displacements measured on
the bones, which is bounded by the largest of them wherever it is evaluated.
Its neighbour count and smoothing were chosen on the quadriceps, which span
the hip: worst p99 edge stretch 2.926 at 16/3, 2.022 at 48/10.

*A latent defect in the fibre field.* A footprinted muscle is placed every
frame from its harmonic field, and the field stores the rest pose it was
solved on. Nothing refreshed that when a morph rewrote the rest pose, so the
field wrote the old geometry back over the new one every frame. Fixed by
`MuscleAttachmentSystem.refresh_rest_poses`, called from
`refresh_after_skeleton_change`, so the sex morph gets it too. With the fix
the render is byte-identical to the control with attachments switched off.

**Deliberately not carried.** The skin layer that came with the skeleton. It
was scanned from these bones and already fits them; carried by the fit it came
out a head shorter and broad in the shoulders. It follows the sex morph and
nothing else, and renders byte-identically with the fit on or off.

**Still open.** The crown of the skull stands ~4 units proud of the scalp; the
hands end against the ±20° rotation bound; the lerp between the two solved
tables is not the fit of the lerped surface, and nothing checks it.
`docs/skeleton_fit.md` has the full measurements.
