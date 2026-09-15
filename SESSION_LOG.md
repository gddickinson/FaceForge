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

## 2026-09-12 (later still) — The fit learns to pose, and the drawings correct it

**Asked.** Can the skeleton be *posed* to fit the mesh -- arms rotated and
placed -- and can bones be reshaped further per sex?  And, if the solver keeps
falling short, look at the model myself on a grid and fit it by hand.

**Found, on inspection.** Each region's rotation was absolute in body
coordinates, so a turn at the shoulder moved the elbow but left the forearm
pointing where it was: the arm bent instead of swinging, and the hand was the
one region ending against the rotation bound because it had to express the
whole arm's turn alone.

**Built.** Rotations now compose down the chain the way a pose does, and a
region's own rotation is a correction relative to its parent. 17 regions
became 23 -- pelvis, lumbar and thorax separated, and the knuckles and the
ball of the foot made a break between hand and fingers, foot and toes. The
left half is the mirror of the right and the midline regions may not twist.
`tools/inspect_skeleton_fit.py` draws the whole thing orthographically on a
labelled grid in body units.

**Four things the drawings caught that the numbers could not.**

*The arms folded into the thighs.* Free to minimise protrusion, the search
swung both forearms across the body until the hands lay inside the thighs,
where nothing sticks out of anything: the fitted right wrist at x = -6.1,
across the midline, fingertips between the knees -- and containment called it
perfect. "Inside the surface" is not "inside the matching part of it".

*Landmarks cannot say which part.* Tethering each limb's distal end to the
mesh's own landmark for it made things worse, because the "ankle" landmark is
the mean of a band from the lateral half of the leg and sits 13 units off the
leg's axis: both feet were dragged clean out of the mesh, 100% outside at a
median of 8. Calibrating depth from the skin that came with the skeleton also
fails -- that mesh is not hollow, 24,757 of its vertices lie inside a 12-unit
column through the chest.

*What works is a travel limit.* This is a fit, not a reposing: the whole
misfit is 27.7 units at worst, so a bone moves 20 freely and pays beyond. The
shoulder's real 24-unit correction costs almost nothing; the degenerate
wrist's 51 costs more than the rest of the objective.

*A region must answer for its own bones.* Pooled over its whole subtree, the
thorax's own bones were a thirteenth of its objective and the chest wall was
left standing 7 to 9 units through the skin. Weighting by vertex count fixed
that and broke the arm, because vertex count is tessellation, not anatomy: the
hand's 10,000 vertices drowned the humerus's 400 and the scapula was left 17
units out. Half its own, half what it carries.

**Hand-tuned from the drawings.** The ribcage's anteroposterior scale was
sitting exactly on its lower bound with the sternum still 4 to 5 units proud
of the chest, so the bound went from 0.70 to 0.60 and the solver settled at
0.648. The cadaver's chest is deeper than the MakeHuman figure's.

**Measured.** Male 65.7% to 27.1% outside, median +2.42 to -1.13, p95 11.38 to
2.07, worst 27.7 to 7.7. Female 78.4% to 30.2%, p95 8.60 to 1.71. Both beat
the first shipped fit (35.4%, p95 2.98) on every measure, and the two halves
now match. Muscle edge stretch: the field's neighbourhood went 48/10 to 64/16,
worst p99 2.154 to 1.986. The skin that came with the skeleton still renders
byte-identically with the fit on or off.

**One regression, caught by the GUI budget.** The new checkbox took 2.65 s
against the 1.50 s responsiveness budget, all of it sampling the displacement
field on its lattice. The field is now built on first use rather than on
toggle -- 2.4 s to 0.09 s with nothing loaded -- and the lattice went from 2
units to 6. Coarser is better on both counts, because interpolating an
already-smooth field more coarsely only smooths it further: with skin and two
muscle layers loaded the toggle went 2.01 s to 1.45 s and the worst muscle p99
went 1.924 to 1.746. (`test_no_interaction_blocks_the_render_thread` still
fails when the whole UI suite runs in one process, on the pre-existing 71 s
"Run Diagnostic" button; it was failing that way before this work.)

**Still open.** The skull's crown, about 5 units. The anterior chest wall and
the upper thoracic spinous processes pull opposite ways and share one matrix.
The thumb, which the mesh's mitten hand has no room for.

## 2026-09-12 (end) — Posture: the arms were supinated, the skull was in the nape

**Reported, from looking at the model.** The skeleton's arms are supinated
while the mesh's are pronated. The pelvis sits too high. The shoulder, elbow
and wrist need to match better. The mesh's head is further forward than the
skull, which should move forward into the neck. The skull should pitch down
slightly and be resized and reshaped for both sexes.

**Measured. Every one of them held.**

| | measurement |
|---|---|
| palm planes | 88.5 deg apart (male), 88.0 (female) |
| skull depth | 29.1 inside a 26.0 head |
| occiput | 7.2 units out the back |
| crown | 27.5 above the scalp |
| hip joints | -81.0, with the mesh's hip mass far lower |

**Built.** Posture, as a thing the fit carries rather than searches for.
`AXIAL_POSTURE` turns each forearm 92 degrees about the axis its own elbow and
wrist define; `SHAPE_POSTURE` states the skull's proportions, 22 units deep and
17.6 wide, each leaving two units of scalp, moved forward to sit in the face.
A pronated forearm and a supinated one fill the same sleeve, and a skull
cannot be pulled backward by scaling about a joint underneath it, so no
objective was ever going to find either.

**And two more terms in the objective**, both from the same lesson. The 98th
percentile of a region's protrusion, because a mean tolerates one deep patch
and a deep patch is what you see. And a loose guard against burial, because
pressed harder than that the search bought a better worst case by hiding the
hands 9.4 units inside the surface.

**Measured after.** Male 65.7% to 19.0% of bone vertices outside, p95 11.38 to
0.88, worst 27.7 to 6.9. Female 78.4% to 8.1%, p95 8.60 to 0.25. Palms 88.5
deg to 4.8. Hip joints -81.0 to -91.5. Upper arm 40.4 to 31.0 against the
mesh's 27.8. Middle fingertip 62.7 units from the mesh's to 2.0. Skull 29.1
deep to 22.6 inside a 26.0 head, occiput flush, crown 27.5 to 3.3. The two
skulls now differ: the female's is shallower (0.70 against 0.73) and shorter
(0.89 against 0.96).

**Cost.** Dropping the pelvis 10 units puts 2.2x edge stretch into the
adductors at p99, against 1.75 before. More field smoothing buys 2.02 at the
cost of the median, which is not a trade worth making.

**What I could not measure my way to.** Both authored corrections came from
looking at the model on a grid, which is where this round started.

## 2026-09-12 (end) — Verifying the fit: the hands, the thigh, and every layer

**Asked.** Check the fit by eye and fine-tune; make sure the muscles, the
other soft tissue and the organs still fit; make sure the female anatomy is
altered correctly throughout. And, separately: the hands fit the male mesh but
end up inside the torso for the female.

**The female hands.** The female solve had folded the upper arm inward to its
rotation bound and buried the hand 34.5 units from where the mesh keeps it,
while the male's fitted to 2.2. Sexual dimorphism is proportion, not posture,
so the pose is now solved once and the second sex inherits it; only the
proportions are searched again. Hands now 2.2, 1.7 and 2.3 from the mesh's at
gender 0, 0.5 and 1.0. `tools/fit_skeleton_to_skin.py` also fails loudly now
if a fingertip or toe-tip ends far from the mesh's own landmark for it --
containment cannot see an arm folded into a torso, and that is twice it has
hidden one.

**The thigh.** `tools/fit_tissue_check.py`, written for this, found the
quadriceps standing 15 units through the skin where they had stood 4. The fit
was not at fault: under the thigh's own transform they sit *inside*, at -0.6.
The field was. Four versions, each failing in a way the numbers being watched
at the time could not show:

- a thin-plate spline extrapolates outside its hull and tore the shoulder;
- a blend of displacements is bounded but cannot extrapolate a scale, so a
  muscle eight units outside a shrinking femur barely moves;
- a blend of the region matrices extrapolates but is not closed under
  averaging, and a 92-degree forearm turn collapsed it (stretch 2.21 to 5.37);
- a blend of *where each region puts the point* works, with two guards.

The guards are the lesson. No region may move a point further than it moved
its own bones. And influence is limited by distance along the region tree, not
through space: the finger bones hang beside the thigh, close enough to take
31% of the weight on the quadriceps and seven steps away in the skeleton, and
carrying a 92-degree pronation extrapolated 30 units. That one weight was the
whole of the 16-unit error. Worst muscle p99 stretch 2.21 to 1.61, quadriceps
15.9 units out to 1.2.

**Every layer, both sexes.** Furthest protrusion through the surface, before
and after, male: arm 19.3 to 6.1, back 18.9 to 5.3, shoulder 24.1 to 4.7,
torso 17.4 to 8.2, hip 10.2 to 3.2, leg 14.8 to 10.5, hand 7.1 to 1.8, foot
14.0 to 2.2, organs 14.8 to 4.2, vasculature 11.3 to 4.4, ligaments 11.1 to
2.3. Female better throughout. 32 meshes of 442 past some limit at gender 0,
14 at gender 1.

**The female changes.** Measured with the fit off and on: muscles move 3.11
against 3.32 and lose the same bulk (x0.621 against x0.609); organs and other
soft tissue 1.81 against 2.05; the skeleton's own skin identical. The fit
survives a sex change and does not interfere with one.

**Still open.** The male adductors stand 10.5 units out where they stood 2.7;
the female's do not. The digastric intermediate tendons stretch 4.5x.

## 2026-09-13 — Five improvements to the male and female models

**Asked.** What else can be done to improve the models, then: do all of it.

**1. The head had no sex.** Measured, exactly five bone meshes were unchanged
between gender 0 and gender 1 -- cranium, jaw, both sets of teeth, atlas --
which is the whole head. The config's per-bone cranial factors name bones and
the asset is one merged mesh, so none had ever matched. Patterns added; the
merged skull takes published head dimensions; `skull_morph.py` adds the extra
facial narrowing the vault's factor cannot carry, graded from the brow down.

**2. The body surface was too coarse to fit.** 10,582 vertices and 21,160
triangles for a whole body, edges 1.13 / 3.75 / 7.72. One Loop subdivision:
42,322 and 84,640, edges 0.54 / 1.82 / 4.04, in 0.07 s, with the shape
untouched to a tenth of a per cent. Loop rather than a midpoint split, which
would quadruple the triangles and describe the same faceted surface.

**3. Nothing checked the morph against the book.** `tools/anthropometry.py`
now does, and found three errors on its first run: stature 0.950 against
0.928, sitting height 0.985 against 0.932, bi-iliac breadth 1.000 against
0.964. The first two were one defect -- the skull scaled about its own
centroid and the cervical column about a T1 that never moves, so the head
floated while the trunk shortened under it, which also made the vertebral
factors inert (0.95 to 0.91 moved stature by 0.002). Every ratio is now within
0.009.

**4. Two dimorphic features are angles, not proportions.** The carrying angle
and genu valgum, each two degrees wider in the female, neither of which any
amount of bone scaling produces. Only the difference is applied; the absolutes
are the donor's, and asymmetric as a real body's are. Measuring them signed
mattered: the arm hangs abducted, so an unsigned measure read the increase as
a decrease.

**5. The defects the last round left.** The adductors, which stood 10.5 units
through the medial thigh, now stand 4.6 -- fixed by the field work, not by
anything aimed at them. Nothing anywhere on the skeleton now stands more than
2.74 units through the skin, against 27.76 before: the scapula's inferior
angle and the skull's crown.

**Two regressions caught by the guards.** Re-solving against the finer surface,
the male fit folded both arms into the torso again -- fingertips 37 units from
the mesh's hands -- and the tool wrote it before checking. The check now runs
before the write, and the hands and feet are tethered loosely to the mesh's own
landmarks: fourteen units free, which a right answer never uses.

**Every tissue layer, both sexes.** 19 meshes of 442 past a limit at gender 0
and 14 at gender 1. The sex morph's effect on soft tissue is unchanged by the
fit: muscles move 3.24 against 2.96 and lose the same bulk, the skin is
identical.

**Still open.** The digastric intermediate tendons stretch 3.2x. The face mesh
and facial muscles have no sex of their own -- they follow the skull now, but
nothing shapes them.

## 2026-09-13 (later) — The head's soft tissue, the female organs, and a matrix

**Asked.** The head muscles, skull and neck/shoulder muscles do not fit the
skin mesh. Females do not have male reproductive organs and they do have
breast tissue. Then render through the GUI and check every layer with the fit
on and off, male to female.

**The head's soft tissue was never moved by anything.** The neck, jaw and
expression muscles, the face and the face features each keep their own copy of
their rest pose and rebuild from it every frame, because each has its own
deformer. Nothing rewrote those copies, so when the skull moved -- reshaped for
sex, seated lower on the neck, and moved again by the fit -- the muscles on it
stayed. `anatomy/head_tissue.py` moves them by the same field the skinning
gets, from a captured original so nothing compounds. Measured: with the fit on
the skull moves 23.66 and the head tissue a median 23.50, worst lag 1.81, and
switching the fit off restores exactly. They are also excluded from the
skeleton morph now -- "Zygomatic Maj." matches the zygomatic bone's pattern and
was being scaled as one.

**The female had male organs.** Eight of the configured organs are male
reproductive structures and the asset set has no female equivalent of any of
them, so they are hidden by the middle of the slider. The urethra goes too:
both sexes have one, but this mesh is the male's, 20 cm through the penis.

**And no breast.** Built as what it anatomically is: a lens from the skin down
to the chest wall over ribs two to six, deepest at the nipple. The first
attempt built it from the female-minus-male surface difference and was wrong
twice -- the MakeHuman female base gains only one unit of chest projection, so
it came out a sliver, and the inner boundary of a breast is the pectoral
fascia, not the male skin. Men have mammary tissue too, so the depth runs from
0.8 to 4.0 units rather than from nothing.

**The matrix.** `tools/render_model_matrix.py` renders every layer, both
sexes, fit off and on, through the application's own event bus and renderer.
It found two things a single frame would not have.

*The layer isolation was broken in the tool itself* -- a demand-loaded layer
stays in the scene once loaded, so every frame showed everything.

*And the fitted skeleton stood through the skin.* The skeleton's own skin had
been held back from the fit, on the grounds that carrying it mangled it. That
was measured against a field that could not extrapolate a scale; carried by
the field as it now stands it stretches 1.17x at the 99th percentile and
simply becomes the body the fit describes, 203 units tall rather than 227.
Carrying it exposed the asset's welds as webs between the forearm and the hip,
and those welds now have a test that works: real skin stretches about 1.6x
when the arm turns, a weld across a gap stretches without limit, and the
triangles past 2x are not drawn. That needed the index buffer to be
streamable, which it was not -- and the first attempt at that unbound the EBO
from the VAO and stopped every indexed mesh drawing.

**Still open.** A residue of webbing remains around the hip. The breast sits
under the skin where the surface's chest is, which is a rib lower than the
fitted skeleton's ribs two to six.

## 2026-09-14 — Four defects the sex morph and the fit left in the head, chest and pelvis

Four things reported together, and they turned out to have almost nothing in
common except that each was a mesh nobody had told about a change.

**The face was torn, not dragged.** At gender 1 the scalp muscles descended
6.2 units and Orbicularis Oris 0.95 — a 5.3-unit differential across a face
20 units tall. The soft tissue follows a thin-plate spline through the
skeleton's own displacements, and a spline interpolates its control points
*exactly*, so one lying control point is not smoothed away: it is honoured,
and the mesh is torn to reach it. Three were lying.

*"Zygomatic Maj. L" is a muscle of facial expression* and reads as the
zygomatic bone to the name test that decides which bones carry a dimorphism
factor. `SkeletonMorph.apply` already knows a name test is not enough of a
guard and takes an `exclude` set of meshes it must never touch; `control_points`
did not. So the morph correctly left the muscle alone, and the muscle then
reported a displacement of exactly zero in the middle of the cheek while the
cranium two centimetres away reported −5.2.

*The eyeball pivots sit under `faceFeatureGroup`*, which `seat_on_neck` did
not carry, so they held the field at zero over the orbits while the head came
down around them.

*And face-feature records under a pivot* were being rebased as though their
vertices were in body coordinates. They are in the pivot's frame, and they
ride with it anyway.

After: every head part follows the field to within 0.05 units, and the head
descends as one piece — −4.3 at the mouth to −6.4 at the occiput, which is
exactly a 0.95 scale about the cranial centroid plus the seat. Jaw −5.53 and
lower teeth −5.66 in world coordinates against the cranium's −5.17. Face-mesh
stretch p99 1.17 → 1.13.

**The brain floated a whole head above the body.** It hangs off `brainGroup`
rather than off the skull, deliberately, so it stays visible when the skull is
hidden — and the cost was that nothing carried it when the skull moved. With
the fit on the skull came down from z +0.1..+27.6 to −21.7..+2.5 and the brain
stayed at +5.6..+27.1. It is soft tissue in the cranial vault, so it now rides
the same field the neck, jaw and expression muscles do. Brain vertices outside
the body surface: 100% → 13.5% with the fit, 7.3% at gender 1, and exactly
reversible. The brain-in-cranium relationship is unchanged by the fit (5.7%
outside before, 6.6% after — the brainstem leaving through the foramen magnum,
which is the asset's own baseline).

Demand-loaded layers now catch up on arrival: the controller keeps the field in
force and re-applies it from `on_structures_registered`, so a brain loaded
*after* the fit is not left in the pose the asset was authored in.

**A female model kept the shaft of the penis.** The male-only list was kept by
hand and had gone stale: it named the glans but neither erectile body, and
neither deferent duct. Twelve organs in `organs.json` are categorised
reproductive and it named eight. The asset set is a male cadaver, so *every*
reproductive organ in it is a male one — the list is read from the config now,
plus the male urethra by name. Measured through the event bus: 13 structures
hidden at gender ≥ 0.5 and back at 0, in every order of loading, toggling,
fitting and per-structure override.

**And the breast left the skin it was cut from.** `mammaryTissue` hangs off
`bodyRoot` and matches no membership pattern in `fit_regions`, so it inherited
the root region and was carried down with the pelvis: 11 units below the skin
at gender 0, 15 at gender 1. Its outer face *is* the chest's skin, and the fit
never moves the surface, so it never had anything to gain from being moved.
Skipped now, and it sits on the skin exactly — centroid z −56.4 male, −54.4
female, identical with the fit on and off. With the skeleton actually inside
the body, the lens's inner face lies 1.4–1.7 units from pectoralis major and
1.1–1.2 from the rib cage: on the pectoral fascia, which is where a breast
sits.

**One measurement that was not a defect.** The pectoral muscles looked welded
in place while their ribs descended 22 units under the fit. They are placed by
the attachment system every frame, and the probe was not stepping the
simulation. With `Simulation.step` running they track the ribs exactly
(−24.0 → −44.4 against the fourth rib's −20.0 → −42.1). Worth recording
because the wrong reading was very convincing.

**Still open.** The residue of webbing around the hip. And the unfitted
skeleton's head sits a head-height above the MakeHuman surface's crown, which
is the mismatch the fit option exists to correct rather than a bug in itself —
but it means every head layer looks wrong until the fit is switched on.

## 2026-09-14 (later) — What actually sprays the skin across the hip

With the fit **off** the skin renders as a clean body. With it **on** there is
a spray of stray triangles across the gap between each hand and hip. That had
been recorded as "a residue of webbing" -- leftovers of the weld the asset
puts between the hand and the hip, which `weld_webs` culls by stretch.

It is not that. Four hypotheses, each measured and each wrong:

* **Stretch.** The spray's triangles stretch by a median of **1.00**. They are
  not stretched at all, so no threshold on stretch can ever find them.
* **Small components.** The skin already has 528 connected components and 381
  of them are under 32 triangles *before* the fit; culling them changes the
  render barely at all, and the spray is attached to the main 1.53M-triangle
  sheet anyway.
* **Sticking out of the body.** The BP3D skin protrudes from the MakeHuman
  surface all over (p90 +0.96, p99 +3.90), and the spray's median depth is
  +0.75. A threshold that caught 137 of the 330 would drop 9.7% of all skin.
* **A bad binding.** The spray's vertices sit a median 16.7 units from the
  bone that drives them, against p50 6.47 and p90 17.44 for the skin at
  large. Entirely ordinary.

What it is: **the lattice the field is sampled on**. The skinning contributes
*nothing* to these vertices -- drawn position equals rest position to 0.0 --
so the displacement is the fit's field alone, and `FIELD_LATTICE = 3.0` is
coarser than the gap between the hand and the thigh. One lattice node serves
both sides, and trilinear interpolation across it drags skin off each surface
into the middle. Rendered with the field evaluated exactly rather than on the
lattice, the gap is **completely clean**.

Exact is not shippable: 51s against 8s for the whole toggle with every layer
loaded. So two things were done instead.

**The lattice skips the air.** It spans the body's bounding box, and a
standing body fills less than a third of it; every node of the space between
the legs and beside the arms was being evaluated to describe how nothing
moves. Only nodes within `margin` of a control point are evaluated now, and a
cell whose eight corners were not all evaluated falls back to the exact warp,
the escape the lattice's outside already used. Identical on the body to the
last floating-point bit, marginally *more* accurate outside it, and:

| spacing | before | after |
|---|---|---|
| 3.0 (shipped) | 2.59s | 2.13s |
| 1.5 | 9.79s | 5.56s |
| 1.0 | 27.80s | 14.06s |

**A digit is never evidence for which part of the body a point belongs to.**
The field decides that from the region owning the nearest bone, then keeps
only the regions within `REGION_REACH` steps of it. With the arms down the
finger bones hang beside the thigh, and a thigh's *surface* is 7.5 units out
from its own femur and 1.5 from the finger beside it -- so the guard inverted:
it kept the hand, which is what the finger belongs to, and threw away the
thigh, which is what the skin belongs to. Measured, 14,178 skin vertices --
1.8 per cent -- took their body part from a finger or a toe. The anchor comes
from a second tree over the non-digit bones now; a digit's own skin still
anchors on its hand or foot, one step away, which keeps the digits. 19 of 442
meshes past a limit in `fit_tissue_check` before, 18 after.

**Still open.** The hip spray itself. It is the lattice, and the fix is to
spend the sampling: at 1.5 the spray is down to about six triangles for
+3.4s on the toggle. That is a trade against the 1.50s responsiveness budget
the coarse lattice was chosen for in the first place, so the spacing has been
left at 3.0 rather than reversed quietly.

## 2026-09-14 (later still) — The male and female models were never skinned

Reported: the skin meshes do not move with exercise, for either sex.

Two different meshes go by "skin" here and only one of them was bound. The
BP3D **skin layer** is registered by `load_skin` and deforms correctly --
measured through the exercise runtime, a squat moves it up to 72.91 units and
a rendered demo shows it bending around the whole body. The **body-surface
mesh** is the male/female model itself: it comes from `GenderMorphSystem`
rather than from an STL layer, so `load_skin` never saw it, nothing else
registered it, and it was in no binding at all. Posed into a squat it moved
**0.00 units, maximum, over 42,322 vertices** while the bones bent inside it.

Two things were missing.

*A rest pose.* `_morph_body_surface` lerps the male and female shapes into the
vertex buffer, and the lerped shape is the surface's rest pose at that sex --
but it was only ever written to `geometry.positions`, which is where the
skinning writes its *output*. It is written to `rest_positions` too now.

*A binding.* `register_body_surface` hands it to the skinning exactly as
`load_skin` hands over the skin: every chain, and the same two-tier spatial
filter (`SKIN_CHAIN_Z_MARGIN`, `SKIN_SPATIAL_LIMIT`), because it is the same
kind of object -- one closed surface over the whole body. It runs at the end
of `build_skinning`, as soon as the chains exist.

It is also the surface the skeleton fit aims at, so `rebuild_soft_tissue`
holds it back from the fit's own field: carrying it there would move the
target while measuring against it.

Measured after, with hips, knees, shoulders and elbows flexed:

| | moves | median | max | edge stretch p99 | over 2x |
|---|---|---|---|---|---|
| male surface | 56.7% | 46.34 | 151.01 | 1.83 | 1042 / 126,960 |
| female surface | 56.7% | 40.60 | 147.03 | 1.48 | 456 / 126,960 |
| BP3D skin (shipped) | — | — | — | 2.03 | 24,761 / 2,379,747 |

So the surface deforms slightly *better* than the skin already shipping, at
0.8% of edges past 2x against the skin's 1.0%. The 56.7% that move are the
limbs; the trunk is rigid because this pose does not bend the spine, and the
BP3D skin behaves identically.

Nothing else moved. At rest the surface is still the authored MakeHuman shape
to 8e-6 -- float32 -- at both sexes, so the fit's target and the breast lens's
source are untouched; the sex morph still reshapes it (median 5.85, max
11.42); and the fit still leaves it exactly alone (max move 0.0000).

Confirmed in pixels: a bodyweight squat rendered through the application's own
event bus, showing the surface alone, stands at t=0 and is folded at the knees
and hips at t=0.40, its world height going 127.6 -> 70.3 -> 127.6.

## 2026-09-14 (later still) — The ground lock anchored a pivot, not a sole

The lock keeps a chosen support in place by translating the wrapper, and its
foot target came from the pivots' own rest height, on the stated grounds that
"standing places the soles on the floor". It does not: an ankle pivot is
inside the ankle. Measured in the gym through a squat:

| | before | after |
|---|---|---|
| lowest foot pivot | 7.9 | 4.8 |
| lowest foot bone | 5.9 | 2.8 |
| lowest point of the skin | 4.5 | 1.4 |
| lowest point of the body | 0.0 standing, **-44.0** mid-squat | 0.0 throughout |

So the body stood about 4.5 units clear of the floor -- 4 cm at this model's
scale -- steady and self-consistent, but in the air. `calibrate` now measures
how far the body's lowest point rests above `floor_y` once, in the rest pose,
and lowers the pivot target by exactly that. The per-frame anchor is still the
pivot it always was, which is what keeps it cheap; only the target moved. A
rig with no meshes (a test, a bare skeleton) has nothing to measure and keeps
the old target, so the fallback is the previous behaviour rather than a guess.

The `-44.0` is a second bug the same probe exposed: `tools/headless_loader.py`
did not bind the body surface either, so in a headless render the male/female
model was carried down bodily by the wrapper in its standing shape and drove
its feet 44 units through the floor. `register_body_surface` now takes the
morph and the chain ids rather than the application context, and the headless
loader calls it where the app does -- the parity rule these tools are held to.

Checked across the exercise catalogue: every feet-anchored exercise holds its
anchor with drift 0.00 over the clip, at 4.8 instead of 7.9. The four that
drift are the jumps, which leave the floor by design, and `jumping_jack` sits
at 4.8 + its authored `lift=3.0`. Hands-anchored exercises are untouched at
3.0; `mountain_climber`'s feet-below-floor flag is on that same untouched
branch and pre-dates this.

**A correction.** I had described the figure in an earlier demo render as
floating, and it is less visible than that suggested: the demo camera tracks
the body, so a three-unit drop moves both and the frames look alike. The gap
was real and is now measured at contact -- the lowest visible point sits 0.43
units above the platform's top -- but the evidence for it is the measurement,
not the picture.

## 2026-09-15 — The stance was already flat; one foot was in the air

Asked to fix the foot posture so the stance is flat. It already was, and the
measurement says so plainly: in the standing rest pose the right foot's heel
sits at z −196.92 and its ball at −197.09, a difference of 0.17 over a foot 26
units long — **0.6 degrees** from horizontal. The BP3D skin over it measures
−0.3 degrees and the body surface +0.2. Through a whole bodyweight squat both
feet hold 0.6 degrees, and a side-on close-up render shows the sole flat on
the floor at the top and at the bottom of the rep. My earlier remark that the
figure "reads as slightly on tiptoe" was a misreading of a perspective render
and should not have been made without measuring.

What the measurement did find is that **one foot was in the air**. The ground
lock translates the whole body so its lowest support sits on the floor — one
translation for two feet, which is enough only while the feet end a pose at
the same height. They do not, because the donor is a real body:

    knee-to-ankle, rest    R [-0.68, 1.35, -46.75]
                           L [-2.66, 5.77, -46.55]     about 7 degrees apart
    femur length           R 60.37   L 58.58

Standing costs nothing — the ankles sit 0.4 units apart. Bend the same joints
by the same angles and the rotation amplifies it: at the bottom of a squat the
ankles ended **6.8 units apart**, so the lock planted the right foot and left
the left one hanging. The rig itself is sound: both legs turn the same angle
about the same axis for the same DOF (89.92 against 89.99 at the hip, 144.96
against 144.42 at the knee), so this is anatomy, not a bug.

`FootLevelLock` closes it the way `GripWidthLock` closes a grip — measure,
probe for the slope, take a Newton step — with three things learned by
measuring:

*The ankle must follow the knee.* The sole is flat when ankle dorsiflexion
equals pitch minus hip plus knee. Moving the knee alone broke that identity
and tilted the sole 43 degrees, driving the toes 2.2 units through the floor
and levering the heel 17 units up. Both are written in degrees, since the
knee's range is 145 and the ankle's 45.

*The probe must move what the step moves.* Measuring the slope with the ankle
held still answers a different question, and the step then overshot: the left
foot went 1.9 units past the right instead of meeting it.

*A lock that cannot reach must not try.* Deadlift knees are nearly straight
and cannot extend further, so the solve chased a foot it could never reach —
45 degrees of knee on a Romanian deadlift, still missing by 19 units. It now
keeps its work only if it halves the gap, and otherwise restores the pose
exactly as the clip authored it.

Result, at the bottom of a bodyweight squat: the two feet **0.02 units apart**
against 6.46 before, both soles still flat at 0.6 degrees, for 25 degrees of
knee. Across the catalogue: Romanian deadlift 45 degrees of wasted knee down
to 2.6, conventional deadlift's gap 13.4 down to 4.4, split squats untouched
because their anchor names a side, and the hands-anchored exercises untouched
because the lock is only built for feet.

**Still open.** `step_up` holds a 44-unit gap by design (a foot on a box) and
`conventional_deadlift` still ends 4.4 units apart — the knee alone cannot
close every pose, and the hip would have to join the solve to do better.

### The hip joins the solve

The knee alone left four exercises short, because a knee that is already
straight has no extension left to reach the floor with: the conventional
deadlift ended 4.4 units apart, the Romanian deadlift 4.5, a wall sit 2.7.

The hip has the range. It is also the angle that reads as the shape of a lift,
so it is not simply added as a second knob: the step is now one equation in
two unknowns, solved for the least costly solution with the hip charged four
times the knee (`JOINT_COST`). Where the knee can do the work it still does;
where it cannot, the hip makes up the difference and no more. Its ankle
coupling runs the other way -- dorsiflexion is pitch minus hip plus knee -- so
a hip that flexes owes the ankle the same angle back, and `ANKLE_PER` holds
both ratios in degrees over degrees because the three ranges are 145, 90 and
45.

| exercise | knee only | knee + hip | knee used | hip used |
|---|---|---|---|---|
| bodyweight squat | 1.11 | **0.43** | 13.8 deg | 1.8 deg |
| barbell back squat | 0.67 | **0.42** | 13.9 | 1.5 |
| kettlebell swing | 0.70 | **0.63** | 17.4 | 1.3 |
| wall sit | 2.70 | **0.40** | 3.8 | 2.3 |
| conventional deadlift | 4.43 | **0.54** | 21.0 | 6.3 |
| Romanian deadlift | 4.50 | **1.43** | 15.0 | 16.9 |

Every gap is now inside 1.5 units and most inside 0.65, against 6.8 before any
of this. The knee moves *less* than it did with the knee alone -- 13.8 degrees
against 25.4 on a squat -- because sharing the correction conditions the solve
better. The Romanian deadlift spends the most hip, 16.9 degrees, which is what
a hinge with straight knees has to spend.

Across all 63 exercises the anchor still holds with drift 0.00, apart from the
jumps and step-ups that leave the floor by design and the two hands-anchored
exercises, whose numbers are unchanged because the lock is built only for
feet. Split squats remain untouched: their anchor names a side.

## 2026-09-15 — A thumb that would not close, and hands that slid along the bar

**The thumb stuck out of every grip.** Measured on a deadlift bar, the four
fingers sat 1.55 units from the bar's axis and the thumb 7.85. Two causes,
both the same mistake: the thumb's rotations were authored in the fingers'
axes, and the thumb does not share them.

The thumb's metacarpal leaves the wrist about 45 degrees out of the palm --
the rest segment runs [0.71, -0.03, -0.70] against the index's near-vertical
[0.33, -0.24, -0.88] -- so in its own pivot frame:

* *Curl about X swings the tip backwards out of the hand.* Driven harder it
  got worse, not better: at curl 0.50 the thumb tip was 2.95 from the index
  tip, at 0.94 it was 8.34, at 1.00 with full opposition 9.22. Zeroing the
  thumb entirely gave 3.24 against the shipped code's 7.85 -- doing nothing
  beat it, which is the measure of how wrong the axis was. About **Y** the
  same angles bring it to 1.69.
* *Opposition's other two components pushed it out.* Modelled as flexion
  about X plus pronation about Y plus adduction about Z, which describes the
  motion correctly and these axes incorrectly. With the curl moved to Y,
  dropping the X term took the thumb from 4.91 to 2.82 and dropping Z as well
  to **1.84**, against the fingers' 1.55. Opposition in this rig is the Y
  rotation and only that.

Verified on both hands and in pixels: deadlift R thumb 1.84 (fingers 1.55),
L 1.69 (fingers 2.30); back squat R 1.94, L 2.59. The renders show the thumb
wrapped with the fingers where it used to jut away from the bar.

**The hands slid along the bench press bar.** 13.83 units each, every rep --
the grip opening from 110.3 to 138.0 and closing again. `GripWidthLock`
existed for exactly this and was only ever built for an exercise whose hands
are anchored to a bar in the room, which a bench press's are not: the bar
moves with the lifter.

What fixes the width is not the anchor but the *implement*. An
`EquipmentSpec` with `attach="hands"` is a rigid thing held between both of
them -- a barbell -- as against `hand_r`/`hand_l` for a dumbbell in each hand,
whose width is free to change. The lock is built for either now. Bench press
slide 13.83 -> **0.10**, with shoulder abduction still travelling 12 to 78
degrees and back, so the movement keeps its shape.

**The bench press had no arch and a passive back.** The arch is still
missing, and the attempt to add it is worth recording as a failure.

Ten degrees of `spine_flex` extension went into both keyframes on the
strength of a headless measurement that showed 1.03 units of thoracic pivot
movement. Rendered with the plates hidden, the arched and flat frames are
indistinguishable, and a proper sweep says why: from -30 to +30 degrees
against a supine trunk, `spine_flex` leaves the **sternum at 89.71**, the
**shoulders at 68.56** and the **lowest lumbar pivot at 64.56** -- all three
unmoved -- and shifts only the top of the lumbar chain, about a unit per 30
degrees. The 1.03 figure was a pivot displacement summed over all three axes
on a different code path; it never corresponded to anything visible.

A bench arch is the opposite shape to what this DOF makes: pelvis and
shoulders down, the middle of the back lifted off the bench. A serial chain
driven from the pelvis cannot produce it, and the ribcage would have to follow
the thoracic pivots to show it at all -- the sternum not moving by a
thousandth across a 60-degree sweep says it does not. So the pose carries no
arch rather than a number that moves nothing.

The back is not passive in a bench press. Latissimus dorsi is a shoulder
extensor and adductor, so it resists the bar on the way down and keeps the
humerus packed at the chest; the rhomboids and middle trapezius hold the
blades retracted and depressed for the whole set, which is what gives the
press something to push from. The lat goes from 0.30 to 0.45 with a note
covering both directions, and rhomboids (0.40), middle trapezius (0.35) and
erector spinae (0.30, bracing the trunk) join it. All four are stabilisers, so
the activation model holds them at full level through every phase -- measured
0.45 on the lats at each of nine samples across the rep, the lowering
included.


## 2026-09-15 (later) — The thorax follows the spine now; the arch still does not

Two items were left open: the ribcage not following the thoracic pivots, and
a bench arch needing a mechanism that fixes both ends and lifts the middle.
The first is fixed. The second is not, and the measurement that settles it is
worth more than the attempt was.

**The ribs hang off the spine now.** The thoracic pivots carried their own
vertebra and disc and nothing else -- `thoracic_spine_pivot_0` held `T2` and
`T1-T2 Disc` -- while all 24 ribs, 16 costal cartilages and the sternum hung
off `<bone>_breath_pivot` under `rib_cage`, a *sibling* of the spine on
`bodyRoot`. So the spine could bend and the thorax would stay where it was.

`attach_ribs_to_spine` reparents each of those 41 pivots onto the vertebra its
rib articulates with, reading the level from the bone's own name and sending
the sternum, manubrium and xiphoid to the top of the chain. Rest positions are
summed up the chain rather than read from world matrices, because it runs
while the skeleton is still being assembled, and the reparenting preserves
each pivot's rest position exactly -- so a binding solved against the rest pose
is undisturbed and only what happens when the spine moves is different.

Measured across a 60-degree sweep: the sternum travels 10 units where it did
not move by a thousandth before, the ribs average 1.2 units of displacement at
30 degrees, and at zero every one of 115 thoracic bones is where it was.
Breathing still moves the ribs (max 3.60, mean 1.18).

**The arch is still not possible, and now for a reason that is measured rather
than guessed.** I re-added it on the strength of that 10-unit sternum travel,
which was the wrong axis to read -- the body frame's +Z is head-ward, and a
supine lifter's *up* is the frame's -Y. Broken out properly, across the same
sweep:

| spine_flex | y (anterior = up, supine) | z (superior) |
|---|---|---|
| -30 | -16.40 | -21.47 |
| 0 | -16.71 | -26.66 |
| +30 | -15.97 | -31.53 |

The chest slides **10 units head-ward** and lifts **0.32**. `spine_flex` bends
the trunk as a chain from the pelvis; a bench arch fixes both ends -- pelvis
and shoulders on the bench -- and bows the middle clear of it. That is a
different mechanism from a joint chain, not a different number in one, so the
pose carries no arch and the comment in the catalogue carries the table.

That is twice I read a displacement on the wrong axis and reported an arch
that was not there. The measurement to trust for anything supine is the
anterior component, not the largest one.

**The grip lock, checked across every bar.** Slide per hand over a rep:
sumo deadlift 0.00, triceps pushdown 0.00, bench press 0.10, overhead press
0.18, barbell row 1.95, lying triceps extension 4.05 -- against the bench
press's 13.83 before any of it. Six exercises' anchor positions moved in the
catalogue sweep, all of them bar-holders, which is the lock doing its work.

## 2026-09-15 (later) — The bench press was gripped underhand, and a tendon was a blade

**The palms faced the head.** A standard bench press is gripped overhand, so
the palms face the lifter's feet. Measured at the bottom keyframe, the palm
normal ran **+0.463** along the head-ward axis -- the wrong way -- and no
forearm rotation could fix it: swept through a full circle the best available
was +0.448, because it is the shoulder's *axial rotation* that decides which
way the hand ends up, and the pose had it at -20 degrees.

Shoulder rotation +30 with the forearm at 60 gives **-0.364**, palms to the
feet, with the forearm still 0.90 of the way to vertical. The lockout keyframe
was already right at -0.090 and a two-axis sweep found nothing better; the
palm there faces mostly up at the bar, as it should. Through the clip the
right palm now measures -0.514 where it was +0.463.

**Both grips ship now.** The numbers the standard press used to carry are what
a reverse-grip bench press wants, so `reverse_grip_bench_press` takes them.
The supinated grip externally rotates the humerus and tucks the elbows, which
moves the emphasis to the **clavicular head** of pectoralis major -- it is the
primary there at 0.9 against the sternal head's 0.7, where the standard press
has them the other way round -- and loads **biceps brachii** as a secondary at
0.45 rather than a stabiliser at 0.2, because a supinated forearm holds the
bar rather than sitting under it.

**A tendon was drawn as a blade.** Palmaris longus ends in the palmar
aponeurosis and has no tendon to any finger. It was bound to all five digit
chains -- 10,775 of its 22,179 vertices, 48.6 per cent -- so closing the fist
round a bar tore it to **10.59x** its rest edge length at the 99th percentile,
three times the next worst muscle in the forearm, and it rendered as a flat
green sheet fanning out past the fingers.

`MUSCLE_CHAIN_OVERRIDES` now holds the forearm muscles that never reach a
phalanx -- palmaris longus, the four carpal flexors and extensors, the
pronators, supinator and brachioradialis -- at `spine+arm`. The wrist muscles
stop at the carpus or a metacarpal base and the metacarpals move 8 degrees
with a curl, so following the arm costs them nothing. Palmaris longus:
**10.59x -> 1.52x**, 48.6 per cent on digit chains -> 0.

`resolve_sided_chains` gained `"hand1".."hand5"` for a muscle with one tendon,
and the thumb's long muscles use it: extensor pollicis longus 2.89x -> 2.70x,
flexor pollicis longus held at 1.80x.

*Extensor indicis and extensor digiti minimi were tried the same way and are
deliberately not there.* Restricting extensor indicis to digit 2 took it from
3.48x to **4.20x** -- worse -- because the vertices that had been following
the middle finger were pulled onto the index instead of let go. The mesh's
distal end spans more than the one tendon its name implies, and pinning it
harder is the wrong correction. It is the residual visible in the render.

## 2026-09-15 (later) — 29 more exercises: bench variations, kettlebells, and the rest

The catalogue went from 64 to **93**. Every one validates against the rig's
joint limits, builds a playable clip, and places the body without a warning
from `render_exercise_demo --probe --all` -- the 13 warnings in that sweep are
all pre-existing.

**The bench press by what is changed about it** (`bench_variants.py`, 5):
close grip, wide grip, incline barbell, decline barbell, floor press. Grip
width is set by shoulder abduction at the bottom -- 45, 75 and 92 degrees give
78, 110 and 132 units between the hands. A wider grip abducts the shoulder
further and shortens the bar path, so pectoralis major takes more and triceps
less; a narrower one tucks the elbow and lengthens the path, so the triceps
take it (ANDERSEN). Incline sits at 30 degrees because clavicular pectoralis
peaks there and anterior deltoid keeps rising past it (BENCH_INCLINE). The
floor press loses the bottom third, the stretch and the leg drive, which
leaves the lockout.

**The kettlebell family** (`kettlebell.py`, 11): deadlift, clean, single-arm
press, thruster, high pull, snatch, windmill, halo, front rack carry, farmer's
carry, Turkish get-up. Three things shape all of them and are written into the
module docstring: the mass hangs below and behind the handle, so a racked or
overhead bell is a lever rather than a weight over the hand; loading one side
makes a frontal-plane problem for the obliques and gluteus medius; and the
handle is thick, so grip ends the carries before the legs do. With the swing
and the goblet squat that is 14 kettlebell exercises.

**Pressing away from the bench** (`press_variants.py`, 4): push press, Arnold
press, close-grip push-up, overhead triceps extension. The last is there
because the long head of triceps crosses the shoulder, so a pushdown cannot
load it at length and an overhead extension can.

**Pulling** (into `upper_pull.py`, 4): wide-grip and neutral-grip pull-ups,
Pendlay row, inverted row. The neutral grip is the one brachialis and
brachioradialis are strongest in; the Pendlay row's dead stop on the floor is
what stops the trunk helping.

**Squats and hinges** (into `lower_body.py`, 5): box squat, pause squat,
deficit deadlift, rack pull, single-leg Romanian deadlift. Both new modules
reuse `_squat_phases` and `_deadlift_phases` rather than re-authoring the
pattern.

Two angles were caught by the validator rather than by eye: a neutral-grip
pull-up at 150 degrees of elbow flexion (the rig allows 145) and a single-leg
RDL at 35 degrees of hip extension (it allows 27). Both are in the definitions
at the allowed value.

`upper_pull.py` is now 499 lines and `lower_body.py` 493, which is why the
bench, press and kettlebell families went into modules of their own rather
than into `upper_push.py` at 453.

## 2026-09-15 (later still) — 24 more: calisthenics, yoga and stretches, and five poses the rig refused

The catalogue went from 93 to **117** in three new modules
(`calisthenics.py`, `yoga.py`, `stretches.py`) under two new categories, and
`tools/render_exercise_grid.py` was written because checking 117 exercises one
render at a time is not something anyone does twice: one scene, one GL
session, two frames per exercise, twelve to a contact sheet.

**`calisthenics.py`** (8): muscle-up, L-sit, pistol squat, archer push-up,
pike push-up, Nordic hamstring curl, hollow body hold, bench dip. The limit in
these is a position rather than a load, so each entry's notes say what the
easier version is.

**`yoga.py`** (8): chair, warrior II, triangle, tree, high lunge, downward
dog, cobra, cat-cow. Entered, held, released — so the work is isometric and
the prime mover is whatever holds the shape.

**`stretches.py`** (8): standing forward fold, standing quadriceps, wall calf
(both knee positions, because gastrocnemius crosses the knee and soleus does
not), overhead side bend, chest opener, supine knee-to-chest, supine
figure-four, supine spinal twist. These invert the catalogue's convention:
the muscle listed PRIMARY is the one being *lengthened*, so the heatmap
colours what the stretch is for. It is stated at the top of the module,
because a reader would otherwise take it for a bug.

### Five poses the rig would not hold, and what the measurements said

Every one of these was found by `render_exercise_demo --probe`, not by eye,
and fixed against a measured sweep rather than by nudging numbers.

**The Nordic curl is knee *extension*.** Kneeling upright is the prone body
pitched 90° head-up about the knee. As the body falls forward by φ the shins
stay on the floor, so the knee angle is 90 − φ: the hamstrings resist the knee
straightening, which is why it is written `knee_flex = -pitch`. At that exact
relation the shank still sloped 13°, leaving the knee 25 units off the floor,
so `_SHANK_ON_FLOOR = 13.0` is added at every pitch; the knee then sits at
11.0 with the ankle at 14.2 and does not move (56.9 → 58.7 in x across the
whole descent), which is what "the ankles are held" means.

**A pike push-up cannot be as piked as it looks.** With hands and feet both on
the floor and the legs straight, hip-to-toe is shorter than hand-to-hip
through a vertical arm, so the hips cannot rise past ~100 units: at 55° of
trunk pitch the feet floated 54 units, at 35° they sat at 3.5. Bending the
elbows then drops the shoulder 37 units (65.9 → 29.2), and since the body
hangs from the anchored hands the trunk has to steepen to 58° at the bottom or
the feet go 49 units through the floor.

**Wide hands sink the body.** The archer push-up re-used the push-up's
pitches (−20 top, −8 bottom) and put the feet 20 units under the mat: at
40-55° of shoulder abduction the shoulders sit lower, so the body needs far
less head-up tilt — −7 and 0. Foot height moves 2.7 units per degree of pitch
here, which is why this is not a thing to guess.

**A bench dip lifts the feet unless the hips extend.** The hands are fixed, so
pressing up raises the hips 45 units and the heels with them. Hip flexion of
88° at both ends put the feet at 50 at the top; 84° at the bottom and 63° at
the top holds them at 8.8 and 8.7. The measured sensitivity is 1.5-1.8 units
of foot height per degree of hip flexion.

**Cobra lifts the legs, not the chest.** The wrapper turns about the body
origin, which is at the head, so 40° of spinal extension swung the *lower*
body up: the feet measured 44.6 with the pelvis supposedly on the mat. Hip
flexion of a fifth of the extension angle puts them back (10.8), and it is
written as a function of the extension so the correction follows it rather
than being a magic number in one pose.

**Side-lying was measured and abandoned.** The open-book rotation was the
obvious thoracic drill, but on its side the body's frontal plane is vertical:
10° of shoulder abduction separated the two hands by 100 units and left the
underneath one 25 below the mat, and the rig's adduction limit (−31.5°) cannot
close that. Supine keeps abduction in the plane of the floor, so the pose
became a supine spinal twist, which measures clean (hands 25.3, feet 9.7).

The high lunge lost its straight back leg to the same kind of limit: hip
extension stops at 27°, so a straight back leg lands under the hip and the
toes go through the floor. A 35° back knee puts the back toes at 6.0 against
the front foot's 6.3 with the heel 23 up — which is the shape of the pose
anyway.

### Half the catalogue was being filmed from the wrong place

Rendering all 117 at once showed two things the per-exercise renders never
made obvious.

The gym's `side` preset sits at **+X**, and a prone or supine body's long axis
**is** X — so `camera="side"` on a lying exercise looks straight down the body
from the feet. Eighteen exercises were framed that way, the whole bench press
family among them; they now use `front` (+Z), which is the profile.

A loaded barbell lies along X as well, so a side camera puts a plate in the
lens. Eight standing barbell exercises — both deadlift variants, the rack
pull, the rows, the curl, the push press, the power clean — were rendered as
a black disc with a shin behind it. They now use `three_quarter`.

Both are now tests (`test_a_lying_body_is_never_filmed_from_the_foot_end`,
`test_a_barbell_is_never_between_the_camera_and_the_lifter`) so the next
exercise added cannot quietly repeat them.

The box squat's plyo box was at the origin, which is under the lifter's feet:
the render showed a man squatting while standing on his box. The lifter faces
+Z, so the box now sits at z = −38, behind them.

`tests/exercise` is green (62), the fast tier is green apart from the
long-standing `test_obj_groups_name_the_bodyparts3d_source_ids`, and
`docs/exercises.md` is regenerated at 117 exercises.

## 2026-09-15 (last) — What looking at all 117 actually found

`tools/render_exercise_grid.py` was written to make this possible at all: one
scene, one GL session, two frames per exercise, twelve to a contact sheet,
twenty sheets. Reviewing them found six real defects, four of them mine.

**The field-of-view fit was a no-op.** `Camera` caches its projection matrix
and only `set_aspect` invalidates it, so `camera.fov = …` rendered at the
previous field of view. Every figure in the first sweep was framed by its
look-at target alone, which is why standing exercises came out cropped at the
ribs while I congratulated myself on the framing.

**`close_grip_push_up` was authored standing.** It rendered as a man stood
upright with his arms in the air. It is prone now, hands anchored, on the
push-up's own measured pitches.

**`overhead_triceps_extension` was standing with hips and knees at 90 degrees**
— a man sitting on nothing, which the ground lock then folded onto the floor.
`SEATED_ON_BENCH` was already imported in that module and unused, which is
about as clear a clue as a file can leave. It is `orientation="seated"` now.

**A supine lifter's bar runs along Z.** Fixing the lying exercises to `front`
(+Z) fixed the profile but pointed the camera straight down the barbell: the
floor press rendered as two black discs over the torso. The nine supine
barbell exercises use `three_quarter`. The test now knows both cases.

**The box squat's box was at the origin**, which is under the lifter's feet.

**The spine DOFs turn vertebrae, not the body.** This is in CLAUDE.md — the
arm chains hang off `bodyRoot`, not the thoracic spine — and I authored four
poses as if it were not. Measured: `spine_flex`, `spine_lat_bend` and
`spine_rotation` at full range move `shoulder_R`, `hip_R`, `wrist_R` and
`knee_R` by **0.0 units**. Triangle pose rendered as a man standing upright
with his arms out, and the overhead side bend as a man standing upright with
one arm up. Both now lean with a wrapper `roll` about the hips (40 and −28
degrees; the shoulder moves from x = 25 to x = 110), with the spine DOF kept
at a third of its old value as the anatomical detail it actually is.

Cobra is the same problem without the same answer: its whole shape is spinal
extension with the pelvis on the floor, and a wrapper pitch about the hips
takes the legs down through the mat with it (measured: pitch −35 lifts the
shoulder from 29 to 66 and puts the feet at −88, and hip extension stops at
−27). See the module for what was done about it.

Six more were left for a second pass: the **side plank** looked flat, the
**lateral band walk** shipped no band, **battle ropes** and the two
**treadmills** had no equipment at all, the **hip thrust**'s bar looked wrong,
and the **kettlebell windmill** and **halo** looked understated. Every one of
those is settled below.

### The last four, and one pose that had to change

**Cobra is not a pose this rig has.** Its whole shape is spinal extension with
the pelvis on the mat, and the spine DOFs move no joint at all. Pitching about
the hips lifts the shoulder from 29 to 66 but puts the feet at −88, and hip
extension stops at −27. Pitching about the **knees** is the pose the rig does
have: the shins stay down (feet 7.6), the thighs and pelvis lift (hip 45.7,
knee 28.5) and the chest comes up — which is an upward-facing dog, so that is
what it is now called. The module says why cobra is absent.

**Triangle pose leans, and pays for it.** Rolling a body whose legs are rigid
with its pelvis lifts the far foot: a 25-degree roll on a 42-degree stance put
the back toes at 72. Adducting the back hip to −8 puts them back at 6.4
against the front foot's 7.9, at the cost of stance width. Measured, kept,
and written down as a trade rather than a fix.

**The rowing machine's athlete was never on it.** `base_position` does not
place a foot-anchored body: at base x of −40, −120, −190 and −240 the hip held
98.6–163.6 every time, because the ground lock re-anchors to the feet's own
rest position. The rower moves to the athlete (x = 150) instead, and the
camera to `three_quarter` — at `side` it sat 50 units from a body it was
supposed to be filming, which is why that tile was a close-up of a back.

**Cat-cow and the hollow body hold stay understated.** Both are shapes the
spine makes, and the spine moves vertebrae only. The poses are anatomically
right and the renders are honest about what the rig can show.

## 2026-09-15 (last, really) — The second pass over what the sheets flagged

Measuring the six left over turned four of them into defects, one into a
feature and one into a correction of my own reading.

**The hip thrust's bar was over the lifter's face.** At 45 degrees of shoulder
abduction with the elbow at 90 the hands fold across the chest, and the bar
rode at x = −68 against a hip at −4. Nearly straight arms at the sides
(abduction 18, elbow 10) put it at −6.8, sixteen units above the hip joint,
which is where a bar resting on the hip crease sits.

**The Bulgarian split squat's back foot was not on its bench.** The bench top
is at 50 and the body rises 50 units between the bottom and the top, so the
back knee has to straighten as it goes or the foot rides up with the hips:
at 95/70 degrees the rear ankle went 46.2 → 76.0. At 104/45, with the foot
flattened at the top, it holds 51.3 → 55, on the bench at both ends.

**`lunges.py` contained three identical copies of the Bulgarian split squat.**
Two were dead — the module-level name was simply rebound twice — and they came
in with the file split earlier today. 91 lines removed, and the unused imports
the split left in nine catalogue modules with them.

**The band walk now has a band.** A loop round the legs had nowhere to attach:
`EquipmentSpec.attach` offered hands and the room. `attach="knees"` is the
same geometry as `"hands"` one storey down — centred between the knee joints,
its axis along the line between them — and it is two tests.

**The treadmills and the battle ropes now have equipment.** `make_treadmill`
puts its belt top at the sole height the ground lock produces, with the
console at +Z, the direction the gym body faces. `make_battle_rope` trails a
waving, sagging rope from each hand; the first attempt merely offset cylinders
from one another and rendered a staircase, so each segment is turned onto the
curve's own tangent.

**The windmill was a forward hinge pretending to be a windmill.** Its
`spine_lat_bend=-22` moved nothing; a −22 degree wrapper roll about the hips
moves the bell from x = 9 to x = 73 and the hip from 12 to 42, which is the
sideways half the exercise is named for.

**The side plank was fine and I misread it.** Measured, the hips go 38.0 →
56.7 between the rest and the hold — an 18.7-unit lift. It looks flat in a
contact-sheet tile because the whole body is horizontal either way.

The halo stays as it is: with the bell centred between two grip points and the
far shoulder's adduction stopping at −31.5 degrees, both hands cannot get to
one side of the head, so the bell travels about 17 units instead of circling.
It is a small arc rather than a halo, and the rig has no way to make it more.

### The third pass, and one thing I got backwards

The regression probe over the twenty changed exercises flagged four more.

**The treadmill's belt was a plinth.** A 6-unit deck with the soles at the
floor buried the feet; 2 units is a belt.

**The close-grip push-up inherited the push-up's −20 top pitch but not its
hand width**, and came out 2.9 below the mat. −18 puts it at +2.8.

**The rowing machine was the right idea placed backwards.** Moving it to the
athlete's hips (x = 150) put the seat under them but the footplate 154 units
past their feet, because the athlete's hips sit at *greater* x than their
feet and the ergometer's own geometry runs the other way. Turned about
(`rotation_deg=(0, 180, 0)`) at x = 116, the footplate meets the measured
ankle at 56 and the seat sits under the hips at the finish. The probe still
flags the feet at 36 — they are on the footplate, which is what a rower is.

**The side plank's resting arm I made worse before I made it better.**
Swinging the supporting forearm to 90 degrees of flexion measured −13.5; my
first "fix" — leaving it abducted as in the hold — measured −23.5, because in
side-lying abduction drives the underneath arm straight into the floor. Seven
poses measured: the underneath arm cannot be got above the mat once the hips
are down (that shoulder sits about 5 units up and the arm is 60 long).
Tucked and adducted, elbow nearly shut, is the least of them at −8.3, and is
what an arm does when you lie on it.

### Closing the loop: the facts go where facts go, and a gap gets a test

Three tidying jobs after the third pass.

**The measured rig facts moved into `docs/exercise_animation.md`.** That file
is where `INTERFACE.md` says measured rig facts live, and this session's were
only in this log. Four went in: the spine DOFs move `shoulder_R`, `hip_R`,
`wrist_R` and `knee_R` by 0.0 units at full range; a lying body's long axis is
world X while a supine lifter's bar is world Z, so `side` and `front` are each
wrong for one of them; `base_position` does not place a foot-anchored body;
and rolling a body tips its legs with it.

**Two tests now guard the gap that let four exercises ship nothing.** The band
walk, the battle ropes and both treadmills validated, built playable clips and
placed the body correctly while the athlete mimed, and nothing caught it.
`test_an_exercise_that_names_an_implement_ships_one` maps each implement tag
to the kinds that satisfy it; `test_no_equipment_means_no_equipment` is the
other direction.

**The halo was measured properly rather than assumed.** Seven arm
configurations: the hands never come closer than 75 units apart — the
shoulders are 21 out on each side and adduction stops at −31.5 degrees —
except with both arms straight overhead, at 37, where the bell is on the
midline anyway. The bell's whole available excursion is about ±8 in x and 11
in z against the ±25 a halo needs. Cobra was replaced when the rig could not
show it; the halo is kept, because unlike cobra its muscles, cues and sources
are all still right and only the amplitude is short. The measurement is in the
module and in the rig doc.

## 2026-09-15 (later) — The Turkish get-up never lay down

Reported: the get-up looks wrong, it should start on the floor with the bell
in a hand on the ground. It should, and it did not: **`orientation` was never
set**, so the whole rep defaulted to `standing` while the setup text said "on
the back". The figure stood up throughout with the bell already overhead, and
there was no start position at all — the first phase was already "roll to the
elbow".

It is re-authored as eight phases and is the first thing in the catalogue to
use `Phase.orientation`: four supine, three standing, then back down. Four
things had to be measured.

**Supine, the arms rest 13 units above the mat.** The wrist is anterior to the
shoulder in the rest pose, and on your back anterior is up, so a spread-eagled
arm at zero flexion floats. Eight degrees of extension lays it down and puts
the bell's centre at 14.2 with an 11-unit radius — resting on the mat.

**The planted foot could not reach the floor at all.** At hip 75 / knee 95 it
floated 55 units in the set-up and was 8 below the mat by "to the hand". The
fold that lands it is `60 sin(hip − pitch) + 47 sin(hip − knee − pitch) = 0`:
knee 112 throughout, with the hip gaining whatever the trunk gains (45 → 75 →
90). The flat-foot rule then asks for 67 degrees of dorsiflexion against the
rig's 45, so the heel is down and the toes are up.

**The free leg swung through the floor** as the trunk came up — −41 units at
the elbow, −58 at the hand — until that hip flexed by the same angle the trunk
gained, which is the hinge rule.

**The half-kneel floated 60 units**, because it borrows the standing
orientation's base while sitting 55 below a standing figure; `Phase.position`
places it. Its back shin was in the air too: a 140-degree back knee folds the
shank up behind the thigh, and 60 lays it along the floor (knee 13.5, ankle
16.0, toes 6.2).

All eight phases probe clean and the rendered sheet shows the five positions.

### Then the same check over the whole catalogue

Comparing every definition's *words* with its `orientation` found one more:
**`arnold_press`** says "Seated" and has hips and knees at 90 in every phase,
authored `standing` with `anchor="feet"` — a man sitting on nothing that the
ground lock folds onto the floor, exactly the `overhead_triceps_extension`
fault. Seated now.

`tools/audit_exercise_placement.py` is the generalisation. `--probe` looks at
the feet and the hands only, and only against the exercise's own anchor, which
is why the get-up's planted foot could float while the probe read the other
one and called it fine. The audit walks **every joint pivot** and reports the
lowest per phase: through the floor, or nothing touching down.
