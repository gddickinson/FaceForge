# Exercise demonstrations: design and evidence

FaceForge can now demonstrate weight-training, conditioning and athletic
movements on the anatomical figure: the joints move through a rep, the working
muscles are coloured by how hard they work, and the tab describes which joints
are moving and how many degrees. This page records how it is built, what was
measured to build it, and what the numbers mean. The exercise list itself is in
[`exercises.md`](exercises.md) (generated from the catalogue).

## Where it lives

| piece | module |
|---|---|
| data model (definition, phase, muscle role, equipment) | `faceforge/exercise/model.py` |
| functional muscle groups → mesh names | `faceforge/exercise/muscle_groups.py` |
| pose authoring in degrees, foot-flat rule | `faceforge/exercise/pose_library.py` |
| activation model and time track | `faceforge/exercise/activation.py` |
| joint-motion description | `faceforge/exercise/motion_description.py` |
| definition → clip + phase spans + track | `faceforge/exercise/clip_builder.py` |
| the exercises, by category | `faceforge/exercise/catalog/*.py` |
| gym equipment and its per-frame placement | `faceforge/exercise/equipment.py`, `equipment_rig.py` |
| feet/hands kept on the floor | `faceforge/body/ground_contact.py` |
| runtime (player + lock + rig + heatmap), no Qt | `faceforge/exercise/runtime.py` |
| app controller and tab | `faceforge/controllers/exercise.py`, `faceforge/ui/tabs/exercise_tab.py` |
| shared DOF range table | `faceforge/body/dof_ranges.py` |
| headless demo renderer / probe | `tools/render_exercise_demo.py` |
| catalogue → markdown | `tools/export_exercise_docs.py` |

## What was measured about the rig before authoring anything

The poses are authored against the rig as it actually behaves, which was
probed on the real BodyParts3D skeleton (`tools/headless_loader.py`) rather
than read off the code comments. Body frame: **+Z superior, −Y anterior, +X
right**. Rest pivot positions (units; the figure is ~226 units tall):

| pivot | R side |
|---|---|
| shoulder | (21.4, 4.0, −15.1) |
| elbow | (34.4, 5.0, −53.3) |
| wrist | (39.5, −5.3, −83.3) |
| hip | (12.0, 0.2, −81.0) |
| knee | (9.7, 3.8, −141.2) |
| ankle | (9.1, 5.2, −188.0) |
| 3rd metatarsal | (15.8, −8.5, −193.6) |

Three things the probe found, all now fixed or worked around:

1. **Abduction and axial rotation were swapped.** The Python port kept the
   JavaScript original's Y-up axis assignments (abduction about Z, rotation
   about Y). In this Z-up frame that made the "abduct" slider spin the limb
   about its own length (`shoulder_r_abduct=1` moved the wrist 9 units) and
   the "rotate" slider swing it sideways (`shoulder_r_rotate=1` moved it 76
   units medially). `BodyAnimationSystem._apply_limbs` now rotates abduction
   about Y (the anterior–posterior axis) and axial rotation about Z; the same
   correction applies to the hips, the scapular rhythm, ankle inversion, wrist
   deviation / forearm rotation and finger spread. Positive values mean
   flexion, abduction, external rotation, dorsiflexion, supination.
   `tests/body/test_dof_ranges_and_limb_axes.py` pins every axis.
2. **The arms hang off the pelvis root, not the thoracic spine.** `spine_flex`
   bends the vertebrae (and what is skinned to them) but does not carry the
   shoulders, arms or head. Trunk inclination for exercises is therefore a
   **whole-body pitch** of the scene wrapper, with the hips flexed by the
   same amount so the thighs stay where they were (a hip hinge is exactly
   that). Curls of the trunk (sit-up, crunch, bridge) pitch about a chosen
   pivot — the hips, the lower chest, the shoulders — using `Phase.pivot`.
   Reparenting the shoulder girdle under T1 would be the proper fix; it is
   noted below as future work.
3. **Skinned muscles flew off the body in scene mode.** The soft-tissue
   skinning cancels the scene wrapper's transform when it computes joint
   deltas, but its correction passes (bone-offset projection, hull bound,
   superior envelope, static-vertex mask) read joint world matrices directly,
   so they compared body-frame vertices with world-frame bones. Measured on a
   standing body in the gym: a thigh muscle's centroid was 8.5 units from the
   femur before the first skinning update and 146 after. Every joint read now
   goes through one wrapper-cancelling helper (`SoftTissueSkinning._joint_world`),
   and the same probe reads 8.5 standing and 9.0 at the bottom of a squat.
   This affected the existing dance-studio and examination scenes too.

The joint limits were widened for gym ranges: shoulder flexion and abduction
to 2.0 (180°), hip flexion to 1.4 (126°), shoulder extension to −0.7 (63°),
shoulder axial rotation to ±1.3 (91°, the external rotation a bar on the back
needs). The 126° hip cap plus the rig's short arms (upper arm 38 + forearm 30
units against a 107-unit leg) mean the deadlift start leaves the bar a little
above the floor; the proportions are BodyParts3D's, not a pose error.
The Body tab sliders still run −1..1; clips may exceed that.

### The foot-flat rule

With trunk pitch `φ`, hip flexion `θh` and knee flexion `θk` (degrees), the
sole is flat when the ankle is dorsiflexed by `φ − θh + θk`. Every standing
pose in the catalogue is built with `pose_library.squat()` / `hinge()`, which
apply it. A deep squat with a 35° trunk lean, hips 120°, knees 120° needs 35°
of dorsiflexion — the range deep squatting is known to require.

### Keeping the feet on the floor

The rig's root is the pelvis, so flexing the legs lifts the feet and pitching
the body swings them metres away. `GroundLock` measures, after every scene
update, where the lowest foot pivot (or the wrists, for push-ups, planks,
dips and pull-ups) ended up and translates the wrapper by the difference from
its target; the target is the pivot's **rest** height under the base
placement (feet), the floor plus the palm's thickness (hands), or an explicit
point such as a bar. One translation, no accumulation, converges in one frame.
`Phase.lift` (a jump) and `Phase.travel` (landing on a box) offset the target.

### Shoulder-region muscles under large arm movements (2026-09-10)

The first renders tore the latissimus, pectorals and rotator cuff into peaks
in the pull-up and the barbell squat. Measured per muscle with the edge
stretch report (`tools/render_exercise_demo.py`'s scene plus
`stretch_viz._compute_vertex_stretch`), the causes were:

1. **The headless path was not the app's.** `tools/headless_loader.py` kept a
   private chain builder that started the arm at the shoulder, while the app
   starts it at the clavicle and scapula; and it registered muscles without
   the attachment system (no origin/insertion pinning, no stretch clamp, no
   footprints, no lever damping). The rotator cuff was 85–100 % on the
   humerus and a latissimus edge reached 560× its rest length. Both are now
   shared code (`coordination/joint_chains.py`,
   `demand_loaders.register_muscle_layer`), pinned by
   `tests/coordination/test_joint_chains.py`.
2. **Latissimus dorsi listed the scapula as its origin**, so both ends hung
   off the arm. It now originates on T7–T12, the lower ribs and the hip bone
   (and the vertebra meshes are registered as attachment bones; they were
   hidden under the nested spine pivots).
3. **Only three muscles had attachment footprints, all right-sided.** The
   three were mirrored to the left (`tools/author_footprints.py mirror`,
   every vertex matched within 0.6 units), and footprints were seeded from
   bone proximity (`… seed`, origin radius 4, insertion radius 3) for the
   supraspinatus, infraspinatus, subscapularis, teres minor and major, the
   posterior deltoid, the abdominal pectoralis head and the latissimus, both
   sides. Seeding by proximity is what the repo's notes warn against for
   muscles that wrap the humerus; it works for these because their bellies
   lie on their origin bone. Kept only because it measured better on every
   listed muscle and regressed none.

| muscle, p99 edge stretch (× rest) | pull-up top before → after | dead hang | rack |
|---|---|---|---|
| Supraspinatus R | 78.7 → 6.1 | 20.2 → 21.0 | 63.7 → 12.5 |
| Subscapularis R | 24.8 → < 7 | 12.2 → < 9 | 24.2 → < 7 |
| Teres minor R | 21.6 → < 8 | 9.3 → < 9 | 20.8 → 8.2 |
| Teres major R | 15.9 → 9.6 | 25.6 → 20.1 | 15.3 → 11.8 |
| Posterior deltoid R | 13.6 → 5.3 | 13.8 → 11.4 | 15.1 → 9.3 |
| Pectoralis major, abdominal R | 12.2 → 6.1 | 25.2 → 12.3 | 27.5 → 9.7 |
| Latissimus dorsi R | — | 13.1 → 8.9 (over-stretched vertices 9 850 → 5 974) | — |

"before" is the app-equivalent path with girdle joints; what remains at the
dead hang is the 110° of glenohumeral rotation that a 165° abduction asks of
these short muscles.

### Sagging arms, bowed lats, spikes, and bars through hands (2026-09-10, later)

The second review found the upper-arm muscles hanging below the humerus in
the back squat, the lats bowing away from the trunk as a loop at the dead
hang, spikes elsewhere, and bars passing through the hands. Each had one
measured cause:

1. **The stretch clamp dragged muscles off moving bones.** It measured a
   muscle's "length" along the mesh's anterior–posterior extent and, when
   that meaningless ratio exceeded 1.35, blended the whole mesh halfway back
   to its rest position in space. At the back-squat rack pose the biceps sat a
   median 7.2 units below the humerus with it and 0.4 without. Legacy
   Y-extent pinning toward a bone centroid's translation tore the
   non-footprinted arm muscles as well (triceps medial head p99 1.30× →
   5.90×). The clamp is measurement-only now and the legacy pinning is gone.
2. **Rigid-image blending bows a belly.** A footprinted muscle was placed by
   blending the rigid images of its two joints; a belly vertex 40 units below
   the shoulder has a 40-unit arc as its humerus image, and the half-weight
   blend of that arc is the loop. `anatomy/fibre_field.py` replaces it with
   harmonic interpolation: the footprints move rigidly with their bones and
   the belly solves Laplace's equation between them, so it stretches along
   its own length and never bows. Because each footprint moves rigidly the
   per-frame cost is two small matrix products; the eight bind-time solves
   are disk-cached. Proximity-seeded footprints are far larger than real
   attachments (a deltoid "origin" of 35 % of its vertices, a teres major
   "origin" of 57 %), so each set is trimmed to its far end
   (`trim_footprints`) and the passes that pull a vertex toward its own
   joint's rigid image — bone-offset projection, superior envelope, hull
   bound, the balloon solve, capsule collision — are skipped for these
   muscles (measured: the balloon moved the deltoid 19 units off the field,
   collision push-out took its worst edge from 17× to 59×).
3. **Every collision capsule was in the wrong place.** Bone meshes are
   reparented under joint pivots, so their vertex arrays are pivot-local; the
   capsules were built from them as if they were world positions and all
   twelve clustered near the neck. A phantom humerus displaced ~3,000
   vertices of every deep neck muscle up to 3 units at the *neutral* pose
   (semispinalis stretch p99 9.3× at rest, 26 of 169 muscles non-unity). The
   capsules now live in the bone's local frame, follow it each frame, and
   resolve only penetration beyond a vertex's rest depth: 0 of 169 muscles
   deviate at rest.
4. **Hands could not close.** Digit pivots sat at bone centroids (a phalanx
   rotating about its middle opens the joint) and 90° of curl was shared
   over four joints. Pivots are at each phalanx's proximal end, each joint
   has its own maximum (MCP 90°, PIP 100°, DIP 60°; functional grasp is about
   60/60/40 after Hume et al., J Hand Surg 1990), `grip()` closes the hand
   to an 85° MCP, and the equipment rig puts a bar's axis through the
   centroid of the closed finger joints instead of a wrist offset.

| muscle, p99 edge stretch (× rest) | pull-up top before → after | dead hang | rack |
|---|---|---|---|
| Teres major R | 4.0 → 2.0 | 8.6 (max 58) → 4.3 (max 7) | 4.5 → 2.2 |
| Teres minor R | < 3 → < 2 | 5.7 → 3.4 | 5.0 → 2.2 |
| Deltoid, clavicular R | 2.2 → 1.8 | 7.3 → 4.3 | 3.9 → 2.9 |
| Deltoid, acromial R | < 3 → 1.8 | 7.6 → 3.4 | 4.9 → 2.7 |
| Pectoralis major, sternal R | 3.0 → < 1.7 | 6.6 → < 2.8 | 11.3 → 2.2 |
| Pectoralis major, abdominal R | 3.2 → 1.7 | 8.8 → < 2.8 | 5.5 → 2.1 |
| Latissimus dorsi R | 2.2 → 1.7 | 5.2 (bowed) → 2.8 (straight) | — |
| Semispinalis cervicis L (rest) | 9.3 → 1.0 | 9.3 → 1.0 | 9.3 → 1.0 |

Upper arm at the rack pose, residual from the humerus' rigid image (median
over the muscle, units): biceps short head 7.2 → 0.0, biceps long head 7.3 →
0.1, triceps long head 2.9 → 0.0, coracobrachialis 0.0 → 0.0.

The deformation gate (`tools/deformation_quality.py`, back and arm muscles
at 90° shoulder flexion) went from seam p99 37.65 / max 665 to 0.19 / 257
with containment still exactly 0; bulk p99 rose 0.090 → 0.26 because bellies
now stretch instead of tearing, and the thresholds were re-ratcheted from
that measurement.

What remained after that round is covered by the next section.

### Grips on bars, a winged scapula, and hands that slid (2026-09-10, third pass)

The third review: in the back squat the bar passed between the fingers; in
the bench press the hands were supinated; at the pull-up's dead hang
something still stood out from the back; and the hands slid along the bar
during the pull. Measured causes and fixes:

1. **The "wing" was the scapula, not the lats.** At 165° of abduction the
   scapulohumeral rhythm rotated the blade about an anterior–posterior axis
   through its centroid, which put the inferior angle at x = 28, seven units
   outside the ribcage's lateral extent; teres major, infraspinatus and
   subscapularis (2,200 / 1,300 / 900 vertices displaced outward by more
   than 8 units) followed it as a wing. The latissimus had 0. The blade now
   rotates about the thorax's surface normal at the scapula, so the inferior
   angle glides laterally and forward round the ribcage (r = 21.1 at rest
   and at 165°, was 28.3), the clavicle elevates 30° at the sternoclavicular
   joint and carries the acromion up 11.5 units (was 3), and the
   acromioclavicular joint stays together. Outward displacement of the three
   muscles after: 0. The spine-to-scapula muscles then needed attachment
   footprints of their own (trapezius parts, rhomboids; their config
   origins had been the scapula itself and the first ribs): rhomboid minor
   11.1× → 3.7×, transverse trapezius 6.2× → off the list.
2. **Palms could not face a bar.** The wrist quaternion composed
   flexion, deviation and pronation as XYZ, which applies pronation *first*:
   the flexion axis stayed fixed in the forearm, so extending a pronated
   wrist acted as deviation. Pronation is now the outermost rotation and the
   fingers' flexion axis turns with it (`tests/body/test_hand_grip.py`).
3. **Grips were authored by measuring the hand frame.** A probe reads the
   wrist pivot's frame (X = finger flexion axis, −Y = palm normal) — the
   digit pivots carry the grip curl and are useless for this — and scores a
   pose by the angle between the flexion axis and the bar, the palm's angle
   to where the bar is, and the hand's position. Back squat: elbows out and
   down, forearms up to a bar 16 units behind the neck, flexion axis 0.9°
   from the bar, palm 21° from vertical (the earlier pose had the forearms
   nearly parallel to the bar). Bench press: arms abducted 75°, elbows 80°,
   forearm pronated with the wrist extended 70°, flexion axis 15° / 13° from
   the bar at the bottom / lockout, palm 19° from the bar. Pull-up: humerus
   externally rotated 90°, forearm pronated 90°, flexion axis 10° from the
   bar at the hang.
4. **Hands stayed put.** `exercise/grip_lock.py` re-poses the skeleton after
   each frame's joint angles and solves the two shoulder abductions so each
   hand's offset from the trunk along the bar holds the value calibrated at
   the first frame (a 2×2 finite-difference Newton step, twice). Grip width
   through the pull-up: 54.5–68.7 before, 55.1–55.1 after. It is armed only
   for exercises whose hands are anchored to a point.

Still open: the deltoids remain over-stretched at full elevation; the
rhomboid major reaches 5.5× at the dead hang because the scapula's medial
border glides 20 units from its spinous processes; the pull-up top's flexion
axis is 18° off after the grip lock adjusts abduction, because abduction also
turns the hand. (C7 and T1 *are* registered attachment bones as of
2026-09-11 — see the neck-muscle section below.)

### Neck muscles that distorted during exercise (2026-09-11)

The neck muscles do not go through the soft-tissue skinning. They have their
own path in `anatomy/neck_muscles.py`: a per-vertex slerp of the head
quaternion, plus a *body delta* — how far the skeleton below them has moved
since rest — blended into the lower vertices. Three defects in that path
were measured, all in the same report of "the neck distorts during
exercise". `tools/neck_deformation_quality.py` reproduces every number
below; the figures are `results/neck_frame_diagnosis.png` and
`results/neck_attachment_drag.png`.

1. **The body anchors were read in the wrong frame.** The rest anchors are
   snapshotted at load, before scene mode exists. The current ones came from
   `SceneNode.get_world_position()`, which inside the gym includes the
   `scene_wrapper` (Rx(−90°) at Y = 203). One frame into a bodyweight squat
   the thoracic anchor read (0.225, 192.088, −5.661) against a rest of
   (0.225, 5.661, −10.912): a 186-unit delta. Every neck muscle was dragged
   ~153 units with a 99th-percentile edge stretch of **63×**, at the neutral
   pose, purely from entering the room. This is the third instance of the
   same frame mismatch (after the skinning's correction passes and the bone
   registry), so the cancellation now lives in one place,
   `coordination/body_anchors.py`, and `Simulation.frame_cancel()` settles
   it at step 9.5 — before the neck muscles, the neck pinning and the
   platysma read a pivot, rather than at step 12 where the skinning used to
   set it.

2. **Ten muscles named attachment bones that did not exist.** The config
   said `"Thoracic Vertebra T1"`; the scene node is `T1`. Worse, T1 hangs
   off the *cervical* pivot chain, so the registry — which walked only the
   thoracic and lumbar groups — had never registered it at all. The whole
   cervical group is registered now, and `tests/anatomy/
   test_neck_attachment_config.py` checks every `lowerBones` entry against
   the skeleton configs so a name that matches nothing fails loudly.

3. **A muscle followed its *region* even when it named its bone.** At full
   thoracic flexion the top thoracic pivot travels 4.58 units while T1 does
   not move at all, because the cervical chain that carries T1 hangs off
   `bodyRoot`. Fourteen of the 38 neck muscles were therefore dragged 3.76
   units by a thorax most of them are not attached to — including the six
   suboccipitals, whose origins are on C1 and C2. The body delta is now the
   muscle's own attachment-bone displacement when it names one, and the
   regional average only for muscles that name none (there are none left).
   At full flexion: 14 muscles displaced → 4, worst 99th-percentile stretch
   2.522 → 1.712.

4. **A zero delta read as "nothing to do".** `update` early-exited when the
   head quaternion was unchanged and the body delta was zero. Returning to
   rest *is* a zero delta, so the frame that should have straightened the
   neck was the frame that was skipped: a neck bent by a sit-up stayed bent.
   The exit now compares the deltas the current vertex buffers were built
   from, so it still skips a genuinely static frame and no longer skips the
   frame that undoes the last one.

### Muscles that loaded onto the floor behind the skeleton (2026-09-11, later)

The exercise module enters the gym *first* and loads the muscle regions
after it, so every muscle the demonstration needs is registered while the
`scene_wrapper` is already up. All of them came out about 160 units away,
rotated a quarter turn: measured across four regions, all 138 registered
meshes landed in a different place depending on the order, by a median of
160.6 units and up to 203.0. Pronator Quadratus R loaded at its correct
body-frame centroid of (37.7, -3.4, -79.4) and the first skinning pass put
it at (37.7, 79.4, -206.4), which is exactly the wrapper's inverse applied
to its rest pose.

The cause was an aliasing bug one line wide. `SceneNode.update_world_matrix`
rewrites world matrices **in place** — deliberately, so cached
`(mesh, world_matrix)` tuples stay valid — and `np.asarray` on an array that
is already float64 hands back the same object. `SoftTissueSkinning._joint_world`
returned that object unchanged whenever there was no wrapper to cancel, which
is exactly the situation at load time, so every joint's *rest* matrix was a
live view of its node. The instant the body stood up in the gym, all 152 rest
matrices became the current world matrices: joints read 219 units from where
they were snapshotted. `_joint_delta` then cached the inverse of a world
matrix as if it were a rest matrix, and the attachment pinning carried each
muscle bodily to that image of its rest pose.

The figure is `results/muscle_load_order_offset.png`, which shows the arm
and thigh muscles flat on the floor behind the standing skeleton.

`_joint_world` now always returns a fresh array. Placement is byte-identical
between the two load orders for all 138 meshes, and two tests in
`tests/body/test_skinning_under_scene_wrapper.py` pin it: a joint's rest
matrix must not follow its node, and an unmoved joint's delta must be the
identity after the body enters a scene. Both fail on the old code. This also repaired the two `tests/ui/test_exercise_viewer_mode.py` failures
when that file runs first; they had been red in every order because the
viewer's muscles were not where the test looked for them. They remain
order-dependent — red when `tests/tools/test_deformation_quality.py` runs
before them — because the fixture waits a fixed one second for every muscle
layer to load rather than waiting on a condition.

Measured after all four, with `tools/neck_deformation_quality.py`: the gym
rows are now identical to the clinical rows, pose for pose. What is left is
a rig limitation, not a neck-muscle one — the cervical spine and skull hang
off `bodyRoot` rather than off the top of the thoracic chain, so thoracic
flexion moves T3 under a stationary head and the four longus colli that
originate there really are stretched (1.712 at full flexion, ~1.1 at a
sit-up's 30°). Head rotation itself still reaches 2.087 on the infrahyoids
at full pitch; that path was not touched.

### Skin that tore during exercises (2026-09-11, last)

Reported as "individual pixels either being left behind or attached to
movements of the different body parts". Measured with the new
`tools/skin_deformation_quality.py` over four poses, on the 791,729-vertex
skin and its 2,379,747 edges. Edge stretch is the metric because it is
invariant to the rigid rotation a limb legitimately undergoes; raw
displacement is not, and a foot vertex moving 154 units in the body frame
during a deadlift setup is correct, not a fault.

The skin was intact at rest, in the clinical view and in the gym at the
neutral pose, and tore the instant any joint rotated. Decomposing the
pipeline at a deadlift-style hip hinge, by torn edges (stretched past twice
their rest length):

| stage | 99th pct | worst | torn |
|---|---|---|---|
| rigid, primary joint only | 1.000 | 214 | 9,900 |
| two-joint linear blend | 1.766 | 109 | 18,850 |
| four-influence linear blend | 4.741 | 247 | 53,485 |
| four-influence dual-quaternion blend | 4.857 | 253 | 55,568 |
| engine output, after every correction pass | 4.879 | 253 | 56,125 |

So neither the dual-quaternion blend nor any correction pass was responsible
-- the corrections moved the offending vertices by 0.000 units. The tearing
arrived with the third and fourth influences, and it was not a question of
count: two influences tore 18,850 edges, three tore 58,841 and four tore
55,568. A third influence is *worse* than none at all.

The reason is that the influence cutoff was rank-based. Each skin vertex took
its four nearest bone segments with inverse-distance weights cut off at the
distance to the fifth, which is smooth but not local: for a limb vertex the
third and fourth segments are a whole joint further along the chain, so a
thigh vertex carried real weight on the ankle and a hip hinge pulled it two
ways at once.

`SoftTissueSkinning.INFLUENCE_CUTOFF_BAND` replaces that with a compact
support measured from the vertex's *nearest* segment: nothing more than three
model units further contributes, whatever its rank. A departing segment's
weight still reaches zero smoothly, which is the property the rank-based
cutoff existed to provide, but the set stays local. The band is additive
rather than a multiple of the nearest distance because a multiple collapses
to nothing where the skin lies on the bone -- measured with a 1.5x ratio, the
worst edge went from 253x to 1982x even as the 99th percentile improved.

Bracketed over the four poses, worst-pose 99th percentile and total torn
edges:

| cutoff | worst p99 | torn | worst seam p99 |
|---|---|---|---|
| rank-based (before) | 8.073 | 185,170 | 61.950 |
| band 1.5 | 1.763 | 66,749 | 60.811 |
| band 3.0 (shipped) | 1.954 | 70,092 | 56.054 |
| band 6.0 | 2.513 | 104,770 | 53.958 |
| two influences | 2.273 | 87,711 | 70.236 |

1.5 and 3.0 are within a few per cent on stretch; 3.0 is taken because it is
better on the seam tail in every pose and on the single worst edge, and a
seam is what reads on screen as a vertex stuck to the wrong limb. At the hip
hinge, vertices with a torn incident edge fall from 39,672 to 12,967 of
791,729 -- 5.01% to 1.64%. Containment stays at 0.000 throughout: nothing was
ever left behind by a joint that did not move, so "left behind" was the far
side of a torn edge. `results/skin_tearing.png` is the before and after.

**Rejected, with the measurements that rejected them.** Turning on
`DIFFUSE_WEIGHTS`, the existing heat-diffusion pass, halves the seam tail
(24.5 to 10.7 at the hip hinge) but raises the bulk tail 25-32%, raises torn
edges 43% and pushes the worst edge from 253x to 3179x -- it rebuilds the
influence set from the top four of a diffused field, which destroys the
compact support that kept the set stable between neighbours. Tightening
`SKIN_SPATIAL_LIMIT` from 25 to 12, to stop lateral abdomen skin binding to
the hand that hangs beside it, is worse still: seam p99 goes to 74/119/327
and the worst edge to 4225x, because a harder eligibility cut adds partition
boundaries rather than removing them.

### Torso skin that moved when the arms moved (2026-09-11, last)

Reported after the cutoff band went in. Measured by holding the trunk still
and abducting both arms, then asking how far trunk skin travels. The lower
trunk and the chest do not move at all. What moved was the back: 261 vertices
in the midline strip over the thoracic spinous processes, by up to 10.4 units,
plus the paraspinal and scapular region.

The paraspinal motion is correct and the project's own data says so: the
muscle layer assigns trapezius, rhomboids and latissimus dorsi the arm chain
in `MUSCLE_CHAIN_OVERRIDES`, so skin over them follows the shoulder girdle.
The midline strip does not — it lies over spinous processes that do not move.

Those vertices had `clavicle_R` as their primary joint. By Euclidean distance
they should not have: `thoracic_1` 8.78, `clavicle_R` 12.13, `rib_1` 13.52.
The chain ranking is geodesic, and the geodesic fields are *seeded* by
Euclidean radius -- a vertex within `SEED_RADIUS` of a bone is told its
geodesic distance to that chain equals its Euclidean one. That makes seeding a
contest between **superficial** bones rather than the right ones. The clavicle
and scapula are subcutaneous, the vertebral bodies are not, so skin 8.78 units
from its own vertebra was never a spine seed and measured its distance to the
spine the long way round, while the collar bone was one short hop away. It
came out with a quarter of its motion on the clavicle.

`SEED_FROM_OWNED_SKIN` seeds each chain from the skin it *owns* -- the
vertices whose nearest bone segment is one of its -- as well as from the
radius. Every seed this adds was measured to fall on skin no bone reaches
within the radius, so it is exactly the deep-tissue skin the old rule never
saw. Midline back skin under full abduction goes from 261 vertices moving up
to 10.4 units to zero moving at all, and the flank speckle below the scapula
goes with it (`results/skin_arm_follow.png`).

It is a trade, and the metrics say so plainly:

| | midline movers | squat worst edge | axilla seam p99 | torn, 4 poses |
|---|---|---|---|---|
| radius seeds only | 261 | 212.80 | 56.054 | 70,092 |
| plus owned skin | 0 | 100.57 | 69.974 | 77,434 |

The reported defect goes to zero and the single worst edge in a deep squat
halves; the cost is 10% more moderately torn edges overall and a quarter more
seam stretch in the axilla, which was already the worst region and stays so
either way. Restricting the new seeds to skin the radius rule leaves unseeded
changes nothing, measured -- the benefit and the cost are the same seeds.

Because the *rule* changed rather than a number, `skinning_cache.CACHE_VERSION`
moved to 5. A tunable would not have covered it, and a stale entry served the
old seeding while the measurement showed no change at all.

### The axilla, and 528 pieces of skin (2026-09-11, last)

The worst edges that survived the two fixes above all had the same shape: two
vertices a tenth of a unit apart, one bound to the trunk and one flying off
with the arm. The worst in the whole mesh at full abduction joined a pair on
the left flank, at (-18.9, 2.1, -68.5) and (-19.0, 2.1, -68.5). The first took
lumbar_1 to lumbar_4. The second took elbow_L 0.32, shoulder_L 0.27, wrist_L
0.22 and travelled 60 units. In the rest pose the arms hang against the trunk,
so the forearm is about 4 units from that skin and the lumbar spine about 19:
straight-line distance treats the gap between arm and waist as if it were
tissue, and every rule built on it inherits the mistake.

Two things were wrong, and both are about the *surface* rather than the bones.

**Ambiguous skin was seeding chains.** A vertex now seeds a chain only when
that chain is clearly the nearest -- `SEED_CONFIDENCE_MARGIN`, 1.5x the
runner-up. Skin in the band between arm and flank seeds nothing, and the
geodesic fields reach it by propagation instead: a few units from the trunk,
and from the arm only the long way over the shoulder. That is the distinction
Euclidean distance cannot make and surface distance can. Bracketed over four
poses by total torn edges and the axilla's seam tail: 1.00 gives 77,434 and
69.974, 1.25 gives 68,486 and 54.439, 1.50 gives 67,053 and 52.704, 2.00 gives
72,093 and 53.791, 3.00 gives 71,432 and 54.246.

**The skin is not one surface.** Measured: 791,729 vertices in 528 connected
components, 454 of them under 100 vertices, 28,201 vertices off the main one.
Dijkstra never reaches an island, so its geodesic distance is infinite and the
solve falls back to exactly the Euclidean measurement the geodesic pass exists
to overrule. The three worst edges at full abduction each joined an island
vertex to its intact neighbour. The islands are not coincident duplicates --
their nearest main-surface vertex is a median 1.03 units away -- so welding
does not stitch them. `GEODESIC_BRIDGE` joins them into the Dijkstra graph and
nowhere else; `edge_pairs`, which the stretch metrics and edge relaxation read
as real topology, never sees a bridge. Joining a patch at its few closest
contacts rather than vertex by vertex matters: gluing every island vertex to
whatever is nearest attached a patch on the lateral chest across the armpit to
the arm. Three contacts and eight give the same numbers to four significant
figures, so the choice sits on a plateau.

Together, over the four gate poses:

| | torn edges | axilla seam p99 | worst edge | hip hinge p99 |
|---|---|---|---|---|
| rank-based cutoff, radius seeds | 185,170 | 61.950 | 537.04 | 4.857 |
| local cutoff | 70,092 | 56.054 | 570.25 | 1.508 |
| + owned-skin seeds | 77,434 | 69.974 | 570.25 | 1.596 |
| + confident seeds, islands bridged | 62,280 | 48.066 | 301.64 | 1.461 |

At the hip hinge, vertices with a torn incident edge are 11,535 of 791,729 --
1.46%, against 5.01% before any of this. Containment stays 0.000 throughout.

**What is left: the anterolateral spikes.** Reported as skin on the front of
the torso drawn into long spikes when the arms come up to shoulder height --
which is what full abduction is here, since the DOF's range is 90 degrees.
`tools/skin_deformation_quality.py` counts them now, as vertices travelling
more than 5 units further than their mesh neighbours: 676 at shoulder height,
225 at the hip hinge, 166 at the deep squat, none at rest or with the trunk
alone moving. `results/skin_arm_spikes.png` draws each one from where it
starts to where it lands.

They are a narrow rim of trunk skin at the lateral silhouette, |x| 15.8 to
21.9 and z -17 to -74, about half a unit wide and separated from properly
bound trunk skin by a median 1.9 units. 91% sit on the main mesh component,
so they are not the island problem. They are bound almost entirely to the arm
-- the worst, at (21.3, -8.0, -67.8), holds elbow_R 0.39, shoulder_R 0.31 and
wrist_R 0.30, and travels 74 units while the skin around it travels 25.

The cause is that below the rib cage there is no trunk bone within reach. The
ribs stop at z = -51, and for a vertex at z = -67.8 both the chain Z margin
and the proportional spatial limit exclude them: 22 of 152 segments remain
eligible, and the nearest is the elbow at a hybrid distance of 36.6 against
the lumbar spine's 48.3. The lumbar spine loses on *geodesic* distance, not
Euclidean, because flank skin lying against the hanging forearm seeds the arm
chain and shortens its field across the whole flank. The abdominal wall is
what actually fills that space, which is why binding skin by proximity to
muscle -- where the obliques are chained to spine and ribs -- is the shape of
the answer, and nearest-bone binding is not.

**Rejected here, with the measurements.** Bootstrapping chain ownership from
bone in contact with skin (`SEED_CONTACT_RADIUS`, left in place and disabled)
is the right idea and does not work: at 1.5 units it improves every seam tail
(hip 17.889 to 15.683, squat 11.122 to 10.012, axilla 48.066 to 44.985) but
raises total torn edges from 62,280 to 64,317 and the worst edge at shoulder
height from 301.64 to 469.48, while the spikes it was aimed at go from 88 to
70 above 10 units. Unioning the radius seeds back in is worse again at 68,309
torn. At 2.5 units the spikes are 68 and the count above 2 units rises. The
island bridge is not implicated: with it off the spike count is 549 against
552.

### Reach, and flesh (2026-09-11, last)

Two more causes, found by asking what the spikes were actually short of.

**The rib cage could not reach.** A chain's spatial limit is proportional to
its size, and size was measured as vertical extent. The rib cage is 41 units
tall, so the rule gave it 10.35 units of reach, floored at 12; the arm chain,
80 units tall, got 20. Flank skin sits a median 12.4 units from the ribs and
14.6 from the arm, so for 482 of the 700 spikes a trunk segment was already
*nearer* than any arm segment and had been masked out while the further one
was kept. The floor is 16 now, bracketed over the four gate poses:

| floor | spikes at shoulder height | axilla seam | total torn |
|---|---|---|---|
| 12 | 676 | 48.066 | 62,280 |
| 14 | 434 | 28.704 | 60,747 |
| 16 | 329 | 20.633 | 59,946 |
| 18 | 325 | 20.060 | 60,049 |
| 24 | 430 | 24.768 | 61,155 |

Measuring chain size by bounding-box diagonal instead, so that a short wide
structure is not called small, was tried and is redundant once the floor is
right: 329 spikes by height against 356 by diagonal, because the diagonal also
widens chains that did not need it.

**Bone is the wrong thing to be near.** The remaining spikes sit where no bone
can tell the trunk from the arm, because in the rest pose the arm hangs
against it. Muscle can: it fills the soft tissue, so the flesh nearest a patch
of skin is the flesh that skin sits on. Measured on the spikes that survived
everything else, the nearest muscle is a trunk muscle for 78% of them while
their nearest bone says arm. `body/muscle_field.py` holds a distance per body
part, sampled from the muscle meshes by `tools/build_muscle_field.py` into
`assets/config/muscle_field.npz`, and the skin binding adds it to the bone
distance. Weighted equally with bone, bracketed by spikes / that pose's torn
edges / total torn:

| weight | spikes | arms torn | total torn |
|---|---|---|---|
| 0.0 | 329 | 11,456 | 59,946 |
| 0.6 | 306 | 9,543 | 57,179 |
| 1.0 | 297 | 8,712 | 56,279 |
| 2.0 | 261 | 7,843 | 57,992 |

The sample is deliberately coarse: stride 7 gives 1.13 M points and 9.6 MB,
stride 60 gives 158 k points and 1.2 MB, and they agree to within noise
(56,279 torn against 55,971), because the question is which body part's flesh
is nearest and muscles are large.

Where the whole effort has got to, over the four gate poses:

| | torn edges | spikes | axilla seam | worst edge |
|---|---|---|---|---|
| rank-based cutoff, radius seeds | 185,170 | — | 61.950 | 537.04 |
| local cutoff | 70,092 | — | 56.054 | 570.25 |
| confident seeds, islands bridged | 62,280 | 676 | 48.066 | 301.64 |
| reach and flesh | 55,971 | 291 | 17.835 | 203.89 |

`tools/skin_defect_views.py` draws any of it from three viewpoints, before and
after: `results/skin_spike_views.png`, `skin_stretch_views.png`,
`skin_armweight_views.png`, `skin_hiphinge_views.png`. The arm-weight view is
the clearest — before, arm-driven skin trails down the chest and flank from
both armpits; after, it stops at the arm.

### What the renderer actually shows (2026-09-12)

Every measurement above is arithmetic on vertex positions.
`tools/render_skin_proof.py` draws pixels instead: the app's own scene in
front of `Session`, so the same GL renderer, framebuffer and blank-frame
guard the headless CLI uses. It is worth doing, because the pixels and the
metrics do not agree about what was gained.

Arms at shoulder height, lit pixels before and after, and the stray skin that
went away:

| view | before | after | removed | added |
|---|---|---|---|---|
| front | 133,270 | 111,614 | 24,988 | 3,332 |
| three-quarter | 117,462 | 101,901 | 18,075 | 2,514 |
| back | 132,235 | 111,310 | 23,835 | 2,910 |
| side | 58,957 | 58,915 | 78 | 36 |

So the artefact loses about a sixth of the lit area from the front and the
back, and nothing from the side, where the sheets of stretched triangles are
edge-on. `results/skin_render_proof.png` shows it: two dense wings from the
armpits to the hips before, much thinner wings after. **Thinner, not gone.**
The spike count fell 676 to 291, but each spike vertex drags a fan of
triangles, so the count understates how much screen it covers; the render is
the better evidence and it says the defect is reduced rather than solved.

The hip hinge is the opposite case. Torn edges there fell from 22,638 to
8,817, and the render barely changes -- 81 pixels of 110,000 from the front,
515 from the side (`results/skin_render_proof_hip.png`). That tearing is
inside the silhouette, so it shows as shading rather than as stray geometry.

The control is the neutral pose, where the binding changes should do nothing
at all: before and after are pixel-identical, 0 of 990,000 differing, in both
the front and side views. Whatever the changes do, they do it only to
deformation.

### Driven by the render (2026-09-12)

With the renderer as the arbiter rather than the metrics, four more changes
were tried. Stray skin at shoulder height, in lit pixels, is the score:

| change | front | back | verdict |
|---|---|---|---|
| starting point | 111,614 | 111,310 | |
| re-bind in a separated pose | — | — | rejected, far worse |
| discount influences whose flesh is far | 111,447 | 111,241 | rejected, inert |
| flesh grants chain eligibility | 105,856 | 105,164 | rejected, noise |
| **bone must lie inward of the skin** | **106,443** | **105,479** | **kept** |

**Re-binding in a separated pose** is the standard remedy for limbs that rest
against the trunk, and it fails here for a reason worth keeping: the skin has
to be deformed into that pose before it can be re-solved there, and the only
thing available to deform it is the rest-pose binding whose mistakes are the
problem. Round one drags the flank out along the arm; round two finds those
vertices beside the arm and binds them to it harder. Spikes went 291 to 1,237
and the shoulder-height seam tail 17.8 to 504.0. Disabling the muscle field
for the second solve, in case a rest-pose field against a deformed skin was
the cause, changed nothing.

**Discounting influences by flesh distance** improves every seam tail (17.835
to 11.039 at shoulder height) and does not move the picture: 167 pixels of
111,614. The reason is worth recording, because it also says what *would*
work: the vertices drawn into wings hold all four influences on arm joints --
elbow 0.39, shoulder 0.31, wrist 0.30 -- so there is no trunk share to shift
weight toward. The set has to change, not the shares.

**The inward test** changes the set, and is the only thing that separates the
flank from the forearm hanging beside it. Both bones are close and both have
flesh close, so no distance decides it; direction does. From flank skin the
abdominal wall is inward and medial, the forearm outward and lateral across a
gap; from the inner surface of the forearm it is the other way round. The
inward direction comes from the muscle field rather than the mesh normals,
because this asset's winding is inconsistent and 49% of its normals point the
wrong way. A chain may seed the geodesic field only where its bone lies on the
flesh's side. Torn edges 55,971 to 55,445, spikes 669 to 587, and 5,171 more
stray pixels off the front and 5,831 off the back.

Where the rendering has got to, at shoulder height:

| view | before all of this | now | removed | reduction |
|---|---|---|---|---|
| front | 133,270 | 106,443 | 29,128 | 20.1% |
| three-quarter | 117,462 | 98,603 | 20,671 | 16.1% |
| back | 132,235 | 105,479 | 28,978 | 20.2% |

The control holds throughout: at the neutral pose the render is
pixel-identical before and after, 0 of 990,000 in all three views.

### Chasing the wings to the asset (2026-09-12, later)

Two more changes were kept and five rejected, each judged on pixels.

**Kept: eligibility judged across the surface, not in a straight line.** The
spatial limit and the ranking beside it were measuring different things, and
they disagree exactly where a limb lies against the trunk. Measured on a flank
vertex: the rib cage 17.27 away in a straight line and 17.78 across the skin,
the arm 17.16 and 34.29. The ranking had it right and never got the chance --
the ribs were 1.27 over their chain's limit and were masked, the arm was
inside its own and was kept, and the vertex came out entirely arm-driven. The
limits scale by 1.9 to account for a surface path being longer, bracketed at
1.6 / 1.8 / 1.9 / 2.2 / 2.8. Stray skin: 6,813 pixels off the front.

**Kept: a chain seeds only skin sitting on its own flesh.** The direction test
is not enough beside a hanging arm, because chest skin further out than the
humerus has the humerus inward of it. The flesh answers outright: of the 123
vertices still drawn into flaps, 121 sat on trunk flesh a median 1.33 units
away with arm flesh 4.49 away. At shoulder height the worst edge falls from
213.66 to 59.69 and torn edges from 8,021 to 4,958.

**Rejected, with the measurements.** A wider influence band trades 60% more
torn edges for no gain at the shoulder. A support proportional to the nearest
segment fixes hip spikes and takes the deep squat from 22,505 torn edges to
53,020. Weighting the flesh term more heavily (2.5) doubles the spikes.
Grouping the glenohumeral muscles with the arm improves the spike count and
costs the worst edge, 59.69 to 184.79, for 135 pixels of difference. A flat
price on crossing between body parts prices out the genuine transition at the
shoulder along with the welds: worst edge 16,987.

**Where it stops, and why.** The arm and the chest are separate sheets facing
each other across air below the armpit, and the surface path between them
should run up to the rim and back: measured, a median 36.74 units against a
straight line of 8.47. For 1,443 lateral-chest vertices it does not, because
the asset has the two sheets touching -- the shortest such path is 0.11 units.
Through those welds the arm's field reaches the chest whatever the binding
does. Cutting crossings below the shoulder catches every real boundary too
(hip, wrist, neck) and makes things worse. Telling a weld from a boundary
needs to see the surface fold back on itself, and this asset's winding is
inconsistent enough that the sign of a fold cannot be read.

So the wings are not gone. At shoulder height they are 28.7% smaller in
rendered area than where this started, and what is left is barely stretched --
the worst edge went 578 to 59.7 -- so it is a coherent sheet of chest skin
half-driven by the arm rather than a fan of torn triangles. Removing it needs
the asset repaired: consistent winding first, then cut the welds where the arm
rests against the body. That is a data fix, not an engine one, and it is the
same repair the surface fitting has been wanting.

The rest of the residual is where the skin genuinely folds -- the hip crease
at deep flexion and the buttock. Linear and dual-quaternion skinning cannot
represent a fold, so closing that needs a different deformation model rather
than a better binding.

## The animation model

A rep is a sequence of **phases**; each holds the pose reached at its end, how
long that takes, and the kind of contraction (`eccentric`, `concentric`,
`isometric`, `transition`). The existing keyframe player interpolates
between phase poses with the phase's easing (`ease_in_out` for controlled
lifts, `ease_out` for a hip snap, `ease_in` for a drop). Tempo follows
teaching convention — 2–3 s eccentric, 0.3–0.5 s pause, 1–1.5 s concentric —
and the tab's tempo control scales all of it. Cyclic movements (pedalling,
gait, rowing) are one cycle of `linear` phases sampled from published
joint-angle curves.

Every keyframe carries the **full** pose, because the player treats a missing
DOF as zero. While an exercise runs the live body state is written directly as
well as the target, removing the 0.25 s interpolation lag that would otherwise
damp a 2 Hz jump rope into a shuffle.

## The activation model (what the colours mean)

Each muscle group in a definition has a **role** and a **peak** level (as a
fraction of maximal voluntary contraction): primary movers default to 0.95,
synergists 0.6, stabilisers 0.3, unless the definition gives a sourced number.
The level in a phase is the peak scaled by the phase kind:

| kind | primary / synergist | stabiliser |
|---|---|---|
| concentric | 1.00 | 1.0 |
| isometric | 0.85 | 1.0 |
| eccentric | 0.75 | 1.0 |
| transition | 0.20 | 0.6 |

The eccentric factor comes from surface EMG being 7–31 % lower in eccentric
than in velocity-matched concentric actions (see the sources page). Per-phase
overrides let a rowing stroke peak the legs in the drive and the arms only at
the finish, and let cyclic movements switch sides. Levels ramp over the first
and last 15 % of each phase so the heatmap never snaps.

Colours: the existing blue→red ramp, or a **thermal** ramp (dark plum →
orange → yellow) that is ordered in luminance and readable with red–green
colour-vision deficiency. The tab lists each group with its live percentage
and the DiGiovine band (low ≤ 20 %, moderate ≤ 40, high ≤ 60, very high).

Two honest caveats. The levels are a stated model calibrated to published
EMG where it exists (bench press, pull-up, deadlift, squat, planks), not a
simulation; and muscle *length* change is not yet shown — the attachment
system computes a per-muscle stretch ratio each frame, and a "length change"
colour mode is the natural next step.

## Equipment

Procedural geometry from the scene primitives (plus a sphere and a torus):
barbell, dumbbell, kettlebell, medicine ball, bench (flat / incline), pull-up
bar, dip station, plyo box, mat, upright bike, rowing ergometer, jump rope,
cable handle, band. Two-handed items sit between the palms with their axis
along the line between the hands; one-handed items follow their wrist; static
items are placed in the room. The jump rope spins about the hand axis at
2 rev/s. Units: 1 ≈ 0.78 cm, so an Olympic bar is 280 long with 58-unit plates.

## The exercise viewer and implied stabilisers

The viewer is a *mode* of the main window (View ▸ Exercise viewer,
Ctrl+Shift+V; or `python -m faceforge.exercise_viewer`), not a second window:
a QOpenGLWidget can live in one place, so the right-hand control panel is
swapped for `ui/exercise_viewer.py`, which adopts the control panel's exercise
tab (the same instance, so the transport, status and progress plumbing keep
working) and adds camera buttons all round the gym, a whole-body toggle and a
way back. Entering the mode enters the gym scene, loads every muscle layer
(`muscle_groups.ALL_MUSCLE_REGIONS`, hands and feet included) and sets the
three-quarter view. The catalogue, playback and heatmap are the ones the
EXERCISE tab already had.

**Bones drawn outside the body (fixed).** The standalone viewer entered the
mode on `LOADING_COMPLETE`, which the skeleton pipeline publishes several
stages before body animation, the rib pivots, the skinning and the attachment
systems are wired. Two consequences: the exercise started with no body
animation (so no grip lock), and the skeleton was painted — its vertices
uploaded to the GPU — before `reparent_under_pivot` re-based each rib under
its breathing pivot by subtracting the centroid from the vertices in place.
The renderer re-streams only a mesh flagged `needs_update`, and nothing set
it, so every rib was drawn at pivot + original vertices: twice its distance
from the body origin, 40–50 units below and outside the trunk in the front
view (the projected centroid of the 10th rib sat 45 units above the rib as
drawn, while the muscles, uploaded after their first skinning, were where
their matrices said). `reparent_under_pivot` now flags the mesh, and the
standalone viewer enters the mode from the load sequence's `COMPLETE` stage
(`exercise_viewer.watch_load_sequence`). Rule: any code that edits
`geometry.positions` in place after load must set `mesh.needs_update`; the
`MeshInstance.positions` setter does it for whole-array assignment only.

**Per-frame cost with every muscle loaded.** 317 muscles are 7.9 million
vertices, and one `Simulation.step` cost 9.7 s (bone collision 3.4 s, of
which 3372 capsule-mesh distance passes almost all on meshes nowhere near
the bone; the DQS/LBS passes ~2.7 s; hull bound 1.2 s; `np.add.at` face-normal
sums 0.9 s). Three changes, none of which alters the output: the collision
pass measures only capsules whose box overlaps the mesh's, and only the
vertices inside that box (0.75 s, then 0.12 s once fewer muscles move);
face normals accumulate with `np.bincount` (0.31 s); and a muscle none of
whose driving joints moved since its last frame is skipped outright
(`soft_tissue.update`, per-binding driver deltas compared at 1e-9 — the
wrapper cancel is re-derived each frame so the same pose is not bit-identical).
The skinning signature also no longer includes the scene wrapper: the
output is a body-frame quantity, and the ground lock moves the wrapper every
frame, so a paused demonstration re-skinned everything for nothing. Measured
on the pull-up: a full-body recompute 9.7 s → 6.3 s; a frame in which the
legs and hips rest 9.7 s → 1.25 s. A grouped per-joint matmul was tried in
place of the `(V, 4, 4)` gather and was slower from two joints up
(`body/skinning_ops.py` records the numbers).

What is left is a memory-bandwidth floor: every pass over the moving
muscles' vertices costs 50–150 ms, and the correction passes are what make
the muscles look right, so they cannot move to the vertex shader. The next
step for an interactive viewer is a display level of detail — decimated
muscle meshes (trimesh 4.4 and open3d 0.18 are in the environment) with the
vertex-indexed data (footprint seeds, `skinning_overrides.json`, the fibre
field caches) remapped by position — which is a project of its own.

**Implied stabilisers.** A catalogue entry lists the movers. What the body
is also doing follows from the definition — what is held, what it is
anchored by, how it is oriented — and `exercise/stabilisers.py` adds those
groups at the stabiliser role when a clip is built, so they colour without
being authored: a held load implies the grip (forearm flexors and the hand
intrinsics at 0.55), the wrist extensors, the elbow flexors holding the
elbow, the deltoids, upper trapezius and rotator cuff carrying it, and the
trunk brace (erector spinae 0.45 down to quadratus lumborum 0.3); hanging
from a bar implies the grip at 0.6 with the scapular stabilisers; hands on
the floor a lighter grip with serratus anterior; standing the bodyweight
brace, hip abductors, ankle and foot muscles; a bar racked at the shoulders
the upper and middle trapezius, rhomboids and erectors. Authored groups are
never overridden. The hand and foot intrinsic muscles gained functional
groups for this (their config names carry the side as a prefix).

## Using it

**In the app**: Exercise tab → filter by category / equipment → Demonstrate.
The gym scene opens, the muscle regions the exercise colours are loaded, the
clip plays on the shared transport (pause, scrub, 0.25×–2×), and the tab shows
the phase, its cue, the joints moving with degrees, and the muscle list.

Each definition names a gym camera preset (`camera`) and, for bodies that are
not standing at the origin, a look-at point (`camera_target`: a hanging body
is viewed at Y≈190, a dipping one at Y≈130); the preset keeps its offset from
the target. The side presets look past the plates only for unloaded lifts —
a barbell's plates hide the lifter from a true side view, as they do in a gym.

**Headless**:

    python -m tools.render_exercise_demo --exercise barbell_back_squat --frames 36
    python -m tools.render_exercise_demo --probe --all      # placement check, no GL

The demo drives the very same `ExerciseRuntime` the app uses, through the
real GL renderer (`faceforge.session.Session`), and writes frames, a phase
contact sheet, an MP4 and a GIF under `results/exercise_demo/`.

**Adding an exercise**: write an `ExerciseDefinition` in the right catalogue
module using the `pose_library` helpers in degrees (see any neighbour). Two
helpers look alike and are not: `merge()` returns a **full** pose (neutral
plus what you give it) and is what a phase wants; `combine()` unions
**partial** fragments and is what an arm-and-grip bundle wants. Merging a
bundle that was itself built with `merge()` resets every other joint to zero
(the back squat and the lunges shipped that way for an afternoon; the probe
caught it). List
muscles by role with sourced peaks where you have them, cite the sources, and
run `pytest tests/exercise` — the catalogue tests validate every DOF against
the joint limits, every muscle group against the mesh configs, and build the
clip. `ExerciseDefinition.to_dict()/from_dict()` round-trip through JSON for
definitions kept outside the code. Then `python -m tools.export_exercise_docs`.

## Limitations and next steps

- The shoulder girdle and head do not follow spinal flexion (rig hierarchy).
  Reparenting the arm and clavicle chains under T1 would let crunches and
  rows show a bending thorax.
- No scapular elevation / protraction DOFs, so shrugs and the "shrug at the
  top" of a press are not animated (the trapezius is still coloured).
- The lock is a translation only: feet do not roll, and a foot that should
  slide (the trailing foot in a push-up, the seat on a rower) is approximated.
- Muscle length change ("extent of movement") is computed but not yet drawn.
- Fingers close with a fixed grip pose; the bar passes through the ring of
  closed finger joints but the fingers are not wrapped to its radius.
- The shoulder girdle glides on a cylinder-like thorax with a fixed 2:1
  rhythm; there is no scapular tilt, no elevation DOF for shrugs, and C7 / T1
  are not registered attachment bones.
