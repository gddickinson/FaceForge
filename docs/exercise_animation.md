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
border glides 20 units from its spinous processes; C7 and T1 are not
registered attachment bones (the rhomboid minor and descending trapezius
attach to T2 for now); the pull-up top's flexion axis is 18° off after the
grip lock adjusts abduction, because abduction also turns the hand.

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
