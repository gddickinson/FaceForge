# The sex morph, measured

What the morph between male and female does, what it used to do, and the
numbers that decided each choice.  The rig facts it relies on are in
`exercise_animation.md`; this file is about dimorphism only.

## What was wrong

Three defects were reported and all three reproduced.

**The skeleton came apart, and did not change proportion.**  Every bone was
scaled about its own centroid.  A femur scaled 0.92 that way pulls its distal
end 4 % of the bone's length toward the middle, and the tibia does the same at
its proximal end, so the knee opens by 4 % of *both*.  Measured at gender 1 on
the shipped configuration:

| joint | male | female |
|---|---|---|
| knee (femur/tibia) | 0.07 | 3.55 |
| elbow (humerus/radius) | 0.59 | 3.89 |
| patello-femoral | 0.36 | 1.51 |
| humero-ulnar | 0.05 | 0.84 |

Scaling about the centroid also leaves the centroid where it is, so no bone
ever moved: the median bone-centroid displacement across the whole male to
female change was **0.00**.  A shorter clavicle did not bring the shoulder in;
a wider pelvis did not carry the hip joints out.

**Nothing but the bones changed.**  The release path scaled the bones and then
*re-bound* the skinning, which makes the new skeleton the rest pose and sets
every delta back to the identity.  Measured through the full path, a muscle's
centroid, length and radius and the skin's every cross-section were identical
before and after: the bones shrank inside a body that did not move.

**The surface mesh was flattened, worst in the arms.**  The body-surface mesh
is warped onto the skeleton at load, and its last phase projected every vertex
onto the closest point of the BP3D skin.  The fit it reported was excellent
because it had crushed the mesh onto the target: 1235 triangles ended below a
tenth of their original area, 3034 hand edges and 727 forearm edges below half
their length, the worst at 0.4 %.  The hands and feet came out as flat blades.

## What it does now

### The skeleton is scaled as a hierarchy

`body/skeleton_morph.py`.  Bones are already reparented under joint pivots, so
each bone's vertices are expressed relative to the joint it hangs from.  A bone
is therefore scaled about its own **proximal joint**, and the child pivot's
offset -- the segment -- is scaled by the same factor, so the distal joint
lands on the end of the scaled bone and the next bone follows it.  Joints whose
position is absolute rather than an offset belong to a *region* and scale about
that region's anatomical anchor: the shoulder rides on the clavicle from the
sternoclavicular joint, the hip joints ride on the pelvis, the ribs on the
thoracic spine.

One articulation has neither bone hanging off the other -- the
acromioclavicular -- and drifted 1.25 units open; `body/skeleton_joints.py`
shuts it from a contact patch measured on the unscaled skeleton, using the
*change* in that patch's offset so that gender 0 restores exactly.

After: every joint's clearance is within 0.12 units of its male value, and
gender 0 returns the skeleton bit for bit.

### The soft tissue follows a warp, not the skinning

`body/skeleton_field.py`, `body/soft_tissue_morph.py`.  A change of proportion
is not a pose.  Routed through the articulated skinning it tears: joints
translate rather than rotate, two skin vertices either side of a chain boundary
follow different joints, and the cross-chain divergence clamp -- which exists
so a moving arm does not drag the trunk -- stops them being blended back
together.  Measured on the skin at gender 1: **27747 edges beyond twice their
rest length and 6666 below half**, and a body visibly shredded at the waist,
the hip and the shoulders.

So the joint displacements are interpolated over space instead, as a thin-plate
spline.  A spline rather than a weighted average because the field has to
*reproduce* the skeleton change rather than blur it -- a Gaussian average over
control points 10 units apart returned barely half of a control point's own
displacement at the control point itself.  The thin-plate spline interpolates
exactly and, carrying an explicit affine term, reproduces an affine change
exactly: scale a skeleton uniformly and the soft tissue scales uniformly with
it.  It is evaluated on a 2-unit lattice and interpolated (median error 0.005,
max 0.34 against the exact spline) because evaluating it per vertex over 6.2
million of them took 25 seconds.

### Muscles change bulk, not length

`body/muscle_morph.py`.  Length comes from the bones, which the warp already
gives them.  Girth does not: whole-body skeletal muscle mass is about two
thirds of the male value in females, and the cross-sectional-area ratios are
regional -- roughly 0.62 in the upper limb and shoulder girdle, 0.76 in the
lower limb, with the trunk between.  Each muscle is thinned perpendicular to
its own long axis, tapered to nothing at the two axial extremes where the
tendons and bony footprints are, so a belly thins and an attachment stays
attached.  A muscle's region is read from the kinematic chain most of its
vertices bind to, so nothing is named twice.

### The skin carries the soft-tissue difference

`body/skin_morph.py`.  The bones give the skin its size; they cannot give it a
breast, gluteal fat or a waist narrower relative to the hip.  That difference
is measured, not invented: the application already loads a matched male/female
surface pair, and once both are warped onto the skeleton the vector between
them is a sex difference with a shared topology.  Two things are removed from
it -- the uniform size change, which the skeleton already produces, and the
tangential component, keeping only what runs radially out from the body's own
axis, which is what a fat distribution is and cannot change a limb's length.

The residual that survives is 0.8 units over the head (there is essentially no
soft-tissue sex difference there, which is the right answer and a useful
check) against 4.2 at the chest, 5.2 at the waist and 4.5 over the hip.

Transferring it onto the model's much denser skin needed three guards, each
added against a measured failure: a Gaussian kernel rather than inverse
distance (the field latched onto single source vertices and jumped by up to 8
units across a 0.26-unit edge); a smooth confidence fade rather than a hard
distance cut-off (14606 vertices fell outside it and received nothing while
their neighbours received everything, so the mesh tore along the boundary); and
a final edge-length band, which makes "no tearing" a property rather than a
hope.

### Fitting the surface mesh onto the skeleton

The body-surface mesh and the skeleton are different bodies in different
poses, so the mesh is warped onto the skeleton at load.  How that is done is
the whole question, and the first two attempts both failed in ways worth
recording.

The version that shipped remapped the mesh piecewise in Z and blended two arm
rotations against that remap.  Blending a rotation against a translation is
not a rigid motion, and it sheared everything it touched:

| | before the warp | after it |
|---|---|---|
| forearm depth | 24.7 | 17.4 |
| forearm width | 21.5 | 28.4 |
| foot length | 28.2 | 19.2 |
| occiput | rounded | shaved flat |

The fix is in `body/surface_register.py`.  Each limb segment is matched on its
own -- rotated, scaled and moved onto its bone -- and the trunk and head are
matched to the reference body level by level.  Those correspondences are
sampled densely (a ring of points at five stations along every bone, which
pins the rotation) and interpolated by a spline.

Two things had to be added on top, each against a measured failure:

* The spline is a *global* interpolant, so one inconsistent correspondence
  distorts a whole region: 1519 edges ended stretched past twice their length,
  the worst at 8.6x.  The result is therefore held inside a band around the
  mesh's own edge lengths, and where the band bites the fit gives way rather
  than the mesh.
* Pairing the trunk's *outline* rather than just its centre does contain the
  skeleton, but it also transfers the reference body's shape, and the
  reference is an elderly cadaver with a bulbous occiput and a heavy abdomen.
  The surface came out pot-bellied with a lump on the back of its skull.  Only
  the centres are paired now.

Two approaches were tried and rejected, both measured: a spline through the
fourteen landmarks alone (under-determined -- a 2-unit cube in the forearm
came out 1.36 x 1.64 x 2.00), and blending the per-limb similarities directly
by distance weights (the blended centres move where the weights transition,
which put 59.9 % of the skeleton outside the surface against 51.2 %).

Landmarks matter as much as the method.  The mesh's "wrist" was the lowest
tenth of the arm's vertices, which is the fingertips; the forearm was sheared
to match.  It is now the narrowest station between elbow and hand
(`body/surface_landmarks.py`), and the hand has a landmark of its own, taken
from the middle finger's distal phalanx -- without one the skeleton's
fingertips stood 15.6 units outside the surface.

### The head is fitted to the skull

The skull's face stood 4.8 units in front of the surface's, because the
surface's head sits about 4 units behind it and is a little shallower.  The
head is moved and grown by the least that clears the skull with a margin, per
axis and never below 1, and the change is blended to nothing by the shoulders
(`fit_head_to_skull`).  A single uniform factor was tried first and is set by
the worst axis: the skull is deeper than the head but no taller, so it grew
the whole head by a third.

Inflating the surface wherever *any* bone poked through it was tried too, as
the general form of the same idea.  On a 10 500-vertex mesh a 12-unit push
produces self-intersecting spikes, and it is not usable; the head, where the
mismatch is a clean translation, is.

### The skeleton's fit, measured

Sampled bone vertices lying outside the surface mesh, and how far:

| | outside | p95 | worst |
|---|---|---|---|
| as shipped | 53.5 % | -- | -- |
| placed by similarity, no deformation | 70.2 % | 19.2 | 23.9 |
| registered (now) | 51.2 % | 12.7 | 16.5 |
| skull only, registered | 16 % | 1.4 | 2.1 |

The residual is honest: the two bodies genuinely differ, and every method that
closes the gap further damages the mesh.  What the registration guarantees is
that the mesh is never damaged to get there.

### The surface warp no longer flattens

`body/gender_morph.py`.  The projection is now a constrained solve: a few small
steps toward the target with the mesh's own edge lengths held inside a band
after every step, and a distance gate, because a target 5 units away is a
difference of *pose* between two different bodies rather than a difference of
shape.  A surface-normal agreement test would be the textbook guard and cannot
be used here: the triangle winding of both meshes is inconsistent (38 % of the
surface mesh's face normals and 49 % of the BP3D skin's point outward), so the
test is noise.

| | before | after |
|---|---|---|
| triangles below a tenth of their area | 1235 | 1 |
| hand edges below half their length | 3034 | 0 |
| forearm edges below half | 727 | 32 (30 of them already in the input) |
| torso fit | 0.56 → 0.24 | 0.56 → 0.16 |

## Two bugs found on the way

**The inverse-rest-matrix cache outlived the joints it described.**
`_rest_inv_cache` is keyed by joint index and documented as constant for the
rig's lifetime.  It is constant only while the joint list is: rebuilding it
left index *i* pointing at a new joint while the cache held the old joint's
inverse.  The hull bound, the one pass that reads those deltas, then clamped
554875 skin vertices by a median 0.79 and up to 10.4 units.  Cleared on
rebuild.

**Twenty-three muscles read as bones.**  Bone names were matched by plain
substring, so "Tibialis" was a tibia, "Fibularis" a fibula, "Subscapularis" a
scapula and "Iliocostalis" a costal cartilage.  They were scaled and displaced
as bones and became control points for the warp.  Matching now requires whole
words, and the caller additionally hands over every mesh the skinning owns as
an explicit exclusion.

## What the morph measures out at

Female / male, at gender 1, on the shipped configuration:

| measure | ratio | published |
|---|---|---|
| biacromial breadth | 0.89 | 0.89 |
| bi-iliac breadth | 1.00 | 0.96 |
| bi-iliac / biacromial | 0.78 → 0.88 | 0.75 → 0.80-0.86 |
| femur length | 0.91 | 0.91 |
| stature | 0.945 | 0.93 |
| skin waist / hip | 0.90 → 0.80 | 0.90 → 0.75-0.80 |
| biceps belly radius | 0.58 | ~0.6 |
| rectus femoris belly radius | 0.77 | ~0.75 |

Bi-iliac breadth is worth a note: females do **not** have absolutely wider
hips.  The measured ratio is about 0.96, and it is the *relative* width that is
dimorphic, because the shoulders narrow much more.  Scaling the pelvis up
instead, as the configuration used to, gave a bi-iliac/biacromial ratio of 1.01
against a published female 0.80-0.86.

## Cost

A slider release on a body with skin and four muscle layers (6.2 M vertices):

| stage | seconds |
|---|---|
| scale the skeleton | 0.2 |
| build the warp | 0.9 |
| rebuild every rest pose | 1.8 |
| re-snapshot the skinning | 0.9 |
| the frame that follows | 5.9 |

The re-snapshot replaces a full re-binding, which cost 85 seconds to re-solve
which bone each vertex follows and returned the answer it already had: the body
is the same body at a different size.  If the joint list ever comes back
different the assignment really would be stale, and the full re-registration
still runs.

## Still open

* Stature lands at 0.945 against a published 0.93.  The limbs and the spine are
  right; the remaining difference is in the regions anchored absolutely.
* 33 edges of 2.37 million are still compressed past the band after a morph, 32
  of them in the toes, where the surface pair disagrees most.
* The surface mesh's hands sit about 5 units from the BP3D skin's, which is a
  pose difference between two different bodies that the coarse warp does not
  close.  It is no longer hidden by crushing the mesh onto the target.
