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

## The surface warp is off (2026-09-12)

Reinstated at the user's request: the male and female body-surface meshes are
the authored ones, scaled and placed into the BP3D frame and otherwise
untouched. `gender_morph.WARP_SURFACE_TO_SKELETON` turns the fitting above
back on.

The reason is visible rather than numerical. `results/morph_surface_proof.png`
renders both meshes both ways, from three viewpoints, through the app's own GL
renderer: warped, the hands splay, the face sinks and puckers, the feet twist
and the torso loses its symmetry; as authored, they are clean figures and the
male and female shapes read as intended.

It is worth being clear about what the warp was and was not doing. It is
computed from the male mesh and applied to both, so it never distorted the
*morph* -- the difference between the two ends is the authored one either way.
What it distorted was the base both ends sit on. It also inflated the surface:
median edge length 1.336 against 1.131, about 18% longer everywhere.

What turning it off costs is fit, and the whole reason the warp existed. The
surface pair is a MakeHuman male and female; the skeleton is a BP3D cadaver.
They are different bodies, and the measurements in this file record every
method tried to close the gap and what each one damaged. Bone points to the
nearest surface vertex:

| | median | 95th pct | worst |
|---|---|---|---|
| warped onto the skeleton | 2.60 | 10.02 | 16.53 |
| as authored | 5.11 | 15.33 | 26.57 |

So the skeleton sits about twice as far inside the skin, and pokes through it
in more places. That is the trade, taken deliberately: a clean body that does
not quite contain its skeleton, rather than a contained skeleton inside a body
that does not look like one.

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

`tools/anthropometry.py` takes the measurements an anthropometrist takes, at
gender 0 and gender 1, and prints the model's ratio beside the published one.
It exits non-zero when any of them drifts, so this is checked rather than
asserted.

| measure | model | published |
|---|---|---|
| stature | 0.926 | 0.928 |
| sitting height | 0.941 | 0.932 |
| biacromial breadth | 0.894 | 0.894 |
| bi-iliac breadth | 0.964 | 0.964 |
| humerus length | 0.920 | 0.915 |
| femur length | 0.910 | 0.913 |
| tibia length | 0.910 | 0.918 |
| head breadth | 0.953 | 0.954 |
| head length | 0.958 | 0.958 |
| head height | 0.950 | 0.950 |
| bizygomatic breadth | 0.932 | 0.927 |
| bigonial breadth | 0.910 | 0.918 |

And the two features that are angles rather than proportions, where only the
difference between the sexes is modelled because the absolute is the donor's
own anatomy:

| angle | male | female | change | published change |
|---|---|---|---|---|
| carrying angle, right | -9.1 | -7.1 | +2.0 | +2.0 |
| carrying angle, left | -7.2 | -5.2 | +2.0 | +2.0 |
| knee valgus, right | 1.3 | 3.3 | +2.0 | +2.0 |
| knee valgus, left | 4.2 | 6.2 | +2.0 | +2.0 |

What the tool cannot fix is the donor.  The shoulder-to-hip ratio is 0.783 in
the male against a published 0.700, because this cadaver has a wide pelvis for
his shoulders.  That is the body the asset set is.

### The head had no sex at all

Run for the first time, the tool found stature at 0.950 against 0.928 and the
sitting-height ratio at 0.985 against 0.932.  Both were one defect, and it
took a second measurement to see it: exactly five bone meshes were unchanged
between the sexes -- the cranium, the jaw, both sets of teeth and the atlas,
which is the whole head.  The per-bone cranial factors in the config name
bones ("frontal bone", "zygomatic") and the asset is one merged mesh called
"cranium", so none of them had ever matched anything.

Giving the skull a factor is not enough on its own.  It is scaled about its
own centroid, because that is the only anchor a free-standing group has, and
the cervical column is scaled about T1's centroid *as it was before the
morph* -- a point that does not move however far the thoracic column below it
descends.  So the head changed size in place while the trunk shortened under
it.  That also made the vertebral factors almost inert, which is why nobody
had noticed the column was not doing its share: taking the vertebral height
from 0.95 to 0.91 moved stature by 0.002.

`skull_morph.seat_on_neck` puts the neck and head back on top of the column by
the distance its topmost joint moved.  It moves the bones inside those groups,
not the group nodes: the soft-tissue warp is built by comparing each bone's
captured rest position with where it ends up, and a group node's own position
reads as rest either way, so moving the group would have carried the skull and
left the face and the neck muscles behind.

One factor over a merged skull cannot be right everywhere either.  Bizygomatic
breadth is 12.7/13.7 = 0.927 against a vault nearer 0.954, so the vault's
factor leaves the face three per cent too wide.  `skull_morph` adds the
difference back over the lower skull, graded by height from the brow down so
there is no seam, applied about the midline where it is exact.  The same grade
narrows the mastoid process, which is one of the features a skull is sexed by
and larger in males for the same reason.

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

* The `tooth` and `disc` factors are guesses in the same sense the vertebral
  one was: nothing measures them directly.
* Two factors are now set from the whole-body measurement rather than the
  segment one -- the vertebral and disc heights at 0.92 against an osteometric
  body height of 0.95, and bi-iliac breadth at 0.964 where the pelvis used to
  keep the male's breadth exactly.  The whole body is the better measured of
  the two, but the segment figures are the ones with a citation.
* 33 edges of 2.37 million are still compressed past the band after a morph,
  32 of them in the toes, where the surface pair disagrees most.
* The face mesh and the facial muscles have no sex of their own.  They now
  follow the skull, because the skull moves and the soft-tissue warp carries
  them, but nothing shapes them.
