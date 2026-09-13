# Fitting the skeleton into the body-surface mesh

The body-surface mesh is a MakeHuman figure.  The skeleton is a BodyParts3D
cadaver.  They are two different bodies, and until now the only way the project
reconciled them was by deforming the *surface* onto the bones.  Every
measurement in `docs/sex_morph.md` says that damages the mesh, and it is
switched off (`gender_morph.WARP_SURFACE_TO_SKELETON`).

This is the other direction: **Body → Fit skeleton to body mesh** moves the
bones instead and leaves the authored surface exactly as its authors drew it.

## What the disagreement is

Every bone vertex, signed distance to the surface, positive outside
(`tools/skeleton_containment.py`; the sign is calibrated against a point known
to be inside the body, because the MakeHuman surface winds inward and reading
the winding inverts the whole measurement):

| | outside | median | p95 | max |
|---|---|---|---|---|
| male, as loaded | 65.7% | +2.42 | 11.38 | 27.73 |
| female, as loaded | 78.4% | +2.83 | 8.60 | 27.83 |

It is not one global size difference.  Scaling the whole skeleton about the
soles, over scale factors from 0.86 to 1.0 laterally and 0.88 to 1.0
vertically, never brought the outside fraction below 60%: shrinking the
skeleton laterally pulls the humerus *out* of the mesh's sleeve and into the
gap beside the chest.  The mismatch is per-limb, and the worst of it is:

| region | outside | median | max |
|---|---|---|---|
| cranium | 100% | +7.9 | 27.7 |
| humerus | 86% | +12.2 | 19.5 |
| scapula | 100% | +12.8 | 19.3 |
| every bone of the foot | 100% | +4 to +8 | 14.9 |

## The model

`faceforge/body/fit_regions.py` treats the skeleton as a tree of 23 regions,
each anchored at the joint it hangs from:

```
pelvis - lumbar - thorax - neck - head
  |                  |- girdle_R - upperarm_R - forearm_R - hand_R - fingers_R
  |                  `- girdle_L - ...
  |- thigh_R - shank_R - foot_R - toes_R
  `- thigh_L - ...
```

Each region carries a rotation and a per-axis scale, not a scale alone: the
two bodies differ in limb *direction* as well as length -- the forearm axes by
15 degrees and the shank axes by 10 -- and a diagonal cannot say that.  **The
rotations compose down the chain, the way a pose does**: turning the humerus at
the shoulder carries the forearm, the hand and every finger with it, and the
forearm's own rotation is a correction relative to that.

```
R(r)  = R(parent(r)) @ Rlocal(r)      rotations compose, as a pose does
M(r)  = R(r) @ diag(scale(r))
A'(r) = T(parent(r))(A(r))            the anchor, carried by the parent
T(r)(x) = A'(r) + M(r) (x - A(r))     everything else, about the anchor
```

Carrying the anchor is what stops an articulation opening, whatever the
parameters are.  Only two regions also carry a translation: the pelvis,
because it hangs from nothing, and the head, because the skull is a group of
its own rather than a bone on a cervical pivot, so moving it opens no joint
surface.

The first version rotated each region *absolutely* in body coordinates, so a
turn at the shoulder moved the elbow but left the forearm pointing where it
always had.  The arm bent instead of swinging, and the hand was the one region
that ended against the rotation bound, because it had to express the whole
arm's turn by itself.

### Posture

The two bodies are not only different sizes, they are in different *poses*,
and a containment measure is blind to the difference.  Two things are
therefore **authored** rather than searched for, in `fit_regions.py`:

* **The forearms are pronated 92 degrees.**  The plane of the skeleton's
  metacarpals has its normal along Y -- the palm faces forward, the arm is
  supinated -- while the body mesh's hand has its normal along X, the palm
  facing the thigh.  The angle between them is 88.5 degrees on the male mesh
  and 88.0 on the female, and a turn of 92 about the elbow-to-wrist axis
  aligns them to a cosine of 0.988.  A pronated forearm and a supinated one
  fill almost the same sleeve, so no amount of searching would have found it.
  After the fit the palms agree to 4.8 degrees.
* **The skull is reshaped.**  It is 19.8 units wide and 29.1 deep, and has to
  live inside a head 21.6 x 26.0 on the male mesh and 20.8 x 25.3 on the
  female: deeper than the head it goes in, with about two units of scalp to
  spare.  Every objective tried either widened the cranium until it exactly
  filled the head with no scalp at all, or left the occiput four to five units
  out the back, because that is a small patch and a skull cannot be pulled
  backward by scaling about a joint underneath it.  The depth is set to 22
  units and the width to 17.6, both leaving two units of cover, and the skull
  is moved forward to sit in the face rather than the nape.  The solved table
  then refines it per sex, and does: the female skull ends up shallower
  (0.70 against 0.73) and shorter (0.89 against 0.96).

The left half is the mirror of the right, and the midline regions may lengthen,
widen and nod but not twist or lean.  Solved independently, the two halves
found different local optima: the right hand ended 6.8 units inside the mesh
while the left was 44% outside it.

## How the matrices are chosen

`tools/fit_skeleton_to_skin.py`, offline, writes `assets/config/skeleton_fit.json`.
Coordinate descent, parents before children, three passes, right side only.
Four things go into the objective, and three of them are there because
something went visibly wrong without them.

**Protrusion.**  The mean squared depth of the region's bone vertices outside
the surface.  This is the thing the option is for.

**A region answers half for its own bones.**  Pooling a region's whole subtree
into one mean made the thorax's own bones a thirteenth of its objective, and
flattening the chest then cost more in the scale penalty than it saved: the
sternum and costal cartilages were left standing 7 to 9 units through the
chest.  Weighting each region's sample back up to its true vertex count fixed
that and broke something worse, because vertex count is tessellation, not
anatomy -- the hand and fingers carry 10,000 vertices across their many small
bones against the humerus's 400, so the arm's objective became the hand's and
the humerus and scapula were left 10 to 17 units out.

**A travel limit.**  Containment alone has a spectacular degenerate optimum:
the search swung both forearms across the body until the hands lay inside the
thighs, where nothing sticks out of anything.  Measured, the fitted right
wrist sat at x = -6.1, across the midline, with the right fingertips between
the knees -- and the containment score called it perfect.  "Inside the
surface" is not "inside the matching part of the surface", and no protrusion
measure can tell the difference.

Two ways of saying "the matching part" were tried and both are worse.  Tying
each limb's distal end to the body mesh's own landmark for it fails because
those landmarks are biased by how they are found: the "ankle" is the mean of a
band taken from the lateral half of the leg, so it sits 13 units out from the
leg's axis, and pulling the ankle onto it dragged both feet clean out of the
mesh -- 100% of the foot outside, a median of 8 units.  Calibrating how deep a
bone may sit from the skin that came with the skeleton fails because that mesh
is not a hollow surface: 24,757 of its vertices lie inside a 12-unit column
through the chest, so every depth measured against it comes out near zero.

What works is the simplest true statement about the job: this is a fit, not a
reposing.  The whole misfit is 27.7 units at its very worst and 2.4 at the
median, so a bone may move 20 units freely and pays beyond that.  The
shoulder's real correction is 24 units and costs almost nothing; the
degenerate wrist's 51 costs more than the rest of the objective together.

**The worst case, not only the mean.**  A mean tolerates one deep patch, and a
deep patch is exactly what a viewer sees: the skull sat 27.8 units deep inside
a 26.0 head with its occiput 4.4 out the back, and the mean-squared protrusion
barely noticed -- 11% of the region outside at a p95 of 0.44.  The 98th
percentile of each region's protrusion is in the objective too.  Pressed
harder than that, though, the search starts buying a better worst case by
burying a region somewhere roomy: at three times the weight the hands left
their sleeves entirely and sat 9.4 units inside the surface.  A loose guard
against burial -- no region much deeper than it started -- catches that
without telling a bone how deep it should be.

**A shape penalty.**  A skeleton free to shrink fits any surface by vanishing,
so each region is held toward scale 1, bounded to [0.60, 1.20] and to 20
degrees relative to its parent.  Rotation is nearly free: turning a limb costs
a real skeleton nothing, where scaling a bone changes what it is.

One sex at a time; the runtime lerps the two tables with the slider, because
the surface it is fitted to is itself lerped.

## What the drawings changed

`tools/inspect_skeleton_fit.py` draws the skeleton and the surface
orthographically on a labelled grid in body units, front, side and in closeup,
with bones dark where they are inside and red where they are out.  Three of
the four objective terms above exist because a drawing showed something the
summary numbers could not: the arms folded into the thighs, the chest wall
standing through the skin, and the two halves of the body fitted differently.

The grid also showed the ribcage's anteroposterior scale sitting exactly on
its lower bound while the sternum still stood 4 to 5 units proud of the chest,
which is why the bound is 0.60 rather than 0.70.  The cadaver's chest is
simply deeper than the MakeHuman figure's.

## What it achieves

Every bone vertex, signed distance to the surface, positive outside:

| | outside | median | p95 | max |
|---|---|---|---|---|
| male, as loaded | 65.7% | +2.42 | 11.38 | 27.73 |
| male, fitted | 19.0% | -1.50 | 0.88 | 6.91 |
| female, as loaded | 78.4% | +2.83 | 8.60 | 27.83 |
| female, fitted | 8.1% | -2.34 | 0.25 | 6.65 |

And on the things a containment number cannot see:

| | before | after |
|---|---|---|
| angle between the palms | 88.5 deg | 4.8 deg |
| hip joint height | -81.0 | -91.5 |
| upper arm length (mesh 27.8) | 40.4 | 31.0 |
| middle fingertip to the mesh's | 62.7 | 2.0 |
| skull depth inside a 26.0 head | 29.1 | 22.6 |
| occiput proud of the head | 7.2 | 0.0 |
| crown proud of the scalp | 27.5 | 3.3 |

What is left: the crown of the skull, about 3 units; the acromion; the thumb,
which the mesh's mitten-shaped hand has no room for; and a graze along the
fingers and toes.

## Carrying the soft tissue

Moving the joints and routing that through the articulated skinning tears the
skin at every chain boundary -- 27,747 over-stretched edges, measured for the
sex morph -- so the soft tissue is carried by a displacement field instead.
Getting that field right took four attempts, and each failure is worth
recording because each one looked fine in the numbers that were being watched
at the time.

**A thin-plate spline**, as the sex morph uses, is an interpolant: outside the
hull of its control points it extrapolates, and a fit that turns the shoulder
girdle by eight degrees made it extrapolate hard.  The trapezius and deltoid
came away from the thorax in wings.

**A blend of the displacements** is bounded -- every value is a convex
combination of displacements that really happened -- but it cannot extrapolate
a *scale*.  Shrink a femur by a tenth and the blend carries the bone's own
surface correctly, while a muscle eight units outside it barely moves.
Rendered, the quadriceps ballooned out past the leg, 15 units through the skin
where they had been 4, and no neighbourhood or smoothing changed it.

**A blend of the region matrices** extrapolates correctly but is not closed
under averaging: a weighted mean of two rotation matrices is not a rotation,
and with a forearm turned 92 degrees the averages collapse.  The worst
muscle's 99th-percentile edge stretch went from 2.21 to 5.37.

**A blend of where each region puts the point** is what it does now -- each
region's affine applied, the results mixed.  Every value is a convex
combination of positions each of which is correct, and a scale extrapolates
because the affine does.  Two guards make it safe:

* *No region may move a point further than it moved its own bones.*  An affine
  evaluated far outside its region extrapolates wildly.
* *Influence is limited by distance along the region tree, not through space.*
  The finger bones hang beside the thigh, close enough to take 31% of the
  weight on the quadriceps and seven steps away from them in the skeleton.
  Carrying that weight, and with it a 92-degree pronation extrapolated 30
  units, was the whole of a 16-unit error.

| neighbours / smoothing | worst p99 stretch | quadriceps out |
|---|---|---|
| 64 / 16 | 2.148 | 3.44 |
| 32 / 8 | 1.944 | 4.17 |
| 16 / 3 | 1.963 | 1.35 |
| **32 / 4** | **1.611** | **1.17** |

Sampling the field on its lattice is the whole cost of switching the option
on, and it is deferred until a mesh actually asks to be moved, so the toggle
costs 0.09 s with no soft tissue loaded.

**The skin that came with the skeleton is not carried at all.**  It was
scanned from these bones and already fits them.  Dragging it onto the surface
mesh's proportions is the distortion the whole option exists to avoid, and it
is plainly visible: the carried skin comes out a head shorter and broad in the
shoulders.  So it follows the sex morph and nothing else, and renders
byte-identically with the fit on or off.

## Does the soft tissue still fit?

`tools/fit_tissue_check.py` loads every layer and measures three things per
mesh: edge stretch against its own pre-fit rest pose, how far its centroid
moved relative to the nearest bone's, and how far it protrudes through the
body surface -- against a control taken before the fit, because the two
bodies disagreed about a good deal before anything moved.

Furthest any mesh in the layer protrudes through the surface, before and
after:

| layer | male before | male after | female before | female after |
|---|---|---|---|---|
| arm muscles | 19.3 | 6.1 | 16.7 | 3.8 |
| back muscles | 18.9 | 5.3 | 16.2 | 3.4 |
| shoulder muscles | 24.1 | 4.7 | 20.3 | 4.3 |
| torso muscles | 17.4 | 8.2 | 15.0 | 4.0 |
| hip muscles | 10.2 | 3.2 | 9.5 | -0.5 |
| leg muscles | 14.8 | 10.5 | 8.0 | 2.9 |
| hand muscles | 7.1 | 1.8 | 7.7 | 1.7 |
| foot muscles | 14.0 | 2.2 | 8.2 | 1.1 |
| organs | 14.8 | 4.2 | 15.0 | 7.2 |
| vasculature | 11.3 | 4.4 | 10.5 | 2.5 |
| ligaments | 11.1 | 2.3 | 6.3 | 3.5 |

Every layer is better off than it was, at both sexes.  32 meshes of 442 are
past some limit at gender 0 and 14 at gender 1.

## The female changes still happen

The sex morph's effect on the soft tissue is the same whether or not the fit
is in force, which is what it should be: the fit is about where the bones are,
not about what sex the body is.

| | fit off | fit on |
|---|---|---|
| muscles, median displacement | 3.11 | 3.32 |
| muscles, bounding volume | x0.621 | x0.609 |
| organs and other soft tissue | 1.81 | 2.05 |
| organs, bounding volume | x0.703 | x0.669 |
| the skeleton's own skin | 3.23, x0.766 | 3.23, x0.766 |

## A defect this found

A footprinted muscle is placed every frame from its harmonic fibre field, and
the field stores the rest pose it was solved on.  Nothing refreshed that store
when a morph rewrote the rest pose, so the field wrote the *old* geometry back
over the new one while the vertices it does not drive stayed where the morph
put them.  With the fit on this was unmissable; with a sex morph it was small
enough to have gone unnoticed.  `MuscleAttachmentSystem.refresh_rest_poses`
now runs from `refresh_after_skeleton_change`, so both paths get it.  With the
fix, the render is byte-identical to the control with the attachment system
switched off entirely.

## Still open

* The male adductors stand 10.5 units through the medial thigh, where they
  stood 2.7 before.  The female's do not.  They span the pelvis and the thigh,
  two regions one step apart, and the blend between them does not suit.
* The digastric intermediate tendons stretch 4.5x at the 99th percentile.
  They are small and they end up inside, but that is a tear.
* The crown of the skull stands about 3 units proud of the scalp.
* The thumb: the mesh's hand is a mitten and has nowhere to put one.
* The fit is solved against the male surface and the female surface
  separately.  The pose is shared -- sexual dimorphism is proportion, not
  posture, and left free the female solve folded the arm inward and buried the
  hand in the torso, 34.5 units from where the mesh keeps it -- but nothing
  checks that the lerp of two fits is the fit of the lerped surface.
* The joint DOF axes are body axes, so a limb the fit has turned -- and the
  forearm is now turned 92 degrees -- is animated about an axis no longer
  perpendicular to it.
