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
| male, fitted | 27.1% | -1.13 | 2.07 | 7.72 |
| female, as loaded | 78.4% | +2.83 | 8.60 | 27.83 |
| female, fitted | 30.2% | -0.95 | 1.71 | 6.16 |

Over the whole vertex set rather than the sample, 68.9% to 20.3% outside and
the worst protrusion 27.9 to 8.5.  `tools/render_skeleton_fit.py --protrusion`
draws it: blue inside, red outside.

What is left, in the order a drawing shows it: the crown of the skull, about 5
units; the anterior chest wall, 3 to 4; the upper thoracic spinous processes
behind a slim neck; the thumb, which the mesh's mitten-shaped hand has no room
for; and the acromion.

## Carrying the soft tissue

Moving the joints and routing that through the articulated skinning tears the
skin at every chain boundary -- 27,747 over-stretched edges, measured for the
sex morph -- so the soft tissue is carried by a displacement field instead.
Two details are specific to this one.

**The field is an inverse-distance blend, not a spline.**  The sex morph uses a
thin-plate spline, which extrapolates outside the hull of its control points; a
fit that turns the shoulder girdle by eight degrees made it extrapolate hard,
and the trapezius and deltoid came away from the thorax in wings.  An
inverse-distance blend of displacements actually measured on the bones is
bounded by the largest of them wherever it is evaluated.  Its neighbour count
and smoothing were chosen on the worst muscle, the quadriceps, which span the
hip where two regions' transforms differ most:

| neighbours / smoothing | worst p99 stretch | median p99 |
|---|---|---|
| 16 / 3 | 3.300 | 1.193 |
| 32 / 6 | 2.657 | 1.184 |
| 48 / 10 | 2.154 | 1.170 |
| 64 / 16 | **1.986** | 1.187 |
| 96 / 30 | 1.973 | 1.178 |

Sampling the field on its lattice is the whole cost of switching the option
on, and a coarse lattice is better on both counts, because interpolating an
already-smooth field more coarsely only smooths it further:

| lattice spacing | toggle | worst p99 stretch | median p99 |
|---|---|---|---|
| 3.0 | 2.01 s | 1.924 | 1.159 |
| 4.5 | 1.54 s | 1.805 | 1.163 |
| 6.0 | **1.45 s** | **1.746** | **1.137** |

It is also deferred until a mesh actually asks to be moved, so switching the
option on with no soft tissue loaded costs 0.09 s rather than 2.4.

**The skin that came with the skeleton is not carried at all.**  It was scanned
from these bones and already fits them.  Dragging it onto the surface mesh's
proportions is the distortion the whole option exists to avoid, and it is
plainly visible: the carried skin comes out a head shorter and broad in the
shoulders.  So it follows the sex morph and nothing else, and the bones simply
sit a little further inside it.  With the fit on or off, that skin renders
byte-identically.

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

* The crown of the skull stands about 5 units proud of the scalp.
* The anterior chest wall and the upper thoracic spinous processes pull in
  opposite directions and the thorax has one matrix for both.  Splitting the
  thoracic spine from the rib cage would give the fit the freedom it wants.
* The mesh's hand is a mitten; the thumb has nowhere to go.
* The fit is solved against the male and the female surface separately and
  lerped between them.  Nothing checks that the lerp of two fits is the fit of
  the lerped surface, and it will not be exactly.
* The joint DOF axes are body axes, so a limb the fit has turned by 10 to 20
  degrees is animated about an axis no longer quite perpendicular to it.
  Small, but it is there while the fit is on.
