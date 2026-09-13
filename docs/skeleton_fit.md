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

`faceforge/body/fit_regions.py` treats the skeleton as a tree of 17 regions,
each anchored at the joint it hangs from:

```
trunk ── neck ── head
  ├── girdle_R ── upperarm_R ── forearm_R ── hand_R
  ├── girdle_L ── ...
  ├── thigh_R  ── shank_R ── foot_R
  └── thigh_L  ── ...
```

Each region carries a full 3×3 matrix, not a per-axis scale: the two bodies
differ in limb *direction* as well as length -- the forearm axes by 15 degrees
and the shank axes by 10 -- and a diagonal cannot say that.  The anchor is
carried by the parent,

    A'(r) = T(parent(r))(A(r))
    T(r)(x) = A'(r) + M(r) (x - A(r))

so no articulation can come apart however the matrices are chosen.  Only the
trunk and the head also carry a translation: the trunk hangs from nothing, and
the skull is a group of its own rather than a bone on a cervical pivot, so
moving it opens no joint surface.

## How the matrices are chosen

`tools/fit_skeleton_to_skin.py`, offline, writes `assets/config/skeleton_fit.json`.
The objective is the thing the option is for and nothing else: the mean squared
protrusion of every bone vertex the region carries, plus a penalty that stops a
bone shrinking away from a surface it cannot reach (a skeleton free to shrink
fits any surface by vanishing).  Each region is a rotation times a per-axis
scale, bounded to [0.70, 1.15] and ±20°, searched by coordinate descent,
parents first, three passes.  One sex at a time; the runtime lerps the two
tables with the slider, because the surface it is fitted to is itself lerped.

## What it achieves

| | outside | median | p95 | max |
|---|---|---|---|---|
| male, fitted | 35.4% | −0.78 | 2.98 | 7.20 |
| female, fitted | 40.5% | −0.49 | 2.91 | 7.08 |

Over the whole vertex set rather than the sample, 68.9% → 27.0% outside and
the worst protrusion 27.92 → 7.26.  `tools/render_skeleton_fit.py --protrusion`
draws it: blue inside, red outside.

What is left is the crown of the skull (about 4 units), the acromion, the
sacrum and the fingertips.  The hands want more than the ±20° the search
allows; they are the one region that ends against its bound.

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
| 16 / 3 | 2.926 | 1.069 |
| 32 / 6 | 2.396 | 1.080 |
| 48 / 10 | **2.022** | 1.081 |
| 64 / 16 | 2.046 | 1.129 |

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

* The crown of the skull stands about 4 units proud of the scalp.  Bringing it
  in costs more in skull shape than it buys in containment at the penalty the
  search uses.
* The hands end against the ±20° rotation bound; the two bodies' wrists differ
  by more than that.
* The fit is solved against the male and the female surface separately and
  lerped between them.  Nothing checks that the lerp of two fits is the fit of
  the lerped surface, and it will not be exactly.
