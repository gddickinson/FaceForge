"""Registering the body-surface mesh onto the skeleton.

The surface mesh and the skeleton are different bodies in different poses, so
the mesh has to be brought onto the skeleton before either can be drawn with
the other.  How that is done is the whole question: the version this replaced
remapped the mesh piecewise in Z and blended two arm rotations against it,
which is not a rigid motion, and it sheared the limbs -- the forearm's depth
fell from 24.7 to 17.4 while its width rose from 21.5 to 28.4, the foot lost a
third of its length and the occiput was shaved flat.

Here each limb is matched on its own, rotated and scaled onto its bone; the
trunk and head are matched to the reference body level by level; the result is
held inside a band around the mesh's own edge lengths; and the head is finally
moved and grown by the least that clears the skull.  ``docs/sex_morph.md``
records the measurements behind each of those choices.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.body.edge_relaxation import enforce_edge_range
from faceforge.body.surface_landmarks import extract_mesh_landmarks

logger = logging.getLogger(__name__)


#: The body as a chain of segments, each named by the two landmarks that bound
#: it.  Each is matched on its own -- rotated, scaled and moved onto the
#: corresponding bone -- and the transforms are sampled into a cloud of
#: correspondences that a spline interpolates.
SEGMENTS: tuple[tuple[str, str], ...] = (
    ("shoulder_R", "elbow_R"), ("elbow_R", "wrist_R"), ("wrist_R", "hand_R"),
    ("shoulder_L", "elbow_L"), ("elbow_L", "wrist_L"), ("wrist_L", "hand_L"),
    ("hip_R", "knee_R"), ("knee_R", "ankle_R"),
    ("hip_L", "knee_L"), ("knee_L", "ankle_L"),
)

#: Stations along a segment, and the radius of the ring of off-axis samples at
#: each, as a fraction of the segment's length.  The ring pins the rotation:
#: samples on the axis alone would leave the spline free to twist.
SEGMENT_STATIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEGMENT_RING = 0.22
RING_ANGLES = 6

#: Levels traced along the trunk and head, and the half-width counted as
#: "torso" at each -- wide enough for the shoulders, narrow enough to leave
#: the arms out.
AXIS_LEVELS = 16
TORSO_HALF_WIDTH = 16.0

#: The band the registered mesh's own edges are held inside, as a fraction of
#: their length before the registration.
REGISTER_MAX_STRETCH = 0.25
REGISTER_MAX_COMPRESSION = 0.25
REGISTER_SWEEPS = 60


def similarity_to(pos: NDArray, target: NDArray) -> NDArray:
    """Uniform scale and translation putting ``pos`` on ``target``'s extent.

    Landmark thresholds are absolute heights, so the two meshes have to be the
    same height before like-for-like landmarks can be taken from them.
    Returns the displacement, not the positions.
    """
    p = np.asarray(pos, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    span_p = float(p[:, 2].max() - p[:, 2].min())
    span_t = float(t[:, 2].max() - t[:, 2].min())
    scale = span_t / span_p if span_p > 1e-9 else 1.0
    out = p * scale
    out[:, 2] += float(t[:, 2].min()) - float(out[:, 2].min())
    for ax in (0, 1):
        out[:, ax] += float(np.median(t[:, ax]) - np.median(out[:, ax]))
    return out - p


def _segment_samples(src_p: NDArray, src_d: NDArray,
                     dst_p: NDArray, dst_d: NDArray) -> tuple[NDArray, NDArray]:
    """Correspondences for one bone: where its own similarity takes a ring of points."""
    from faceforge.body.surface_projection import rotation_between

    v = np.asarray(src_d, dtype=np.float64) - np.asarray(src_p, dtype=np.float64)
    w = np.asarray(dst_d, dtype=np.float64) - np.asarray(dst_p, dtype=np.float64)
    nv, nw = float(np.linalg.norm(v)), float(np.linalg.norm(w))
    if nv < 1e-6 or nw < 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3))
    rot = rotation_between(v, w)
    scale = nw / nv
    axis = v / nv
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    t = np.cross(axis, u)
    radius = SEGMENT_RING * nv
    pts = []
    for station in SEGMENT_STATIONS:
        base = np.asarray(src_p, dtype=np.float64) + v * station
        pts.append(base)
        for k in range(RING_ANGLES):
            a = 2.0 * np.pi * k / RING_ANGLES
            pts.append(base + radius * (np.cos(a) * u + np.sin(a) * t))
    src = np.asarray(pts)
    dst = np.asarray(dst_p, dtype=np.float64) + scale * (src - np.asarray(src_p)) @ rot.T
    return src, dst


def _axis_samples(src: NDArray, dst: NDArray) -> tuple[NDArray, NDArray]:
    """Matched trunk and head outlines, level by level, plus the crown and soles.

    Only the centres.  Pairing the outlines as well was tried and measured:
    it does contain the skeleton, but it also transfers the reference body's
    *shape*, and the reference is an elderly cadaver with a bulbous occiput
    and a heavy abdomen.  The surface mesh came out pot-bellied with a lump on
    the back of its skull.  Containment is dealt with afterwards instead, by
    inflating the surface only where a bone actually pokes through it, which
    can add volume but never take shape away.
    """
    out_s, out_d = [], []
    lo = max(float(src[:, 2].min()), float(dst[:, 2].min()))
    hi = min(float(src[:, 2].max()), float(dst[:, 2].max()))
    lo = lo + 0.45 * (hi - lo)          # trunk and head; the legs have segments
    picks = (lambda q: q.mean(axis=0),)
    for zl in np.linspace(lo, hi, AXIS_LEVELS):
        bands = []
        for pts in (src, dst):
            band = pts[(np.abs(pts[:, 2] - zl) < 3.0)
                       & (np.abs(pts[:, 0]) < TORSO_HALF_WIDTH)]
            bands.append(band if len(band) >= 8 else None)
        if bands[0] is None or bands[1] is None:
            continue
        for pick in picks:
            out_s.append(pick(bands[0]))
            out_d.append(pick(bands[1]))
    for pick in (lambda q: q[np.argmax(q[:, 2])], lambda q: q[np.argmin(q[:, 2])]):
        out_s.append(pick(src))
        out_d.append(pick(dst))
    if not out_s:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.asarray(out_s), np.asarray(out_d)


#: Clearance left between the skull and the surface of the head.
HEAD_MARGIN = 1.5

#: The head's fit is blended to nothing over this height, ending at the
#: shoulders, so the neck carries the change instead of creasing.
HEAD_BLEND = 30.0


def fit_head_to_skull(pos: NDArray, skull: NDArray, shoulder_z: float,
                      margin: float = HEAD_MARGIN) -> NDArray:
    """Move and grow the head just enough to enclose the skull.  Returns positions.

    The skull's face stood 4.8 units in front of the surface's, because the
    surface's head sits about 4 units behind it and is a little shallower.
    The head is therefore shifted and scaled -- one similarity, so its shape
    is untouched -- by the least that clears the skull with a margin, and the
    change is blended to nothing by the shoulders.  The scale is never allowed
    below 1: a head may grow to fit its skull, never shrink onto it.
    """
    p = np.array(pos, dtype=np.float64, copy=True)
    sk = np.asarray(skull, dtype=np.float64)
    if len(sk) == 0:
        return p
    head = p[p[:, 2] > float(sk[:, 2].min())]
    if len(head) < 20:
        return p

    lo_h, hi_h = head.min(axis=0), head.max(axis=0)
    lo_s, hi_s = sk.min(axis=0) - margin, sk.max(axis=0) + margin
    # Nothing to clear downward: the skull's base sits inside a neck that
    # carries on below it, and the head is only selected from that base up.
    # Letting the inferior direction drive the fit grew a head that already
    # enclosed its skull by an eighth.
    lo_s[2] = lo_h[2]
    centre = 0.5 * (lo_h + hi_h)
    # The smallest uniform scale about the head's own centre that covers the
    # skull once the head has been recentred on it.
    target_centre = 0.5 * (np.minimum(lo_h, lo_s) + np.maximum(hi_h, hi_s))
    half_h = np.maximum(0.5 * (hi_h - lo_h), 1e-6)
    half_need = np.maximum(np.abs(hi_s - target_centre), np.abs(target_centre - lo_s))
    # Per axis, not one uniform factor: a single factor is set by the worst
    # axis, and the skull is deeper than the head but no taller, so a uniform
    # fit grew the whole head by a third.  A diagonal scale has no shear in it.
    scale = np.maximum(half_need / half_h, 1.0)
    shift = target_centre - centre
    if np.allclose(scale, 1.0, atol=1e-9) and np.allclose(shift, 0.0, atol=1e-9):
        return p

    z = p[:, 2]
    t = np.clip((z - shoulder_z) / max(HEAD_BLEND, 1e-6), 0.0, 1.0)
    w = (t * t * (3.0 - 2.0 * t))[:, None]
    moved = centre + shift + scale[None, :] * (p - centre)
    out = p + w * (moved - p)
    logger.info("Head fitted to the skull: scale %s, shift %s",
                np.round(scale, 3).tolist(), np.round(shift, 2).tolist())
    return out


def constrain_to_rest(pos: NDArray, rest: NDArray, indices: NDArray) -> NDArray:
    """Hold every edge of ``pos`` inside a band around its length in ``rest``."""
    tris = np.asarray(indices).reshape(-1, 3)
    edges = np.unique(np.sort(np.concatenate(
        [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    keep = rest_len > 1e-6
    out = np.array(pos, dtype=np.float64, copy=True)
    report = enforce_edge_range(out, edges[keep], rest_len[keep],
                                max_stretch=REGISTER_MAX_STRETCH,
                                max_compression=REGISTER_MAX_COMPRESSION,
                                iterations=REGISTER_SWEEPS)
    logger.info("Surface registration constrained: %s in %d sweeps",
                "converged" if report["converged"] else "residual remains",
                report["iterations_run"])
    return out


def register_onto(pos: NDArray, skel_lm: dict[str, NDArray],
                  target: NDArray, indices: Optional[NDArray] = None) -> NDArray:
    """Displacement taking the surface mesh onto the skeleton's pose.

    It replaced a piecewise Z-remap blended against two arm rotations.
    Blending a rotation against a translation is not a rigid motion and it
    sheared the limbs: measured on the shipped mesh, the forearm's depth fell
    from 24.7 to 17.4 while its width rose from 21.5 to 28.4, the foot lost a
    third of its length and the occiput was shaved flat.

    Every limb is matched on its own -- rotated, scaled and moved onto its
    bone -- and the trunk and head are matched level by level to the reference
    body's own outline.  Those correspondences are interpolated by a spline,
    and the result is then held inside a band around the mesh's own edge
    lengths, because a spline is a global interpolant and one inconsistent
    correspondence distorts a whole region.  Where the band bites, the fit
    gives way rather than the mesh.
    """
    from faceforge.body.skeleton_field import displacement_warp

    disp = similarity_to(pos, target)
    moved = np.asarray(pos, dtype=np.float64) + disp
    src_lm = extract_mesh_landmarks(moved)
    tgt = np.asarray(target, dtype=np.float64)

    src_all, dst_all = [], []
    for prox, dist in SEGMENTS:
        if not all(k in src_lm and k in skel_lm for k in (prox, dist)):
            continue
        a, b = _segment_samples(src_lm[prox], src_lm[dist],
                                skel_lm[prox], skel_lm[dist])
        if len(a):
            src_all.append(a)
            dst_all.append(b)
    a, b = _axis_samples(moved, tgt)
    if len(a):
        src_all.append(a)
        dst_all.append(b)
    if not src_all:
        return disp

    src = np.concatenate(src_all)
    dst = np.concatenate(dst_all)
    logger.info("Surface registration: %d correspondences, median move %.1f",
                len(src), float(np.median(np.linalg.norm(dst - src, axis=1))))
    warped = moved + displacement_warp(src, dst - src)(moved)
    if indices is not None:
        warped = constrain_to_rest(warped, moved, indices)
    return warped - np.asarray(pos, dtype=np.float64)




