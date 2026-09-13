"""Measure the model against published anthropometry, at either sex.

The sex morph's factors came from the literature -- they are quoted in
``assets/config/gender_dimorphism.json`` -- but nothing has ever measured the
result back against it, so drift is invisible.  This does: it takes the same
measurements an anthropometrist would, at gender 0 and gender 1, and prints
the model's female-to-male ratio beside the published one.

    PYTHONPATH=src python -m tools.anthropometry
    PYTHONPATH=src python -m tools.anthropometry --surface   # the skin, not the bones

Reference values are adult means.  Stature, sitting height, biacromial and
bi-iliac breadth are the standard survey figures quoted in the morph config;
the long-bone lengths are the osteometric means it was built from; the
cranial dimensions are the head measurements used by
``faceforge.body.skull_morph``.  They are means of populations, not of this
cadaver, so the interesting number is the ratio, not the absolute.
"""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray

#: measure -> (male reference, female reference, unit, what it is)
PUBLISHED: dict[str, tuple[float, float, str]] = {
    "stature":            (175.6, 162.9, "cm"),
    "sitting height":     (91.4, 85.2, "cm"),
    "biacromial breadth": (39.7, 35.5, "cm"),
    "bi-iliac breadth":   (27.8, 26.8, "cm"),
    "humerus length":     (33.0, 30.2, "cm"),
    "femur length":       (47.3, 43.2, "cm"),
    "tibia length":       (38.0, 34.9, "cm"),
    "head breadth":       (15.2, 14.5, "cm"),
    "head length":        (18.9, 18.1, "cm"),
    "head height":        (13.0, 12.35, "cm"),
    "bizygomatic breadth": (13.7, 12.7, "cm"),
    "bigonial breadth":   (11.0, 10.1, "cm"),
}

#: Angles, in degrees, measured in the frontal plane and signed so that
#: positive is lateral deviation of the distal segment.  The published figures
#: are the carrying angle at the elbow and genu valgum at the knee.
#:
#: Only the difference between the sexes is modelled; the absolute angles are
#: the donor's own anatomy, and asymmetric as a real body's are.  So what is
#: checked is the change, not the value.
ANGLES: tuple[tuple[str, str, str, str, float], ...] = (
    ("carrying angle", "shoulder", "elbow", "wrist", 2.0),
    ("knee valgus", "hip", "knee", "ankle", 2.0),
)

#: Proportions, which are what a reader actually recognises as male or female.
#: Each is (numerator, denominator, male reference, female reference).
RATIOS: tuple[tuple[str, str, str, float, float], ...] = (
    ("shoulder-to-hip", "bi-iliac breadth", "biacromial breadth", 0.700, 0.755),
    ("sitting-height fraction", "sitting height", "stature", 0.520, 0.523),
    ("relative femur", "femur length", "stature", 0.269, 0.265),
)


def bone_points_by_name(root) -> dict[str, NDArray]:
    """Every bone mesh's vertices in body coordinates."""
    from faceforge.body.fit_regions import SKIP_SUBTREES

    out: dict[str, NDArray] = {}
    stack = [(root, np.zeros(3))]
    while stack:
        node, offset = stack.pop()
        for child in node.children:
            name = getattr(child, "name", "") or ""
            if name in SKIP_SUBTREES:
                continue
            here = offset + np.asarray(child.position, dtype=np.float64)
            mesh = getattr(child, "mesh", None)
            if mesh is not None and name:
                geo = mesh.geometry
                p = np.asarray(geo.positions, dtype=np.float64).reshape(-1, 3)
                out[name] = p[:geo.vertex_count] + here
            stack.append((child, here))
    return out


def frontal_deviation(joints: dict, proximal: str, joint: str, distal: str,
                      side: str) -> float | None:
    """Signed angle the distal segment makes with the proximal one, in degrees.

    In the frontal plane, positive laterally.  Unsigned it is useless here:
    the arm hangs abducted in this skeleton's rest pose, so the forearm starts
    out deviating *medially* from the humerus, and a measure that only knows
    the size of the deviation reads an increase in the carrying angle as a
    decrease.
    """
    keys = [f"{n}_{side}" for n in (proximal, joint, distal)]
    if any(k not in joints for k in keys):
        return None
    a, b, c = (np.asarray(joints[k], dtype=np.float64) for k in keys)
    u = (b - a)[[0, 2]]
    v = (c - b)[[0, 2]]
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return None
    u, v = u / nu, v / nv
    lateral = 1.0 if side == "R" else -1.0
    return float(np.degrees(np.arctan2(lateral * (u[0] * v[1] - u[1] * v[0]),
                                       float(u @ v))))


def span(points: NDArray, axis: int) -> float:
    return float(points[:, axis].max() - points[:, axis].min())


def measure(bones: dict[str, NDArray], joints: dict) -> dict[str, float]:
    """The measurements, in body units."""
    everything = np.vstack(list(bones.values()))
    out: dict[str, float] = {}
    out["stature"] = span(everything, 2)

    hips = [bones[n] for n in ("Right Hip Bone", "Left Hip Bone") if n in bones]
    if hips:
        pelvis = np.vstack(hips)
        out["bi-iliac breadth"] = span(pelvis, 0)
        # Sitting height: crown to the ischial tuberosities, which is what a
        # seated body rests on.
        out["sitting height"] = float(everything[:, 2].max() - pelvis[:, 2].min())

    scapulae = [bones[n] for n in ("Right Scapula", "Left Scapula") if n in bones]
    if scapulae:
        out["biacromial breadth"] = span(np.vstack(scapulae), 0)

    def joint_gap(a: str, b: str) -> float | None:
        if a not in joints or b not in joints:
            return None
        return float(np.linalg.norm(np.asarray(joints[a], dtype=np.float64)
                                    - np.asarray(joints[b], dtype=np.float64)))

    for label, a, b in (("humerus length", "shoulder_R", "elbow_R"),
                        ("femur length", "hip_R", "knee_R"),
                        ("tibia length", "knee_R", "ankle_R")):
        gap = joint_gap(a, b)
        if gap is not None:
            out[label] = gap

    if "cranium" in bones:
        skull = bones["cranium"]
        out["head breadth"] = span(skull, 0)
        out["head length"] = span(skull, 1)
        out["head height"] = span(skull, 2)
        z0, z1 = skull[:, 2].min(), skull[:, 2].max()
        face = skull[skull[:, 2] < z0 + (z1 - z0) * 0.30]
        if len(face):
            out["bizygomatic breadth"] = span(face, 0)
    if "jaw" in bones:
        out["bigonial breadth"] = span(bones["jaw"], 0)

    for label, proximal, joint, distal, _delta in ANGLES:
        for side in ("R", "L"):
            angle = frontal_deviation(joints, proximal, joint, distal, side)
            if angle is not None:
                out[f"{label} {side}"] = angle
    return out


def report(male: dict[str, float], female: dict[str, float],
           tolerance: float) -> int:
    print(f"\n{'measure':<22}{'male':>8}{'female':>9}{'ratio':>8}"
          f"{'published':>11}{'error':>8}")
    worst = 0.0
    off: list[str] = []
    for name, (m_ref, f_ref, _unit) in PUBLISHED.items():
        if name not in male or name not in female or male[name] <= 0:
            continue
        ratio = female[name] / male[name]
        published = f_ref / m_ref
        error = ratio - published
        worst = max(worst, abs(error))
        flag = "  <--" if abs(error) > tolerance else ""
        if flag:
            off.append(name)
        print(f"{name:<22}{male[name]:>8.2f}{female[name]:>9.2f}{ratio:>8.3f}"
              f"{published:>11.3f}{error:>+8.3f}{flag}")

    print(f"\n{'proportion':<24}{'male':>9}{'published':>11}"
          f"{'female':>9}{'published':>11}")
    for label, num, den, m_ref, f_ref in RATIOS:
        if num not in male or den not in male or male[den] <= 0:
            continue
        m = male[num] / male[den]
        f = female[num] / female[den]
        print(f"{label:<24}{m:>9.3f}{m_ref:>11.3f}{f:>9.3f}{f_ref:>11.3f}")

    print(f"\n{'angle':<18}{'side':>5}{'male':>8}{'female':>9}"
          f"{'change':>9}{'published':>11}")
    for label, proximal, joint, distal, delta in ANGLES:
        for side in ("R", "L"):
            m = male.get(f"{label} {side}")
            f = female.get(f"{label} {side}")
            if m is None or f is None:
                continue
            print(f"{label:<18}{side:>5}{m:>8.1f}{f:>9.1f}{f - m:>+9.1f}"
                  f"{delta:>+11.1f}")

    print(f"\nworst ratio error {worst:+.3f}"
          + (f"; off by more than {tolerance:.3f}: {', '.join(off)}"
             if off else "; every ratio within tolerance"))
    return 1 if off else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="ratio error worth reporting (default 0.02)")
    args = ap.parse_args(argv)

    from tools.headless_loader import load_headless_scene

    hs = load_headless_scene()
    root = hs.named_nodes["bodyRoot"]
    joints = getattr(hs.pipeline.joint_setup, "joint_positions", {}) or {}
    morph = hs.pipeline.gender_morph
    if morph is None or not morph.loaded:
        raise SystemExit("the sex morph did not load; nothing to measure")

    male = measure(bone_points_by_name(root), joints)
    morph.set_gender(1.0)
    morph.scale_skeleton(root, joints)
    female = measure(bone_points_by_name(root), joints)
    return report(male, female, args.tolerance)


if __name__ == "__main__":
    raise SystemExit(main())
