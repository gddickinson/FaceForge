"""Render the body skin from several viewpoints, coloured by defect.

Three defects, measured per vertex against the rest pose:

``stretch``
    The worst edge incident on the vertex, as a multiple of its rest length.
    This is what tearing looks like, and it is invariant to the rigid rotation
    a limb legitimately undergoes.
``spike``
    How much further the vertex travels than its mesh neighbours' mean.  Edge
    stretch does not see a long thin strip drawn off the body: only the few
    edges at its base stretch, while what reads on screen is the strip.
``armweight``
    The share of the vertex's influence weight held by arm-chain joints.  On
    the trunk that is the binding fault behind both of the above.

    PYTHONPATH=src python -m tools.skin_defect_views --out results/skin_now.png
    PYTHONPATH=src python -m tools.skin_defect_views --save before.npz
    PYTHONPATH=src python -m tools.skin_defect_views --baseline before.npz \\
        --out results/skin_before_after.png

``--save`` writes the measurements without rendering, so a baseline can be
captured, the engine changed, and the two drawn side by side.  Every number is
measured at run time.
"""

from __future__ import annotations

import argparse

import numpy as np

#: Poses to draw.  Shoulder abduction's range is 90 degrees, so 1.0 is
#: shoulder height.
POSES: dict[str, dict] = {
    "arms to shoulder height": {"shoulder_r_abduct": 1.0, "shoulder_l_abduct": 1.0},
    "hip hinge": {"hip_r_flex": 0.85, "hip_l_flex": 0.85,
                  "knee_r_flex": 0.35, "knee_l_flex": 0.35},
}

#: (axis pair, labels, title) for each viewpoint.
VIEWS = (
    ((0, 2), ("x (right)", "z (superior)"), "front"),
    ((1, 2), ("y (posterior +)", "z (superior)"), "left side"),
    ((0, 1), ("x (right)", "y (posterior +)"), "from above"),
)

SETTLE_FRAMES = 90
SPIKE = 5.0

METRICS = {
    "stretch": ("worst incident edge stretch", 1.0, 4.0, "inferno_r"),
    "spike": ("travel beyond the neighbours' mean (units)", 0.0, 20.0, "inferno_r"),
    "armweight": ("share of influence held by arm joints", 0.0, 1.0, "inferno_r"),
}


def _skin(soft):
    for b in soft.bindings:
        if not b.is_muscle and b.edge_pairs is not None:
            return b
    return None


def _measure(ctx, soft, b, pose: dict) -> dict:
    from faceforge.core.state import BodyState

    defaults = BodyState()
    for state in (ctx.state.body, ctx.state.target_body):
        for name, value in vars(defaults).items():
            if hasattr(state, name):
                setattr(state, name, value)
        for name, value in pose.items():
            setattr(state, name, value)
    for _ in range(SETTLE_FRAMES):
        ctx.simulation.step(1 / 60)

    ref = soft._resolved_reference(b)
    n = len(ref)
    pos = np.asarray(b.mesh.geometry.positions,
                     dtype=np.float64).reshape(-1, 3)[:n]
    disp = np.linalg.norm(pos - ref, axis=1)

    e = np.asarray(b.edge_pairs).reshape(-1, 2)
    e = e[(e[:, 0] < n) & (e[:, 1] < n)]
    rest_len = np.linalg.norm(ref[e[:, 0]] - ref[e[:, 1]], axis=1)
    keep = rest_len > 1e-6
    e, rest_len = e[keep], rest_len[keep]
    ratio = np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1) / rest_len
    stretch = np.zeros(n)
    np.maximum.at(stretch, e[:, 0], ratio)
    np.maximum.at(stretch, e[:, 1], ratio)

    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    count = np.bincount(src, minlength=n).astype(np.float64)
    total = np.bincount(src, weights=disp[dst], minlength=n)
    has = count > 0
    mean = np.zeros(n)
    mean[has] = total[has] / count[has]

    names = [j.name for j in soft.joints]
    is_arm = np.array([("shoulder" in nm or "elbow" in nm or "wrist" in nm
                        or "clavicle" in nm or "scapula" in nm
                        or nm.startswith("finger")) for nm in names])
    inf = getattr(b, "influences", None)
    if inf is not None and b.influence_weights is not None:
        armw = (np.asarray(b.influence_weights)[:n]
                * is_arm[np.asarray(inf)[:n]]).sum(axis=1)
    else:
        armw = is_arm[np.asarray(b.joint_indices)[:n]].astype(np.float64)

    return {
        "ref": ref.astype(np.float32),
        "pos": pos.astype(np.float32),
        "stretch": stretch.astype(np.float32),
        "spike": (disp - mean).astype(np.float32),
        "armweight": armw.astype(np.float32),
        "n_spikes": int(((disp - mean) > SPIKE).sum()),
        "n_torn": int((ratio > 2.0).sum()),
        "worst": float(ratio.max()),
    }


def _panel(ax, pts, value, metric, lim, title, step):
    label, lo, hi, cmap = METRICS[metric]
    sc = ax.scatter(pts[::step, lim[0]], pts[::step, lim[1]], s=0.25,
                    c=np.clip(value[::step], lo, hi), cmap=cmap, vmin=lo, vmax=hi)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    return sc


def _render(sets, metric, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(sets)
    fig, axes = plt.subplots(rows, len(VIEWS),
                             figsize=(5.0 * len(VIEWS), 5.6 * rows),
                             squeeze=False)
    sc = None
    for r, (tag, data) in enumerate(sets):
        for c, (pair, labels, view) in enumerate(VIEWS):
            ax = axes[r][c]
            sc = _panel(ax, data["pos"], data[metric], metric, pair,
                        f"{tag} — {view}", 4)
            ax.set_xlabel(labels[0], fontsize=7)
            ax.set_ylabel(labels[1], fontsize=7)
    cb = fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.02)
    cb.set_label(METRICS[metric][0], fontsize=8)
    fig.suptitle(f"Body skin coloured by {metric}", fontsize=13)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose", default="arms to shoulder height",
                    choices=sorted(POSES))
    ap.add_argument("--metric", default="spike", choices=sorted(METRICS))
    ap.add_argument("--out", default=None, help="figure path")
    ap.add_argument("--save", default=None, help="write the measurements and stop")
    ap.add_argument("--baseline", default=None,
                    help="a saved measurement to draw above the current one")
    ap.add_argument("--label", default="now")
    ap.add_argument("--min-spatial", type=float, default=None,
                    help="skinning.min_spatial, for capturing an earlier state")
    ap.add_argument("--seed-margin", type=float, default=None,
                    help="SoftTissueSkinning.SEED_CONFIDENCE_MARGIN")
    ap.add_argument("--bridge", type=float, default=None,
                    help="SoftTissueSkinning.GEODESIC_BRIDGE")
    args = ap.parse_args(argv)

    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])

    from faceforge.appcontext import build_app_context
    from faceforge.body.soft_tissue import SoftTissueSkinning
    from faceforge.controllers import build_controllers
    from faceforge.coordination.asset_load_sequence import AssetLoadSequence

    if args.seed_margin is not None:
        SoftTissueSkinning.SEED_CONFIDENCE_MARGIN = args.seed_margin
    if args.bridge is not None:
        SoftTissueSkinning.GEODESIC_BRIDGE = args.bridge
    ctx = build_app_context(argv=[])
    controllers = build_controllers(ctx)
    AssetLoadSequence(ctx).run()
    soft = ctx.simulation.soft_tissue
    if args.min_spatial is not None:
        soft.min_spatial = args.min_spatial
    controllers.loaders.load_skin()
    b = _skin(soft)
    if b is None:
        print("no skin binding with edges -- nothing to draw")
        return 1

    data = _measure(ctx, soft, b, POSES[args.pose])
    print(f"{args.pose}: {data['n_spikes']} spikes, {data['n_torn']} torn edges, "
          f"worst edge {data['worst']:.2f}")

    if args.save:
        np.savez_compressed(args.save, **{k: v for k, v in data.items()
                                          if isinstance(v, np.ndarray)})
        print("wrote", args.save)
        return 0

    sets = []
    if args.baseline:
        base = np.load(args.baseline)
        sets.append(("before", {k: base[k] for k in base.files}))
    sets.append((args.label, data))
    out = args.out or f"results/skin_{args.metric}.png"
    _render(sets, args.metric, out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
