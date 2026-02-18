"""CLI entry point for headless skinning diagnostics and parameter optimization.

Usage::

    # Diagnose current parameters (all layers, all poses):
    python -m tools.run_skinning_diagnostic --diagnose

    # Skin only, quick subset of poses:
    python -m tools.run_skinning_diagnostic --diagnose --layers skin --poses anatomical extreme_arm_raise

    # Optimize skin parameters:
    python -m tools.run_skinning_diagnostic --optimize --layers skin --max-iter 50

    # Optimize specific parameters only:
    python -m tools.run_skinning_diagnostic --optimize --layers skin --params spatial_limit chain_z_margin

    # Test specific parameter values:
    python -m tools.run_skinning_diagnostic --test --spatial-limit 8.0 --chain-z-margin 12.0 --layers skin

    # Save results to JSON:
    python -m tools.run_skinning_diagnostic --diagnose --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from faceforge.body.diagnostics import SkinningDiagnostic

logger = logging.getLogger(__name__)


# ── Parameter specification ───────────────────────────────────────────

@dataclass
class ParamSpec:
    """Specification for a tunable parameter."""
    name: str
    default: float
    low: float
    high: float
    # Which attribute on SoftTissueSkinning to set (instance or class attr)
    attr: str
    # Whether this is a register_skin_mesh kwarg instead of an instance attr
    is_register_kwarg: bool = False


PARAM_SPECS: list[ParamSpec] = [
    ParamSpec("spatial_limit",          10.0,  3.0,  25.0, "spatial_limit",          is_register_kwarg=True),
    ParamSpec("chain_z_margin",         15.0,  5.0,  30.0, "chain_z_margin",         is_register_kwarg=True),
    ParamSpec("min_spatial",             4.0,  1.0,  10.0, "min_spatial"),
    ParamSpec("spatial_factor",          0.35, 0.1,   0.8, "spatial_factor"),
    ParamSpec("min_z_pad",               2.0,  0.5,   8.0, "min_z_pad"),
    ParamSpec("lateral_threshold",       5.0,  2.0,  15.0, "lateral_threshold"),
    ParamSpec("midline_tolerance",       5.0,  2.0, 15.0, "midline_tolerance"),
    ParamSpec("CROSS_CHAIN_RADIUS",      8.0,  2.0, 20.0, "CROSS_CHAIN_RADIUS"),
    ParamSpec("MAX_CROSS_WEIGHT_MUSCLE", 0.5,  0.1,  0.9, "MAX_CROSS_WEIGHT_MUSCLE"),
    ParamSpec("MAX_CROSS_WEIGHT_OTHER",  0.25, 0.05, 0.5, "MAX_CROSS_WEIGHT_OTHER"),
    ParamSpec("BLEND_ZONE",              0.15, 0.05,  0.4, "BLEND_ZONE"),
    ParamSpec("DIVERGENCE_MIN",          5.0,  1.0,  15.0, "DIVERGENCE_MIN"),
    ParamSpec("DIVERGENCE_MAX",         20.0,  5.0,  50.0, "DIVERGENCE_MAX"),
]

PARAM_BY_NAME: dict[str, ParamSpec] = {p.name: p for p in PARAM_SPECS}
ALL_LAYERS = [
    "skin", "back_muscles", "shoulder_muscles", "arm_muscles",
    "torso_muscles", "hip_muscles", "leg_muscles", "organs", "vasculature",
]


def _params_to_dict(specs: list[ParamSpec], values: np.ndarray) -> dict[str, float]:
    """Convert parameter array to dict for register_layer."""
    d: dict[str, float] = {}
    for spec, val in zip(specs, values):
        d[spec.name] = float(val)
    return d


# ── Diagnose mode ────────────────────────────────────────────────────

def run_diagnose(
    layers: list[str],
    pose_names: list[str] | None,
    output_path: Path | None,
) -> dict:
    """Load scene, register layers with current params, run all poses, print report."""
    from tools.headless_loader import (
        load_headless_scene, load_layer, register_layer,
    )
    from tools.skinning_scorer import SkinningScorer, LAYER_WEIGHTS

    print("Loading headless scene...")
    t0 = time.time()
    hs = load_headless_scene()
    print(f"  Scene loaded in {time.time() - t0:.1f}s "
          f"({len(hs.skinning.joints)} joints, {len(hs.chain_ids)} chains)")

    # Load and register each layer
    for layer in layers:
        print(f"Loading layer: {layer}...")
        t1 = time.time()
        meshes = load_layer(hs, layer)
        register_layer(hs, meshes, layer)
        print(f"  {layer}: {len(meshes)} meshes registered in {time.time() - t1:.1f}s")

    # Score
    print(f"\nScoring across {len(SkinningScorer(hs, pose_names).poses)} poses...")
    scorer = SkinningScorer(hs, pose_names)
    result = scorer.evaluate()

    # Print report
    _print_evaluation_report(result)

    # Detailed diagnostic report (registration-time binding analysis)
    diag = SkinningDiagnostic(hs.skinning)
    reports = diag.analyze_bindings()
    anomalies = diag.check_displacements(max_displacement=5.0, relative=True)
    distortion = diag.check_mesh_distortion()
    static_verts = diag.check_static_vertices()
    print("\n" + diag.format_report(reports, anomalies, distortion, static_verts=static_verts))

    output = {
        "mode": "diagnose",
        "layers": layers,
        "evaluation": result.to_dict(),
    }

    if output_path:
        output_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {output_path}")

    return output


# ── Optimize mode ────────────────────────────────────────────────────

def run_optimize(
    layers: list[str],
    pose_names: list[str] | None,
    param_names: list[str] | None,
    max_iter: int,
    output_path: Path | None,
) -> dict:
    """Load scene, run Nelder-Mead optimization, output best params."""
    from tools.headless_loader import (
        load_headless_scene, load_layer, register_layer, reset_skinning,
    )
    from tools.skinning_scorer import SkinningScorer

    # Determine which parameters to optimize
    if param_names:
        specs = [PARAM_BY_NAME[n] for n in param_names if n in PARAM_BY_NAME]
        unknown = [n for n in param_names if n not in PARAM_BY_NAME]
        if unknown:
            print(f"Warning: unknown parameters ignored: {unknown}")
    else:
        specs = list(PARAM_SPECS)

    if not specs:
        print("Error: no valid parameters to optimize")
        sys.exit(1)

    print("Loading headless scene...")
    t0 = time.time()
    hs = load_headless_scene()
    print(f"  Scene loaded in {time.time() - t0:.1f}s")

    # Pre-load layer meshes (one-time cost)
    print("Pre-loading layer meshes...")
    layer_meshes: dict[str, list] = {}
    for layer in layers:
        t1 = time.time()
        meshes = load_layer(hs, layer)
        layer_meshes[layer] = meshes
        print(f"  {layer}: {len(meshes)} meshes ({time.time() - t1:.1f}s)")

    # Initial defaults
    x0 = np.array([s.default for s in specs])
    bounds_low = np.array([s.low for s in specs])
    bounds_high = np.array([s.high for s in specs])

    # Evaluate baseline
    print("\nEvaluating baseline (default parameters)...")
    _register_all(hs, layer_meshes, specs, x0)
    scorer = SkinningScorer(hs, pose_names)
    baseline = scorer.evaluate()
    print(f"  Baseline composite score: {baseline.composite:.4f}")

    # Optimization objective
    eval_count = [0]
    best_score = [baseline.composite]
    best_x = [x0.copy()]

    def objective(x: np.ndarray) -> float:
        # Penalty for out-of-bounds
        penalty = 0.0
        for i, (val, lo, hi) in enumerate(zip(x, bounds_low, bounds_high)):
            if val < lo:
                penalty += 100.0 * (lo - val) ** 2
            elif val > hi:
                penalty += 100.0 * (val - hi) ** 2
        if penalty > 0:
            return 1e6 + penalty

        reset_skinning(hs)
        _register_all(hs, layer_meshes, specs, x)
        s = SkinningScorer(hs, pose_names)
        result = s.evaluate()
        score = result.composite

        eval_count[0] += 1
        if score < best_score[0]:
            best_score[0] = score
            best_x[0] = x.copy()
            _print_progress(eval_count[0], score, specs, x)
        elif eval_count[0] % 5 == 0:
            print(f"  [{eval_count[0]:3d}] score={score:.4f} (best={best_score[0]:.4f})")

        return score

    # Try scipy Nelder-Mead, fall back to coordinate descent
    print(f"\nOptimizing {len(specs)} parameters over {max_iter} iterations...")
    try:
        from scipy.optimize import minimize
        result = minimize(
            objective, x0, method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 0.01, "fatol": 0.01},
        )
        final_x = result.x
        final_score = result.fun
        method_used = "Nelder-Mead"
    except ImportError:
        print("  scipy not available, using coordinate descent fallback")
        final_x, final_score = _coordinate_descent(
            objective, x0, bounds_low, bounds_high, max_iter,
        )
        method_used = "coordinate-descent"

    # Clamp to bounds
    final_x = np.clip(final_x, bounds_low, bounds_high)

    # Final evaluation with best params
    reset_skinning(hs)
    _register_all(hs, layer_meshes, specs, final_x)
    scorer = SkinningScorer(hs, pose_names)
    final_eval = scorer.evaluate()

    # Print comparison
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Method: {method_used}")
    print(f"Evaluations: {eval_count[0]}")
    print(f"\nBaseline score: {baseline.composite:.4f}")
    print(f"Optimized score: {final_eval.composite:.4f}")
    improvement = baseline.composite - final_eval.composite
    pct = 100.0 * improvement / baseline.composite if baseline.composite > 0 else 0
    print(f"Improvement: {improvement:.4f} ({pct:.1f}%)")

    print("\nParameter comparison:")
    print(f"  {'Parameter':<28s} {'Default':>10s} {'Optimized':>10s} {'Change':>10s}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}")
    for spec, val in zip(specs, final_x):
        diff = val - spec.default
        print(f"  {spec.name:<28s} {spec.default:10.4f} {val:10.4f} {diff:+10.4f}")

    _print_evaluation_report(final_eval, label="Optimized")

    output = {
        "mode": "optimize",
        "method": method_used,
        "evaluations": eval_count[0],
        "layers": layers,
        "baseline": baseline.to_dict(),
        "optimized": final_eval.to_dict(),
        "parameters": {spec.name: float(val) for spec, val in zip(specs, final_x)},
        "defaults": {spec.name: spec.default for spec in specs},
    }

    if output_path:
        output_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {output_path}")

    return output


# ── Test mode ────────────────────────────────────────────────────────

def run_test(
    layers: list[str],
    pose_names: list[str] | None,
    param_overrides: dict[str, float],
    output_path: Path | None,
) -> dict:
    """Register with user-specified param overrides, run diagnostics."""
    from tools.headless_loader import (
        load_headless_scene, load_layer, register_layer,
    )
    from tools.skinning_scorer import SkinningScorer

    print("Loading headless scene...")
    t0 = time.time()
    hs = load_headless_scene()
    print(f"  Scene loaded in {time.time() - t0:.1f}s")

    print(f"\nParameter overrides: {param_overrides}")

    for layer in layers:
        print(f"Loading layer: {layer}...")
        meshes = load_layer(hs, layer)
        register_layer(hs, meshes, layer, params=param_overrides)
        print(f"  {layer}: {len(meshes)} meshes registered")

    print(f"\nScoring...")
    scorer = SkinningScorer(hs, pose_names)
    result = scorer.evaluate()
    _print_evaluation_report(result)

    diag = SkinningDiagnostic(hs.skinning)
    reports = diag.analyze_bindings()
    anomalies = diag.check_displacements(max_displacement=5.0, relative=True)
    distortion = diag.check_mesh_distortion()
    static_verts = diag.check_static_vertices()
    print("\n" + diag.format_report(reports, anomalies, distortion, static_verts=static_verts))

    output = {
        "mode": "test",
        "layers": layers,
        "parameters": param_overrides,
        "evaluation": result.to_dict(),
    }

    if output_path:
        output_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {output_path}")

    return output


# ── Helpers ──────────────────────────────────────────────────────────

def _register_all(
    hs,
    layer_meshes: dict[str, list],
    specs: list[ParamSpec],
    values: np.ndarray,
) -> None:
    """Register all layers with the given parameter values."""
    from tools.headless_loader import register_layer

    params = _params_to_dict(specs, values)
    for layer, meshes in layer_meshes.items():
        register_layer(hs, meshes, layer, params=params)


def _print_progress(
    iteration: int, score: float,
    specs: list[ParamSpec], values: np.ndarray,
) -> None:
    """Print a compact progress line."""
    parts = [f"{s.name}={v:.3f}" for s, v in zip(specs, values)]
    params_str = ", ".join(parts[:4])
    if len(parts) > 4:
        params_str += f", ... (+{len(parts)-4} more)"
    print(f"  [{iteration:3d}] NEW BEST score={score:.4f}  ({params_str})")


def _print_evaluation_report(result, label: str = "Current") -> None:
    """Print a formatted evaluation summary."""
    print(f"\n--- {label} Evaluation ---")
    print(f"  Composite score:         {result.composite:.4f}")
    print(f"  Cross-body bindings:     {result.total_cross_body}")
    print(f"  Worst anomaly %:         {result.worst_anomaly_pct:.2f}%")
    print(f"  Worst max displacement:  {result.worst_max_displacement:.2f}")
    print(f"  Mean displacement:       {result.mean_mean_displacement:.2f}")
    print(f"  Worst stretched edges:   {result.worst_stretched_edges}")
    print(f"  Worst inverted tris:     {result.worst_inverted_tris}")
    print(f"  Worst max stretch ratio: {result.worst_max_stretch_ratio:.2f}")
    print(f"  Cross-chain stretched:   {result.total_cross_chain_stretched}")
    print(f"  Worst static vertices:   {result.worst_static_vertices}")
    print(f"\n  Per-pose breakdown:")
    for ps in result.per_pose:
        parts = []
        if ps.anomaly_count > 0:
            parts.append(f"{ps.anomaly_count} anomalies")
        if ps.stretched_edges > 0:
            parts.append(f"{ps.stretched_edges} stretched")
        if ps.inverted_tris > 0:
            parts.append(f"{ps.inverted_tris} inverted")
        if ps.static_vertex_count > 0:
            parts.append(f"{ps.static_vertex_count} static")
        status = ", ".join(parts) if parts else "OK"
        print(f"    {ps.pose_name:<25s}: {status}")


def _coordinate_descent(
    objective, x0: np.ndarray,
    bounds_low: np.ndarray, bounds_high: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, float]:
    """Simple coordinate-descent fallback when scipy is not available.

    Cycles through each parameter, trying steps of ±10% of the parameter
    range, accepting any improvement.
    """
    x = x0.copy()
    best = objective(x)
    ranges = bounds_high - bounds_low

    for iteration in range(max_iter):
        improved = False
        for i in range(len(x)):
            step = ranges[i] * 0.1
            # Try positive step
            x_try = x.copy()
            x_try[i] = min(bounds_high[i], x[i] + step)
            val = objective(x_try)
            if val < best:
                x = x_try
                best = val
                improved = True
                continue
            # Try negative step
            x_try = x.copy()
            x_try[i] = max(bounds_low[i], x[i] - step)
            val = objective(x_try)
            if val < best:
                x = x_try
                best = val
                improved = True

        if not improved:
            # Shrink step size
            ranges *= 0.5
            if np.all(ranges < 0.01):
                print(f"  Coordinate descent converged at iteration {iteration}")
                break

    return x, best


# ── Bone specificity ─────────────────────────────────────────────────


def _run_bone_test(
    layers: list[str],
    full: bool = False,
    muscle_threshold: float = 0.1,
    skin_threshold: float = 0.5,
    output_path: Path | None = None,
) -> None:
    """Load scene, register layers, run bone-specificity test."""
    from tools.bone_specificity import run_bone_specificity, format_report as fmt
    from tools.headless_loader import load_headless_scene, load_layer, register_layer

    print("Loading headless scene...")
    hs = load_headless_scene()

    for layer in layers:
        print(f"Loading layer: {layer}...")
        meshes = load_layer(hs, layer)
        register_layer(hs, meshes, layer)

    report = run_bone_specificity(
        hs,
        muscle_threshold=muscle_threshold,
        skin_threshold=skin_threshold,
        full=full,
    )
    print()
    print(fmt(report))

    if output_path is not None:
        import json
        data = {
            "muscle_violations": [
                {
                    "mesh": v.mesh_name,
                    "dof": v.dof_name,
                    "displaced": v.displaced_count,
                    "total": v.total_verts,
                    "max_disp": v.max_displacement,
                    "mean_disp": v.mean_displacement,
                }
                for v in report.muscle_violations
            ],
            "skin_violations": [
                {
                    "region": v.region_name,
                    "dof": v.dof_name,
                    "displaced": v.displaced_count,
                    "total": v.region_total,
                    "max_disp": v.max_displacement,
                    "mean_disp": v.mean_displacement,
                }
                for v in report.skin_violations
            ],
            "total_violations": report.total_violations,
        }
        output_path.write_text(json.dumps(data, indent=2))
        print(f"\nResults saved to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless skinning diagnostic and parameter optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--diagnose", action="store_true", default=True,
        help="Run diagnostics with current parameters (default mode)",
    )
    mode.add_argument(
        "--optimize", action="store_true",
        help="Optimize parameters via Nelder-Mead (or coordinate descent)",
    )
    mode.add_argument(
        "--test", action="store_true",
        help="Test with specific parameter values",
    )
    mode.add_argument(
        "--bone-test", action="store_true",
        help="Test bone specificity: verify vertices only respond to their associated bones",
    )

    parser.add_argument(
        "--layers", nargs="+", default=None, metavar="LAYER",
        help=f"Layers to evaluate. Choices: {', '.join(ALL_LAYERS)}. Default: skin",
    )
    parser.add_argument(
        "--poses", nargs="+", default=None, metavar="POSE",
        help="Subset of poses to test. Default: all 11",
    )
    parser.add_argument(
        "--output", type=Path, default=None, metavar="FILE",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    # Optimize-specific
    parser.add_argument(
        "--max-iter", type=int, default=50,
        help="Maximum optimization iterations (default: 50)",
    )
    parser.add_argument(
        "--params", nargs="+", default=None, metavar="PARAM",
        help=f"Parameters to optimize. Choices: {', '.join(PARAM_BY_NAME)}",
    )

    # Bone-test specific
    parser.add_argument(
        "--full-dofs", action="store_true",
        help="Test ALL DOFs (slower). Default: 6 quick DOFs per major joint.",
    )
    parser.add_argument(
        "--muscle-threshold", type=float, default=0.1,
        help="Displacement threshold for muscle specificity (default: 0.1 units)",
    )
    parser.add_argument(
        "--skin-threshold", type=float, default=0.5,
        help="Displacement threshold for skin specificity (default: 0.5 units)",
    )

    # Test-specific parameter overrides
    for spec in PARAM_SPECS:
        flag = "--" + spec.name.lower().replace("_", "-")
        parser.add_argument(
            flag, type=float, default=None, metavar="VAL",
            help=f"{spec.name} (default: {spec.default}, range: [{spec.low}, {spec.high}])",
        )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    layers = args.layers or ["skin"]
    # Validate layer names
    for layer in layers:
        if layer not in ALL_LAYERS:
            parser.error(f"Unknown layer: {layer!r}. Choices: {', '.join(ALL_LAYERS)}")

    if args.bone_test:
        _run_bone_test(
            layers=layers,
            full=args.full_dofs,
            muscle_threshold=args.muscle_threshold,
            skin_threshold=args.skin_threshold,
            output_path=args.output,
        )
    elif args.optimize:
        run_optimize(
            layers=layers,
            pose_names=args.poses,
            param_names=args.params,
            max_iter=args.max_iter,
            output_path=args.output,
        )
    elif args.test:
        # Collect parameter overrides from CLI flags
        overrides: dict[str, float] = {}
        for spec in PARAM_SPECS:
            attr = spec.name.lower()
            val = getattr(args, attr, None)
            if val is not None:
                overrides[spec.name] = val
        if not overrides:
            parser.error("--test requires at least one parameter override "
                         "(e.g. --spatial-limit 8.0)")
        run_test(
            layers=layers,
            pose_names=args.poses,
            param_overrides=overrides,
            output_path=args.output,
        )
    else:
        run_diagnose(
            layers=layers,
            pose_names=args.poses,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
