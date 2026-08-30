"""``faceforge-cli`` -- the scriptable half of FaceForge.

Every subcommand here is driven by a :mod:`faceforge.core.scene_state` file, so
a figure in a paper regenerates from a file that can sit in git next to the
manuscript::

    faceforge-cli render --state figures/fig3a.state.json --out fig3a.png
    faceforge-cli batch  --states figures/ --out build/figures/
    faceforge-cli scan   --state figures/fig3a.state.json --out ct.png --mode ct
    faceforge-cli export --state figures/fig3a.state.json --out fig3a.glb
    faceforge-cli verify-assets

Why this is ``faceforge-cli`` and not ``faceforge``
--------------------------------------------------
``faceforge`` already exists and launches the GUI.  Three reasons it stays that
way:

1. Bare ``faceforge`` today opens a window.  Making it a subcommand dispatcher
   means either breaking that (every existing user, every shortcut, every doc)
   or keeping the GUI as a hidden default -- in which case a mistyped
   subcommand launches a GUI instead of printing an error, and ``--help`` has
   to explain a command that is both a dispatcher and an application.
2. ``faceforge.app`` imports PySide6 at module scope.  A CLI whose entry
   module imports Qt cannot run on a machine that has no Qt, which is exactly
   the machine the CLI exists for (a cluster node, a CI runner, a Makefile).
   ``tests/session/test_cli.py`` asserts that importing this module leaves
   ``PySide6`` out of ``sys.modules``.
3. The GUI entry point is unchanged, so this addition cannot regress it.

``faceforge-cli gui`` is provided so that one command is still discoverable
from the other; it delegates to ``faceforge.app:main`` and is the only
subcommand that imports Qt.

Exit codes
----------
``0`` success.  ``2`` argument error (argparse).  ``3`` the operation failed --
no GL context, missing assets, an unrenderable state.  ``4`` the anatomy
configs have drifted from the ones the state was captured against, and
``--require-config-match`` was given.  Nothing is ever written on a non-zero
exit: a blank or partial figure that exits 0 is the failure mode this whole
path is built to avoid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("faceforge.cli")

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAILED = 3
EXIT_CONFIG_DRIFT = 4

#: Output size bounds.  Wider than tools/capture_golden.py's 64..4096 at the
#: bottom (a 16px thumbnail is a legitimate smoke test) and at the top (an
#: 8192px figure is a legitimate print deliverable, and the FBO either
#: allocates or fails loudly).
MIN_SIZE, MAX_SIZE = 16, 8192

DEFAULT_SIZE = "512x512"


# ---------------------------------------------------------------------------
# Argument value parsing.  Pure functions, no GL, no filesystem -- so the
# fast test tier can exercise all of it.
# ---------------------------------------------------------------------------


def parse_size(text: str) -> tuple[int, int]:
    """Parse ``WxH``.  Raises ``ValueError`` on anything else."""
    parts = str(text).lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"size must look like WxH, got {text!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"size must be two integers, got {text!r}") from exc
    for label, value in (("width", width), ("height", height)):
        if not MIN_SIZE <= value <= MAX_SIZE:
            raise ValueError(f"{label} {value} outside [{MIN_SIZE}, {MAX_SIZE}]")
    return width, height


def as_vec3(values: Sequence[float]) -> tuple[float, float, float]:
    """Validate a three-component coordinate."""
    if len(values) != 3:
        raise ValueError(f"expected three numbers, got {list(values)!r}")
    x, y, z = (float(v) for v in values)
    for name, value in (("x", x), ("y", y), ("z", z)):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{name} is not finite: {value!r}")
    return x, y, z


def parse_list(text: str | None) -> tuple[str, ...]:
    """Parse a comma-separated list, dropping empties."""
    if not text:
        return ()
    return tuple(p.strip() for p in text.split(",") if p.strip())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _render_mode_names() -> tuple[str, ...]:
    """The RenderMode names, read from the enum so the CLI cannot drift from it."""
    from faceforge.core.scene_state import render_mode_names

    return render_mode_names()


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


class CLIError(RuntimeError):
    """A failure that should be reported as a message and exit code 3."""


def _add_scene_source(parser: argparse.ArgumentParser, *, state_required: bool) -> None:
    """The options that decide where the *geometry* comes from.

    A state file records which structure sat at which scene-graph path and the
    BodyParts3D id each came from, so by default the geometry is rebuilt from
    the state itself and no other argument is needed.  ``--tier`` / ``--layers``
    / ``--structures`` are for states captured against a scene this rebuild
    cannot reproduce -- see ``faceforge.session.build_scene_from_state`` for
    the two things a state file does not record (node transforms, and whether
    the BP3D coordinate transform was applied).
    """
    parser.add_argument(
        "--state", type=Path, required=state_required, metavar="FILE",
        help="SceneState JSON file (as written by the app or Session.save_state)",
    )
    parser.add_argument(
        "--tier", type=int, default=None, metavar="N",
        help="build the scene by loading tier N (0, 1 or 2) instead of rebuilding "
             "it from the state's provenance",
    )
    parser.add_argument(
        "--structures", default=None, metavar="IDS",
        help="comma-separated BodyParts3D ids to load instead (e.g. FMA52734,FMA52748)",
    )
    parser.add_argument(
        "--layers", default=None, metavar="NAMES",
        help="comma-separated on-demand layers to add (organs, vasculature, "
             "arm_muscles, ...); see --list-layers",
    )
    parser.add_argument(
        "--skull-mode", default="original", choices=("original", "bp3d"),
        help="skull variant for --tier loading",
    )
    parser.add_argument(
        "--stl-dir", type=Path, default=None, metavar="DIR",
        help="override the STL directory (default: assets/stl)",
    )
    parser.add_argument(
        "--transform", default="none", choices=("none", "bp3d"),
        help="coordinate transform to apply when rebuilding geometry from a "
             "state file: 'none' for raw STL coordinates (a script-built scene), "
             "'bp3d' for the app's BP3D->skull transform",
    )
    parser.add_argument(
        "--visible-only", action="store_true",
        help="rebuild only the structures the state records as visible",
    )
    parser.add_argument(
        "--no-config-check", action="store_true",
        help="skip the anatomy-config fingerprint check on load",
    )
    parser.add_argument(
        "--require-config-match", action="store_true",
        help="exit 4 if the anatomy configs have changed since the state was "
             "captured; the default is to warn and continue",
    )


def _transform_for(args: argparse.Namespace) -> Any:
    if getattr(args, "transform", "none") != "bp3d":
        return None
    from faceforge.loaders.stl_batch_loader import CoordinateTransform

    return CoordinateTransform.from_config()


def _load_state(args: argparse.Namespace) -> tuple[Any, str | None]:
    """Load the state file, returning ``(state, config_drift_detail)``.

    The drift detail is captured from the ``ConfigFingerprintMismatch`` warning
    that :func:`faceforge.core.scene_state.codec.verify_config` emits, so the
    CLI can report it in its JSON manifest and, with
    ``--require-config-match``, refuse to render against it.  The mechanism
    already exists; this only makes it visible from a command line.
    """
    from faceforge.core.scene_state import ConfigFingerprintMismatch, SceneStateError, codec

    path = args.state
    if path is None:
        return None, None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConfigFingerprintMismatch)
            state = codec.load(path, check_config=not args.no_config_check)
    except SceneStateError as exc:
        raise CLIError(f"{path}: {exc}") from exc
    except OSError as exc:
        raise CLIError(f"cannot read {path}: {exc}") from exc

    drift = None
    for warning in caught:
        if issubclass(warning.category, ConfigFingerprintMismatch):
            drift = str(warning.message)
    if drift:
        print(f"WARNING: {drift}", file=sys.stderr)
        if args.require_config_match:
            raise _ConfigDrift(drift)
    return state, drift


class _ConfigDrift(RuntimeError):
    """--require-config-match was given and the configs have drifted."""


def _scene_kwargs(args: argparse.Namespace, state: Any) -> dict[str, Any]:
    structures = parse_list(args.structures)
    chosen = sum(x is not None and x != () for x in (args.tier, structures or None))
    if chosen > 1:
        raise CLIError("pass --tier or --structures, not both")
    kwargs: dict[str, Any] = {
        "layers": parse_list(args.layers),
        "skull_mode": args.skull_mode,
        "stl_dir": args.stl_dir,
        "transform": _transform_for(args),
        "visible_only": args.visible_only,
    }
    if args.tier is not None:
        kwargs["tier"] = args.tier
    elif structures:
        kwargs["structures"] = structures
    elif state is not None:
        kwargs["state"] = state
    else:
        raise CLIError("nothing to build: pass --state, --tier or --structures")
    return kwargs


def _apply_mode_override(scene: Any, mode: str) -> None:
    """Force every structure in *scene* to one render mode."""
    from faceforge.core.material import RenderMode

    target = RenderMode[mode]
    for mesh in scene.subtree_meshes():
        mesh.material.render_mode = target


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def cmd_render(args: argparse.Namespace) -> int:
    from faceforge import session as fs

    state, drift = _load_state(args)
    scene_kwargs = _scene_kwargs(args, state)
    size = parse_size(args.size) if args.size else None

    with fs.Session.create(
        width=size[0] if size else 512,
        height=size[1] if size else 512,
        prefer=args.prefer,
    ) as session:
        scene, anatomy = fs.build_scene(**scene_kwargs)
        session.adopt_scene(scene)

        apply_report = None
        if state is not None:
            apply_report = session.apply_state(
                state, strict=args.strict, resize_to_state=size is None,
            )
            if size is not None:
                # An explicit --size means "the same shot at a different
                # resolution", so the aspect is recomputed for the new pixel
                # size rather than kept from the state.
                session.resize(size[0], size[1], update_aspect=True)
        elif size is not None:
            session.resize(size[0], size[1])

        if args.mode:
            _apply_mode_override(scene, args.mode)

        image = session.render()
        out = fs.write_png(args.out, image)
        manifest = {
            "tool": "faceforge-cli render",
            "out": str(out),
            "sha256": sha256_file(out),
            "size": list(session.size),
            "state": None if args.state is None else str(args.state),
            "state_digest": None if state is None else _state_digest(state),
            "structures": anatomy.structures,
            "anatomy": anatomy.summary(),
            "render_mode_override": args.mode,
            "content_fraction": session.last_content_fraction,
            "config_drift": drift,
            "apply": None if apply_report is None else {
                "ok": bool(apply_report.ok),
                "summary": apply_report.summary(),
            },
            "gl": session.info().as_dict(),
        }

    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"wrote {out}  {manifest['size'][0]}x{manifest['size'][1]}  "
              f"{manifest['structures']} structures  "
              f"content={manifest['content_fraction']:.2%}")
        print(f"  sha256 {manifest['sha256']}")
    return EXIT_OK


def _state_digest(state: Any) -> str:
    from faceforge.core.scene_state import codec

    return codec.payload_digest(state)


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


def cmd_batch(args: argparse.Namespace) -> int:
    from faceforge import session as fs
    from faceforge.core.scene_state import SceneStateError

    states_dir = Path(args.states)
    if not states_dir.is_dir():
        raise CLIError(f"--states {states_dir} is not a directory")
    files = sorted(states_dir.glob(args.glob))
    if not files:
        raise CLIError(f"no files match {args.glob!r} in {states_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = parse_size(args.size) if args.size else None

    results: list[dict[str, Any]] = []
    failures: list[str] = []

    # One session for the whole batch: acquiring a context per state would be
    # the two-session hazard the Session guard exists to prevent, and would
    # recompile 16 shader programs per figure.
    with fs.Session.create(
        width=size[0] if size else 512,
        height=size[1] if size else 512,
        prefer=args.prefer,
    ) as session:
        for path in files:
            per_file = argparse.Namespace(**vars(args))
            per_file.state = path
            try:
                state, drift = _load_state(per_file)
                scene, anatomy = fs.build_scene(**_scene_kwargs(per_file, state))
                session.adopt_scene(scene)
                report = session.apply_state(
                    state, strict=args.strict, resize_to_state=size is None,
                )
                if size is not None:
                    session.resize(size[0], size[1], update_aspect=True)
                if args.mode:
                    _apply_mode_override(scene, args.mode)
                out_path = out_dir / f"{path.stem.split('.')[0]}.png"
                image = session.render()
                fs.write_png(out_path, image)
                results.append({
                    "state": str(path),
                    "out": str(out_path),
                    "sha256": sha256_file(out_path),
                    "size": list(session.size),
                    "structures": anatomy.structures,
                    "content_fraction": session.last_content_fraction,
                    "apply_ok": bool(report.ok),
                    "config_drift": drift,
                })
                print(f"  {path.name} -> {out_path.name}  "
                      f"{session.size[0]}x{session.size[1]}  "
                      f"{anatomy.structures} structures")
            except _ConfigDrift:
                raise
            except (fs.SessionError, SceneStateError, CLIError, OSError,
                    ValueError, KeyError) as exc:
                message = f"{path.name}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"  FAILED {message}", file=sys.stderr)
                if not args.continue_on_error:
                    raise CLIError(
                        f"{message}\nAborting the batch.  Pass --continue-on-error "
                        "to render the rest and report the failures at the end."
                    ) from exc
        gl_info = session.info().as_dict()

    manifest = {
        "tool": "faceforge-cli batch",
        "states_dir": str(states_dir),
        "out_dir": str(out_dir),
        "requested": len(files),
        "rendered": len(results),
        "failed": failures,
        "gl": gl_info,
        "images": results,
    }
    manifest_path = out_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"rendered {len(results)}/{len(files)} state file(s) into {out_dir}")
        print(f"  manifest {manifest_path}")
    return EXIT_FAILED if failures else EXIT_OK


# ---------------------------------------------------------------------------
# scan -- the virtual CT / MRI / X-ray scanner.  No GL context needed.
# ---------------------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    from faceforge import session as fs

    state, _drift = _load_state(args)
    scene, anatomy = fs.build_scene(**_scene_kwargs(args, state))

    image = fs.scan_scene(
        scene,
        origin=as_vec3(args.position),
        orientation=args.orientation,
        width=args.width,
        height=args.height,
        depth=args.depth,
        resolution=args.resolution,
        mode=args.mode,
        reduction=args.reduction,
    )
    rgb = fs.scan_to_rgb(image, args.mode)
    out = fs.write_png(args.out, rgb)

    if args.npy:
        import numpy as np

        np.save(args.npy, image)

    nonzero = float((image > 0).mean()) if image.ndim == 2 else float(
        (image.sum(axis=2) > 0).mean()
    )
    manifest = {
        "tool": "faceforge-cli scan",
        "out": str(out),
        "sha256": sha256_file(out),
        "raw": None if args.npy is None else str(args.npy),
        "mode": args.mode,
        "reduction": args.reduction,
        "orientation": args.orientation,
        "position": list(as_vec3(args.position)),
        "field_mm": {"width": args.width, "height": args.height, "depth": args.depth},
        "resolution": args.resolution,
        "structures": anatomy.structures,
        "hit_fraction": nonzero,
    }
    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"wrote {out}  {args.resolution}x{args.resolution}  mode={args.mode} "
              f"reduction={args.reduction}  {anatomy.structures} structures  "
              f"hits={nonzero:.2%}")
    if nonzero == 0.0:
        print(
            "WARNING: no ray hit any geometry -- the scan plane is outside the "
            "subject.  Check --position against the scene's own coordinates.",
            file=sys.stderr,
        )
    return EXIT_OK


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    from faceforge import session as fs
    from faceforge.export.glb_exporter import export_glb

    state, _drift = _load_state(args)
    scene, anatomy = fs.build_scene(**_scene_kwargs(args, state))
    scene.update()

    if args.format != "glb":
        # OBJ/PLY/STL live in mesh_export, which also writes the provenance
        # sidecar.  `mesh` is the fuller front end for them; this branch keeps
        # `export --format` from lying about what it supports.
        from faceforge.export.mesh_export import MeshExportError, export_mesh

        try:
            result = export_mesh(scene, args.out, args.format)
        except MeshExportError as exc:
            raise CLIError(str(exc)) from exc
        manifest = {
            "tool": "faceforge-cli export",
            **result.as_dict(),
            "sha256": sha256_file(result.path),
            "structures": anatomy.structures,
        }
        if args.json:
            print(json.dumps(manifest, indent=2))
        else:
            print(f"wrote {result.path}  {result.summary()}")
        return EXIT_OK

    count = export_glb(scene, args.out)
    if count == 0:
        raise CLIError(
            "no visible meshes were exported.  An empty GLB is not an export; "
            "check that the state's structures are visible, or drop --visible-only."
        )
    out = Path(args.out)
    manifest = {
        "tool": "faceforge-cli export",
        "format": args.format,
        "out": str(out),
        "sha256": sha256_file(out),
        "bytes": out.stat().st_size,
        "meshes": count,
        "structures": anatomy.structures,
    }
    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"wrote {out}  {count} mesh(es)  {manifest['bytes'] / 1e6:.1f} MB")
    return EXIT_OK


# ---------------------------------------------------------------------------
# verify-assets, gui
# ---------------------------------------------------------------------------


def cmd_verify_assets(args: argparse.Namespace) -> int:
    """Delegate to ``tools/fetch_assets.py``, which already does this properly."""
    from faceforge.session import repo_tool_module

    try:
        fetch_assets = repo_tool_module("fetch_assets")
    except Exception as exc:                     # noqa: BLE001 - reported below
        raise CLIError(str(exc)) from exc
    passthrough = list(args.passthrough or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return int(fetch_assets.main(passthrough or ["verify"]))


def cmd_gui(args: argparse.Namespace) -> int:
    """Launch the GUI.  The only subcommand that imports Qt."""
    from faceforge.app import main as gui_main

    result = gui_main()
    return EXIT_OK if result is None else int(result)


def cmd_list_layers(args: argparse.Namespace) -> int:
    from faceforge.session import LAYER_LOADERS, TIER_LOADS

    print("tiers (--tier N):")
    for tier, description in sorted(TIER_LOADS.items()):
        print(f"  {tier}  {description}")
    print("\nlayers (--layers a,b,...):")
    for name, (method, method_args) in sorted(LAYER_LOADERS.items()):
        detail = f"{method}{list(method_args) if method_args else '()'}"
        print(f"  {name:22s} AssetManager.{detail}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_EPILOG = """\
examples:
  faceforge-cli render --state fig3a.state.json --out fig3a.png
  faceforge-cli render --state fig3a.state.json --out fig3a.png --size 2048x2048
  faceforge-cli render --state fig3a.state.json --out xray.png --mode XRAY
  faceforge-cli batch  --states figures/ --out build/ --size 1024x1024
  faceforge-cli scan   --state fig3a.state.json --out ct.png --mode ct \\
                       --orientation axial --position 0 -91 1556 --resolution 256
  faceforge-cli export --state fig3a.state.json --out fig3a.glb
  faceforge-cli verify-assets -- --json

reproducibility:
  `render` twice on the same state file produces byte-identical PNGs, and a
  state file carries the BodyParts3D id of every structure it contains, so the
  scene is rebuilt from the file itself with no other argument.  What a state
  file does NOT record: per-node transforms (a posed scene), and whether the
  BP3D coordinate transform was applied to the geometry (--transform).

not exposed here:
  video export (faceforge.export.video_export) needs a live Qt GL widget to
  grab frames from, so it cannot run headlessly; use `faceforge-cli gui`.
"""


# ---------------------------------------------------------------------------
# still -- true-resolution offscreen publication stills
# ---------------------------------------------------------------------------

#: ``still`` deliberately does not reuse ``parse_size``'s 8192 ceiling: the
#: whole point of the command is that the real limit is the GL
#: implementation's, and that exceeding it must fail with the limit named
#: rather than against an arbitrary constant.  This is only a sanity ceiling to
#: keep a typo from asking for a terabyte of framebuffer.
STILL_SANITY_MAX = 65536


def parse_still_size(text: str) -> tuple[int, int]:
    """Parse ``WxH`` for a still, bounded only by sanity and by GL."""
    from faceforge.export.still import MIN_STILL_SIZE

    parts = str(text).lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"size must look like WxH, got {text!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"size must be two integers, got {text!r}") from exc
    for label, value in (("width", width), ("height", height)):
        if not MIN_STILL_SIZE <= value <= STILL_SANITY_MAX:
            raise ValueError(
                f"{label} {value} outside [{MIN_STILL_SIZE}, {STILL_SANITY_MAX}]"
            )
    return width, height


def cmd_still(args: argparse.Namespace) -> int:
    """Render a still *at* the requested size through an FBO.

    Compare ``render``: identical pixels for an identical size, but this
    command bounds-checks against the driver's limits first, reports them, and
    can measure the result against a bicubic upscale of a smaller render --
    which is what the old grab-and-scale screenshot path actually produced.
    """
    from faceforge import session as fs
    from faceforge.export import still as st

    state, drift = _load_state(args)
    scene_kwargs = _scene_kwargs(args, state)
    width, height = parse_still_size(args.size)

    with fs.Session.create(width=512, height=512, prefer=args.prefer) as session:
        limits = session.gl_size_limits()
        if args.limits_only:
            print(json.dumps({
                "tool": "faceforge-cli still --limits-only",
                "gl": session.info().as_dict(),
                "gl_limits": limits.as_dict(),
            }, indent=2))
            return EXIT_OK

        # Before anything is allocated.  session.resize() below goes straight
        # to glTexImage2D, which reports an over-large request as a bare
        # GLError -- and leaves the session holding a destroyed framebuffer.
        # Checking here turns that into the message that names the limit.
        try:
            limits.check(width, height)
        except st.StillSizeError as exc:
            raise CLIError(str(exc)) from exc

        scene, anatomy = fs.build_scene(**scene_kwargs)
        session.adopt_scene(scene)
        framing = None
        if state is not None:
            session.apply_state(state, strict=args.strict, resize_to_state=False)
        else:
            # No state means no camera, and the default camera does not look at
            # BodyParts3D geometry (which sits ~1.5 m along +Z).  Frame it, or
            # the still is a picture of the background.
            framing = st.frame_scene(session)
        if args.mode:
            _apply_mode_override(scene, args.mode)
        session.resize(width, height, update_aspect=True)

        try:
            result = session.export_still(args.out, width, height)
        except st.StillSizeError as exc:
            # Reported as a usage error, with the GL limit named, and nothing
            # written: a truncated still that exits 0 is the failure this
            # command exists to prevent.
            raise CLIError(str(exc)) from exc

        manifest: dict[str, Any] = {
            "tool": "faceforge-cli still",
            **result.as_dict(),
            "sha256": sha256_file(result.path),
            "structures": anatomy.structures,
            "framing": framing,
            "config_drift": drift,
            "gl": session.info().as_dict(),
        }

        if args.prove_resolution:
            factor = int(args.prove_factor)
            if factor < 2:
                raise CLIError("--prove-factor must be at least 2")
            if width % factor or height % factor:
                raise CLIError(
                    f"--prove-resolution needs --size divisible by "
                    f"--prove-factor on both axes; {width}x{height} is not "
                    f"divisible by {factor}.  The comparison rests on one "
                    "Nyquist cutoff, which a fractional factor does not have."
                )
            ref_w, ref_h = width // factor, height // factor
            if min(ref_w, ref_h) < st.MIN_STILL_SIZE:
                raise CLIError(
                    f"the {ref_w}x{ref_h} reference render implied by "
                    f"--prove-factor {factor} is below the {st.MIN_STILL_SIZE}px "
                    "floor; use a larger --size or a smaller --prove-factor"
                )
            reference = st.render_still(session, ref_w, ref_h)
            large = st.render_still(session, width, height)
            manifest["resolution_evidence"] = st.resolution_evidence(
                reference, large)
            session.resize(width, height, update_aspect=True)

    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"wrote {result.path}  {result.width}x{result.height}  "
              f"{result.megapixels:.2f} MP  {result.bytes_written / 1e6:.2f} MB")
        print(f"  rasterised at full resolution (no upscale); GL ceiling "
              f"{limits.max_width}x{limits.max_height}")
        evidence = manifest.get("resolution_evidence")
        if evidence:
            band = evidence["band_energy_above_small_nyquist"]
            print(f"  band energy above the {evidence['small_size'][0]}px "
                  f"render's Nyquist: true {band['true_render']:.4%} vs "
                  f"bicubic upscale {band['bicubic_upscale']:.4%} "
                  f"({band['ratio']:.1f}x)")
    return EXIT_OK


# ---------------------------------------------------------------------------
# mesh -- OBJ / PLY / STL / GLB interchange
# ---------------------------------------------------------------------------


def cmd_mesh(args: argparse.Namespace) -> int:
    from faceforge import session as fs
    from faceforge.export.mesh_export import MeshExportError, export_mesh

    state, _drift = _load_state(args)
    scene, anatomy = fs.build_scene(**_scene_kwargs(args, state))

    try:
        result = export_mesh(scene, args.out, args.format,
                             sidecar=not args.no_sidecar)
    except MeshExportError as exc:
        raise CLIError(str(exc)) from exc

    manifest = {
        "tool": "faceforge-cli mesh",
        **result.as_dict(),
        "sha256": sha256_file(result.path),
        "structures": anatomy.structures,
    }
    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"wrote {result.path}  {result.summary()}")
        if result.sidecar:
            print(f"  provenance sidecar {result.sidecar.name}")
        for note in result.notes:
            print(f"  note: {note}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# volume -- DICOM / NIfTI from the virtual scanner
# ---------------------------------------------------------------------------


def cmd_volume(args: argparse.Namespace) -> int:
    from faceforge import session as fs
    from faceforge.export.hounsfield import HUMappingError
    from faceforge.export.provenance import collect_provenance
    from faceforge.export.volume import VolumeError, scan_volume

    state, _drift = _load_state(args)
    scene, anatomy = fs.build_scene(**_scene_kwargs(args, state))
    scene.update()
    records = collect_provenance(scene.collect_meshes())

    try:
        volume = scan_volume(
            scene,
            orientation=args.orientation,
            centre=as_vec3(args.position),
            field_width=args.width,
            field_height=args.height,
            resolution=args.resolution,
            slices=args.slices,
            slice_spacing=args.slice_spacing,
            slab_depth=args.slice_thickness,
            mode=args.mode,
            reduction=args.reduction,
            transform_applied=args.transform,
        )
    except VolumeError as exc:
        raise CLIError(str(exc)) from exc

    manifest: dict[str, Any] = {
        "tool": "faceforge-cli volume",
        "format": args.format,
        "structures": anatomy.structures,
        "volume": volume.as_dict(),
    }

    try:
        if args.format == "dicom":
            from faceforge.export.dicom import export_dicom_series

            result = export_dicom_series(
                volume, args.out, hu_mode=args.hu_mode, provenance=records,
                sidecar=not args.no_sidecar,
            )
        else:
            from faceforge.export.nifti import export_nifti

            result = export_nifti(
                volume, args.out, hu_mode=args.hu_mode, provenance=records,
                sidecar=not args.no_sidecar,
            )
    except HUMappingError as exc:
        raise CLIError(str(exc)) from exc
    manifest.update(result.as_dict())

    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        if args.format == "dicom":
            print(f"wrote {result.slices} DICOM slices to {result.directory}  "
                  f"modality={result.modality}  "
                  f"RescaleType={result.rescale.rescale_type}")
        else:
            print(f"wrote {result.path}  shape={result.shape}  "
                  f"dtype={result.dtype}  axes={''.join(result.axis_codes)}")
        print(f"  pixel values: {result.rescale.unit}")
        for note in result.notes:
            print(f"  note: {note}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    modes = _render_mode_names()
    parser = argparse.ArgumentParser(
        prog="faceforge-cli",
        description="Headless FaceForge: render, batch, scan and export from "
                    "SceneState files.  `faceforge` (no -cli) is the GUI.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log at DEBUG")
    parser.add_argument("-q", "--quiet", action="store_true", help="log warnings only")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # -- render ----------------------------------------------------------
    p_render = sub.add_parser(
        "render", help="render one SceneState file to a PNG",
        description="Render a SceneState file to a PNG through an offscreen "
                    "framebuffer.  Deterministic: the same state file renders "
                    "byte-identical PNGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_scene_source(p_render, state_required=True)
    p_render.add_argument("--out", type=Path, required=True, metavar="PNG",
                          help="output PNG path")
    p_render.add_argument("--size", default=None, metavar="WxH",
                          help="output size (default: the state's own viewport)")
    p_render.add_argument("--mode", default=None, choices=modes, metavar="MODE",
                          help=f"force every structure to one render mode; one of "
                               f"{', '.join(modes)}")
    p_render.add_argument("--prefer", default="auto",
                          choices=("auto", "hardware", "software"),
                          help="GL context preference; 'software' is reproducible "
                               "across machines with different GPUs")
    p_render.add_argument("--strict", action="store_true",
                          help="fail if the scene's structure set does not match "
                               "the state exactly")
    p_render.add_argument("--json", action="store_true",
                          help="print a machine-readable manifest")
    p_render.set_defaults(func=cmd_render)

    # -- batch -----------------------------------------------------------
    p_batch = sub.add_parser(
        "batch", help="render every SceneState file in a directory",
        description="Render many state files with one GL context and one "
                    "renderer.  Writes batch_manifest.json alongside the images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_scene_source(p_batch, state_required=False)
    p_batch.add_argument("--states", type=Path, required=True, metavar="DIR",
                         help="directory of SceneState files")
    p_batch.add_argument("--out", type=Path, required=True, metavar="DIR",
                         help="output directory for the PNGs")
    p_batch.add_argument("--glob", default="*.state.json", metavar="PATTERN",
                         help="which files in --states to render")
    p_batch.add_argument("--size", default=None, metavar="WxH",
                         help="output size (default: each state's own viewport)")
    p_batch.add_argument("--mode", default=None, choices=modes, metavar="MODE",
                         help="force every structure to one render mode")
    p_batch.add_argument("--prefer", default="auto",
                         choices=("auto", "hardware", "software"))
    p_batch.add_argument("--strict", action="store_true")
    p_batch.add_argument("--continue-on-error", action="store_true",
                         help="render the remaining states after a failure and "
                              "report them all at the end (exit 3)")
    p_batch.add_argument("--json", action="store_true")
    p_batch.set_defaults(func=cmd_batch)

    # -- scan ------------------------------------------------------------
    from faceforge.session import SCAN_ORIENTATIONS, SCAN_REDUCTIONS, scan_modes

    p_scan = sub.add_parser(
        "scan", help="virtual CT / MRI / X-ray cross-section",
        description="Cast rays through the scene with ScannerEngine and write "
                    "the cross-section as a PNG.  Needs no GL context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_scene_source(p_scan, state_required=False)
    p_scan.add_argument("--out", type=Path, required=True, metavar="PNG")
    p_scan.add_argument("--mode", default="ct", choices=sorted(scan_modes()),
                        help="imaging modality")
    p_scan.add_argument("--reduction", default="mean", choices=SCAN_REDUCTIONS,
                        help="how hits along a ray combine ('sum' is an "
                             "exponential-attenuation projection)")
    p_scan.add_argument("--orientation", default="axial",
                        choices=sorted(SCAN_ORIENTATIONS),
                        help="scan plane orientation")
    # Three separate values rather than "x,y,z": argparse treats a lone
    # "-0.825" as a negative number (no option string here looks like one) but
    # treats "-0.825,-91,1556" as an unknown option, and anatomical coordinates
    # are routinely negative.
    p_scan.add_argument("--position", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"),
                        help="plane centre, in scene coordinates")
    p_scan.add_argument("--width", type=float, default=400.0, metavar="MM",
                        help="plane width in scene units")
    p_scan.add_argument("--height", type=float, default=400.0, metavar="MM",
                        help="plane height in scene units")
    p_scan.add_argument("--depth", type=float, default=10.0, metavar="MM",
                        help="slab depth along the view direction")
    p_scan.add_argument("--resolution", type=int, default=128, metavar="N",
                        help="output is NxN pixels")
    p_scan.add_argument("--npy", type=Path, default=None, metavar="FILE",
                        help="also save the raw float32 array (before colour "
                             "mapping) for quantitative use")
    p_scan.add_argument("--json", action="store_true")
    p_scan.set_defaults(func=cmd_scan)

    # -- export ----------------------------------------------------------
    p_export = sub.add_parser(
        "export", help="export scene geometry to GLB (binary glTF 2.0)",
        description="Export the visible meshes with their world transforms "
                    "baked in.  `mesh` is the fuller front end (it reports "
                    "what provenance each format carries); this command stays "
                    "for compatibility.  Video export needs a live Qt GL "
                    "widget and is not available headlessly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_scene_source(p_export, state_required=False)
    p_export.add_argument("--out", type=Path, required=True, metavar="FILE")
    p_export.add_argument("--format", default="glb",
                          choices=("glb", "obj", "ply", "stl"),
                          help="output format")
    p_export.add_argument("--json", action="store_true")
    p_export.set_defaults(func=cmd_export)

    # -- verify-assets ---------------------------------------------------
    p_verify = sub.add_parser(
        "verify-assets", help="check the BodyParts3D dataset is present and complete",
        description="Delegates to tools/fetch_assets.py.  Everything after `--` "
                    "is passed through, so `verify-assets -- cache --dry-run` "
                    "works too.",
    )
    p_verify.add_argument("passthrough", nargs=argparse.REMAINDER,
                          help="arguments for tools/fetch_assets.py "
                               "(default: verify)")
    p_verify.set_defaults(func=cmd_verify_assets)

    # -- list-layers -----------------------------------------------------
    p_list = sub.add_parser(
        "list-layers", help="print the loadable tiers and on-demand layers",
    )
    p_list.set_defaults(func=cmd_list_layers)

    # -- gui -------------------------------------------------------------
    p_gui = sub.add_parser(
        "gui", help="launch the GUI (same as the `faceforge` command)",
        description="Delegates to faceforge.app:main.  This is the only "
                    "subcommand that imports Qt.",
    )
    p_gui.set_defaults(func=cmd_gui)

    # -- still -----------------------------------------------------------
    p_still = sub.add_parser(
        "still", help="true-resolution offscreen publication still",
        description="Render a still AT the requested size through an offscreen "
                    "framebuffer, rather than grabbing a window and scaling it "
                    "up.  --prove-resolution measures the result against a "
                    "bicubic upscale of a smaller render and reports the "
                    "spectral energy the upscale cannot contain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_scene_source(p_still, state_required=False)
    p_still.add_argument("--out", type=Path, required=True, metavar="PNG")
    p_still.add_argument("--size", default="2048x2048", metavar="WxH",
                         help="rasterisation size; the ceiling is the GL "
                              "implementation's, not a constant in this tool")
    p_still.add_argument("--mode", default=None, choices=modes, metavar="MODE",
                         help="force every structure to one render mode")
    p_still.add_argument("--prefer", default="auto",
                         choices=("auto", "hardware", "software"))
    p_still.add_argument("--strict", action="store_true")
    p_still.add_argument("--prove-resolution", action="store_true",
                         help="also render at 1/--prove-factor of the size and "
                              "report how much detail the big render holds that "
                              "an upscale of the small one cannot")
    p_still.add_argument("--prove-factor", type=int, default=4, metavar="N",
                         help="ratio between the reference and the still for "
                              "--prove-resolution")
    p_still.add_argument("--limits-only", action="store_true",
                         help="print GL_MAX_TEXTURE_SIZE / "
                              "GL_MAX_RENDERBUFFER_SIZE / GL_MAX_VIEWPORT_DIMS "
                              "and exit without loading geometry")
    p_still.add_argument("--json", action="store_true")
    p_still.set_defaults(func=cmd_still)

    # -- mesh ------------------------------------------------------------
    from faceforge.export.mesh_export import MESH_FORMATS

    p_mesh = sub.add_parser(
        "mesh", help="export scene geometry to OBJ / PLY / STL / GLB",
        description="Export the visible meshes with their world transforms "
                    "baked in.  Every format also gets a "
                    "'<name>.provenance.json' sidecar, because the four "
                    "formats carry the BodyParts3D attribution and the FMA "
                    "ids to four different depths -- glTF in its own schema, "
                    "OBJ and PLY in comments, STL barely at all.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_scene_source(p_mesh, state_required=False)
    p_mesh.add_argument("--out", type=Path, required=True, metavar="FILE")
    p_mesh.add_argument("--format", default=None, choices=MESH_FORMATS,
                        help="output format (default: inferred from --out)")
    p_mesh.add_argument("--no-sidecar", action="store_true",
                        help="do not write the provenance sidecar (the "
                             "attribution then survives only as far as the "
                             "format itself carries it)")
    p_mesh.add_argument("--json", action="store_true")
    p_mesh.set_defaults(func=cmd_mesh)

    # -- volume ----------------------------------------------------------
    from faceforge.export.hounsfield import HU_MODES
    from faceforge.export.volume import VOLUME_MODES

    p_volume = sub.add_parser(
        "volume", help="DICOM series or NIfTI volume from the virtual scanner",
        description="Stack scanner cross-sections into a volume and write it "
                    "as a DICOM series or a NIfTI-1 file with correct "
                    "geometry.  Pixel values are the scanner's dimensionless "
                    "tissue index unless --hu-mode class is given, which is "
                    "only accepted for a CT scan reduced with 'max'; see "
                    "faceforge.export.hounsfield for why.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_scene_source(p_volume, state_required=False)
    p_volume.add_argument("--out", type=Path, required=True, metavar="PATH",
                          help="output directory (dicom) or .nii/.nii.gz file")
    p_volume.add_argument("--format", default="nifti",
                          choices=("dicom", "nifti"))
    p_volume.add_argument("--mode", default="ct", choices=VOLUME_MODES,
                          help="imaging modality; xray and anatomical are not "
                               "tomographic slices and are refused")
    p_volume.add_argument("--reduction", default="max",
                          choices=SCAN_REDUCTIONS,
                          help="how hits along a ray combine; 'max' is the "
                               "only one whose values invert to a tissue class")
    p_volume.add_argument("--hu-mode", default="index", choices=HU_MODES,
                          help="'index' stores the model's dimensionless 0-1 "
                               "value with RescaleType=US; 'class' stores "
                               "nominal per-tissue Hounsfield units")
    p_volume.add_argument("--orientation", default="axial",
                          choices=sorted(SCAN_ORIENTATIONS))
    p_volume.add_argument("--position", nargs=3, type=float,
                          default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                          help="centre of the whole volume, in scene coordinates")
    p_volume.add_argument("--width", type=float, default=400.0, metavar="MM")
    p_volume.add_argument("--height", type=float, default=400.0, metavar="MM")
    p_volume.add_argument("--resolution", type=int, default=128, metavar="N",
                          help="each slice is NxN")
    p_volume.add_argument("--slices", type=int, default=8, metavar="N")
    p_volume.add_argument("--slice-spacing", type=float, default=5.0,
                          metavar="MM", help="between slice centres")
    p_volume.add_argument("--slice-thickness", type=float, default=None,
                          metavar="MM",
                          help="slab sampled per slice (default: --slice-spacing, "
                               "which makes the slabs contiguous)")
    p_volume.add_argument("--no-sidecar", action="store_true")
    p_volume.add_argument("--json", action="store_true")
    p_volume.set_defaults(func=cmd_volume)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_USAGE

    logging.basicConfig(
        level=logging.DEBUG if args.verbose
        else logging.WARNING if args.quiet else logging.INFO,
        format="%(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        return int(args.func(args))
    except _ConfigDrift as exc:
        print(f"REFUSING TO RENDER: {exc}", file=sys.stderr)
        return EXIT_CONFIG_DRIFT
    except CLIError as exc:
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except ValueError as exc:
        print(f"{args.command}: bad argument: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:                     # noqa: BLE001 - top-level report
        # Reported, never swallowed: the traceback is available at -v, and the
        # exit code is non-zero so no caller can mistake this for a written
        # figure.
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
