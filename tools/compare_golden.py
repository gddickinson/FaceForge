"""Compare two golden-image capture directories and report what changed.

Given a reference capture and a current one (both produced by
``tools/capture_golden.py``), this reports per-mode differences and emits an
annotated contact sheet for every mode that changed.

Why not a hash
--------------
A checksum answers "are these byte-identical", which is not the question.  Two
runs of the same code through the same rasteriser are *not* guaranteed
byte-identical -- rasterisers dither flat regions, and different drivers differ
in the low bits of every shaded pixel.  A hash comparison would therefore
report "everything changed" on every run and be ignored within a day.  So this
reports magnitudes instead:

    max_abs      largest per-channel absolute difference, 0-255
    mean_abs     mean per-channel absolute difference over all pixels
    frac_above   fraction of pixels whose max channel delta exceeds a threshold
    bbox         bounding box of the pixels above that threshold

and a mode counts as changed when ``frac_above`` exceeds a tolerance.

The threshold
-------------
``DEFAULT_PIXEL_THRESHOLD`` and ``DEFAULT_FRAC_TOLERANCE`` were measured, not
guessed.  Two consecutive 16-mode captures of an unmodified tree at 256x256
through the Apple Software Renderer (GL 4.1 APPLE-23.1.1) were diffed; the
measured numbers are recorded in ``gpu_validation.md`` and in
``NOISE_FLOOR_MEASURED`` below.  Both defaults sit above the measured floor.

That floor is specific to a same-machine, same-driver pair.  It is NOT valid
across two different rasterisers, which is why :func:`check_comparability`
refuses a cross-renderer diff unless explicitly overridden.  Re-measure with
``--measure-noise-floor A B`` on any new machine.

Usage
-----
    python -m tools.compare_golden REF CUR
    python -m tools.compare_golden REF CUR --sheets out/sheets --json report.json
    python -m tools.compare_golden --measure-noise-floor A B
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("compare_golden")

# ----------------------------------------------------------------------------
# Empirically measured noise floor.  Populated from a real measurement; see
# the module docstring.  Keep this literal in sync with gpu_validation.md.
# ----------------------------------------------------------------------------
NOISE_FLOOR_MEASURED = {
    "measured_on": "2026-08-29",
    "machine": "Apple M1 Max (sandboxed; CPU rasteriser)",
    "gl_renderer": "Apple Software Renderer",
    "gl_version": "4.1 APPLE-23.1.1",
    "size": "256x256",
    "modes": 16,
    "meshes": 16,
    "triangles": 230490,
    "method": (
        "two consecutive `capture_golden --out A` / `--out B` runs on an "
        "unmodified tree, diffed with compare_golden --measure-noise-floor"
    ),
    # Filled in from the actual run; see gpu_validation.md for the raw table.
    "max_abs_over_all_modes": 0,
    "mean_abs_over_all_modes": 0.0,
    "frac_above_threshold_max": 0.0,
    "conclusion": (
        "bit-identical: repeat captures on the same machine and driver differ "
        "in zero pixels, so any non-zero difference is signal"
    ),
}

# A per-channel delta at or below this is treated as rasteriser noise.  The
# measured same-driver floor is 0 (bit-identical), so 2 is pure headroom: it
# tolerates dithering an OS or driver update might introduce without masking a
# real shading change, whose deltas run to tens of levels.
DEFAULT_PIXEL_THRESHOLD = 2

# A mode is reported changed when more than this fraction of its pixels exceed
# the pixel threshold.  0.0001 = 1 pixel in 10,000: at 256x256 that is ~6
# pixels, at 512x512 ~26.  Below a handful of pixels a "difference" is not
# something a human could see or a shader could plausibly cause.
DEFAULT_FRAC_TOLERANCE = 0.0001


class IncomparableCaptures(RuntimeError):
    """The two captures do not describe the same scene; a diff would be noise."""


@dataclass
class ModeDiff:
    """Per-mode difference measurements.  All numbers are measured, not derived."""

    mode: str
    present_in_both: bool
    width: int
    height: int
    max_abs: int
    mean_abs: float
    max_abs_per_channel: list[int]
    frac_above: float
    pixels_above: int
    total_pixels: int
    bbox: list[int] | None      # [x0, y0, x1, y1] inclusive, or None
    changed: bool
    note: str = ""


@dataclass
class CompareReport:
    ref_dir: str
    cur_dir: str
    pixel_threshold: int
    frac_tolerance: float
    comparable: bool
    comparability_notes: list[str]
    modes: list[ModeDiff]
    changed_modes: list[str]
    ref_commit: str = ""
    cur_commit: str = ""
    sheets: list[str] = None  # type: ignore[assignment]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sheets"] = self.sheets or []
        return d


# ----------------------------------------------------------------------------
# Loading and comparability
# ----------------------------------------------------------------------------

def load_manifest(d: Path) -> dict:
    """Read a capture's manifest, or explain why the directory is not a capture."""
    d = Path(d)
    mf = d / "manifest.json"
    if not d.is_dir():
        raise IncomparableCaptures(f"{d} is not a directory")
    if not mf.is_file():
        raise IncomparableCaptures(
            f"{d} has no manifest.json, so it is not a completed capture.\n"
            "  capture_golden.py writes the manifest last, and only after every mode "
            "has passed its blank-frame check.  A directory without one is the "
            "residue of a failed run and must not be diffed."
        )
    try:
        return json.loads(mf.read_text())
    except json.JSONDecodeError as exc:
        raise IncomparableCaptures(f"{mf} is not valid JSON: {exc}") from exc


def check_comparability(
    ref: dict, cur: dict, *, allow_renderer_mismatch: bool = False
) -> list[str]:
    """Verify two manifests describe the same scene.  Returns advisory notes.

    Raises :class:`IncomparableCaptures` when the captures cannot meaningfully
    be diffed.  A differing ``git_commit`` is a *note*, not a failure -- that
    is the case a regression check exists to handle.
    """
    from tools.capture_golden import manifest_comparability_key

    notes: list[str] = []
    problems: list[str] = []

    rk = manifest_comparability_key(ref)
    ck = manifest_comparability_key(cur)

    if ref.get("schema_version") != cur.get("schema_version"):
        problems.append(
            f"manifest schema_version differs: {ref.get('schema_version')} vs "
            f"{cur.get('schema_version')}"
        )
    if rk["viewport"] != ck["viewport"]:
        problems.append(f"viewport differs: {rk['viewport']} vs {ck['viewport']}")
    if rk["camera"] != ck["camera"]:
        diffs = [
            f"{k}: {rk['camera'].get(k)} vs {ck['camera'].get(k)}"
            for k in set(rk["camera"]) | set(ck["camera"])
            if rk["camera"].get(k) != ck["camera"].get(k)
        ]
        problems.append("camera differs: " + "; ".join(sorted(diffs)))
    if rk["meshes"] != ck["meshes"]:
        rn = [m["fma_id"] for m in rk["meshes"]]
        cn = [m["fma_id"] for m in ck["meshes"]]
        if rn != cn:
            problems.append(
                f"mesh list differs: {len(rn)} vs {len(cn)} meshes; "
                f"only in ref={sorted(set(rn) - set(cn))}, only in cur={sorted(set(cn) - set(rn))}"
            )
        else:
            problems.append("mesh triangle counts differ — the STL assets changed")
    if rk["clear_color_rgb8"] != ck["clear_color_rgb8"]:
        problems.append(
            f"clear colour differs: {rk['clear_color_rgb8']} vs {ck['clear_color_rgb8']}"
        )
    if rk["modes"] != ck["modes"]:
        only_ref = sorted(set(rk["modes"] or []) - set(ck["modes"] or []))
        only_cur = sorted(set(ck["modes"] or []) - set(rk["modes"] or []))
        problems.append(f"mode sets differ: only in ref={only_ref}, only in cur={only_cur}")

    if rk["gl_renderer"] != ck["gl_renderer"]:
        msg = (
            f"GL_RENDERER differs: {rk['gl_renderer']!r} vs {ck['gl_renderer']!r}. "
            "The pixel noise floor between two different rasterisers is unmeasured, "
            "so a cross-renderer diff reports change everywhere and means nothing."
        )
        if allow_renderer_mismatch:
            notes.append("OVERRIDDEN: " + msg)
        else:
            problems.append(msg + "  Pass --allow-renderer-mismatch to diff anyway.")

    if problems:
        raise IncomparableCaptures(
            "refusing to compare incomparable captures:\n  - " + "\n  - ".join(problems)
        )

    if ref.get("git_commit") != cur.get("git_commit"):
        notes.append(
            f"commit differs: {ref.get('git_commit')} -> {cur.get('git_commit')} "
            "(expected for a regression check)"
        )
    else:
        notes.append(f"same commit: {ref.get('git_commit')}")
    if ref.get("git_dirty") or cur.get("git_dirty"):
        notes.append(
            f"working tree dirty (ref={ref.get('git_dirty')}, cur={cur.get('git_dirty')}) "
            "— 'same commit' does not imply same code"
        )
    if (ref.get("gl") or {}).get("is_software"):
        notes.append("captures are from a CPU rasteriser: correctness only, not performance")
    return notes


# ----------------------------------------------------------------------------
# Pixel comparison
# ----------------------------------------------------------------------------

def load_rgb(path: Path) -> np.ndarray:
    """Load a PNG as HxWx3 uint8, dropping alpha.

    Alpha is dropped deliberately: the FBO's alpha channel records the shader's
    computed coverage, which for the blending modes is a fractional value that
    was already composited into RGB against the clear colour.  Diffing it as
    well would double-count exactly those five modes.
    """
    from PIL import Image

    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def diff_images(
    ref: np.ndarray,
    cur: np.ndarray,
    *,
    pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD,
    frac_tolerance: float = DEFAULT_FRAC_TOLERANCE,
    mode: str = "?",
) -> ModeDiff:
    """Measure the difference between two RGB frames."""
    if ref.shape != cur.shape:
        return ModeDiff(
            mode=mode, present_in_both=True,
            width=cur.shape[1], height=cur.shape[0],
            max_abs=255, mean_abs=float("nan"), max_abs_per_channel=[255, 255, 255],
            frac_above=1.0, pixels_above=int(cur.shape[0] * cur.shape[1]),
            total_pixels=int(cur.shape[0] * cur.shape[1]), bbox=None, changed=True,
            note=f"shape mismatch: ref {ref.shape} vs cur {cur.shape}",
        )

    delta = np.abs(ref.astype(np.int16) - cur.astype(np.int16))
    per_pixel = delta.max(axis=2)
    above = per_pixel > pixel_threshold
    n_above = int(above.sum())
    total = int(per_pixel.size)
    frac = n_above / total

    bbox = None
    if n_above:
        ys, xs = np.nonzero(above)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    return ModeDiff(
        mode=mode, present_in_both=True,
        width=int(ref.shape[1]), height=int(ref.shape[0]),
        max_abs=int(delta.max()),
        mean_abs=float(delta.mean()),
        max_abs_per_channel=[int(delta[:, :, c].max()) for c in range(3)],
        frac_above=float(frac), pixels_above=n_above, total_pixels=total,
        bbox=bbox, changed=bool(frac > frac_tolerance),
    )


def amplified_difference(ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """A visible rendering of where two frames differ.

    Amplification is per-image, normalised to that image's own maximum delta,
    with the scale reported in the contact-sheet caption.  A fixed gain would
    saturate to solid white on a large change and show nothing on a subtle one,
    which is the opposite of useful.
    """
    delta = np.abs(ref.astype(np.int16) - cur.astype(np.int16)).max(axis=2)
    peak = int(delta.max())
    if peak == 0:
        return np.zeros((*delta.shape, 3), np.uint8)
    norm = (delta.astype(np.float32) / peak).clip(0, 1)
    out = np.zeros((*delta.shape, 3), np.uint8)
    # Black -> red -> yellow -> white ramp: monotonic in luminance, so
    # magnitude ordering survives being printed in greyscale.
    out[:, :, 0] = (norm.clip(0, 0.33) / 0.33 * 255).astype(np.uint8)
    out[:, :, 1] = ((norm - 0.33).clip(0, 0.34) / 0.34 * 255).astype(np.uint8)
    out[:, :, 2] = ((norm - 0.67).clip(0, 0.33) / 0.33 * 255).astype(np.uint8)
    return out


def contact_sheet(
    ref: np.ndarray, cur: np.ndarray, d: ModeDiff, out_path: Path, gain_note: str = ""
) -> Path:
    """Write a reference | current | amplified-difference triptych."""
    from PIL import Image, ImageDraw

    amp = amplified_difference(ref, cur)
    h, w = cur.shape[:2]
    # band holds two header lines (y=3, y=17) plus the per-panel titles (y=34).
    # 48 px keeps them from colliding; 34 put the titles on top of header line 2.
    pad, band = 6, 48
    sheet = Image.new("RGB", (w * 3 + pad * 4, h + band + pad * 2), (18, 18, 20))
    for i, (img, title) in enumerate(
        ((ref, "reference"), (cur, "current"), (amp, "|difference| amplified"))
    ):
        x = pad + i * (w + pad)
        sheet.paste(Image.fromarray(img), (x, band + pad))
        ImageDraw.Draw(sheet).text((x + 2, band - 14), title, fill=(210, 210, 215))
    dr = ImageDraw.Draw(sheet)
    dr.text((pad, 3), f"{d.mode}   {'CHANGED' if d.changed else 'within tolerance'}",
            fill=(255, 120, 120) if d.changed else (140, 220, 140))
    dr.text((pad + 190, 3),
            f"max_abs={d.max_abs}  mean_abs={d.mean_abs:.4f}  "
            f"frac_above={d.frac_above * 100:.4f}%  ({d.pixels_above}/{d.total_pixels} px)",
            fill=(200, 200, 205))
    if d.bbox:
        dr.text((pad, 17), f"changed bbox x0,y0,x1,y1 = {d.bbox}   {gain_note}",
                fill=(200, 200, 205))
        # Outline the changed region on the current frame.
        x0, y0, x1, y1 = d.bbox
        ox = pad + (w + pad)
        dr.rectangle([ox + x0, band + pad + y0, ox + x1, band + pad + y1],
                     outline=(255, 80, 80))
    else:
        dr.text((pad, 17), f"no pixels above threshold   {gain_note}", fill=(200, 200, 205))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG")
    return out_path


def compare(
    ref_dir: Path,
    cur_dir: Path,
    *,
    pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD,
    frac_tolerance: float = DEFAULT_FRAC_TOLERANCE,
    sheets_dir: Path | None = None,
    sheets_for_all: bool = False,
    allow_renderer_mismatch: bool = False,
) -> CompareReport:
    """Compare two capture directories.  Raises IncomparableCaptures if unsafe."""
    ref_dir, cur_dir = Path(ref_dir), Path(cur_dir)
    ref_man = load_manifest(ref_dir)
    cur_man = load_manifest(cur_dir)
    notes = check_comparability(
        ref_man, cur_man, allow_renderer_mismatch=allow_renderer_mismatch
    )

    modes: list[ModeDiff] = []
    sheets: list[str] = []
    for mode in cur_man["modes"]:
        rp = ref_dir / f"{mode}.png"
        cp = cur_dir / f"{mode}.png"
        if not rp.is_file() or not cp.is_file():
            missing = [str(p) for p in (rp, cp) if not p.is_file()]
            modes.append(ModeDiff(
                mode=mode, present_in_both=False, width=0, height=0, max_abs=255,
                mean_abs=float("nan"), max_abs_per_channel=[255, 255, 255],
                frac_above=1.0, pixels_above=0, total_pixels=0, bbox=None,
                changed=True, note=f"missing PNG: {missing}",
            ))
            continue
        a, b = load_rgb(rp), load_rgb(cp)
        d = diff_images(a, b, pixel_threshold=pixel_threshold,
                        frac_tolerance=frac_tolerance, mode=mode)
        modes.append(d)
        if sheets_dir is not None and (d.changed or sheets_for_all):
            gain = f"amplification: delta/{d.max_abs} (per-image)" if d.max_abs else ""
            sheets.append(str(contact_sheet(a, b, d, Path(sheets_dir) / f"{mode}_diff.png", gain)))

    return CompareReport(
        ref_dir=str(ref_dir), cur_dir=str(cur_dir),
        pixel_threshold=pixel_threshold, frac_tolerance=frac_tolerance,
        comparable=True, comparability_notes=notes, modes=modes,
        changed_modes=[m.mode for m in modes if m.changed],
        ref_commit=str(ref_man.get("git_commit", "")),
        cur_commit=str(cur_man.get("git_commit", "")),
        sheets=sheets,
    )


def measure_noise_floor(a_dir: Path, b_dir: Path) -> dict:
    """Diff two captures that should be identical, to establish the floor.

    Run this on two back-to-back captures of an unmodified tree.  Whatever it
    reports is rasteriser noise; the change threshold must sit above it.
    """
    a_man, b_man = load_manifest(a_dir), load_manifest(b_dir)
    check_comparability(a_man, b_man)
    rows = []
    for mode in b_man["modes"]:
        ra, rb = load_rgb(Path(a_dir) / f"{mode}.png"), load_rgb(Path(b_dir) / f"{mode}.png")
        delta = np.abs(ra.astype(np.int16) - rb.astype(np.int16))
        per_pixel = delta.max(axis=2)
        rows.append({
            "mode": mode,
            "max_abs": int(delta.max()),
            "mean_abs": float(delta.mean()),
            "pixels_nonzero": int((per_pixel > 0).sum()),
            "pixels_above_1": int((per_pixel > 1).sum()),
            "pixels_above_2": int((per_pixel > 2).sum()),
            "total_pixels": int(per_pixel.size),
        })
    return {
        "a": str(a_dir), "b": str(b_dir),
        "gl_renderer": (b_man.get("gl") or {}).get("gl_renderer"),
        "gl_version": (b_man.get("gl") or {}).get("gl_version"),
        "size": f"{b_man['viewport']['width']}x{b_man['viewport']['height']}",
        "modes": len(rows),
        "triangles": b_man.get("total_triangles"),
        "per_mode": rows,
        "max_abs_over_all_modes": max(r["max_abs"] for r in rows),
        "mean_abs_over_all_modes": float(np.mean([r["mean_abs"] for r in rows])),
        "max_nonzero_pixels_any_mode": max(r["pixels_nonzero"] for r in rows),
        "max_frac_nonzero_any_mode": max(
            r["pixels_nonzero"] / r["total_pixels"] for r in rows
        ),
    }


def format_report(rep: CompareReport) -> str:
    lines = [
        f"reference : {rep.ref_dir}  (commit {rep.ref_commit})",
        f"current   : {rep.cur_dir}  (commit {rep.cur_commit})",
        f"threshold : per-channel delta > {rep.pixel_threshold}, "
        f"changed if > {rep.frac_tolerance * 100:.4f}% of pixels",
    ]
    for n in rep.comparability_notes:
        lines.append(f"note      : {n}")
    lines.append("")
    lines.append(f"{'mode':<13} {'max':>4} {'mean':>9} {'frac>thr':>10} {'px':>8}  bbox")
    for m in rep.modes:
        flag = "CHANGED" if m.changed else ""
        bbox = ",".join(str(v) for v in m.bbox) if m.bbox else "-"
        lines.append(
            f"{m.mode:<13} {m.max_abs:>4} {m.mean_abs:>9.5f} "
            f"{m.frac_above * 100:>9.4f}% {m.pixels_above:>8}  {bbox:<24} {flag}"
            + (f"  {m.note}" if m.note else "")
        )
    lines.append("")
    if rep.changed_modes:
        lines.append(f"CHANGED MODES ({len(rep.changed_modes)}): {', '.join(rep.changed_modes)}")
    else:
        lines.append("NO MODES CHANGED — all differences at or below the measured noise floor.")
    for s in rep.sheets or []:
        lines.append(f"  contact sheet: {s}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="compare_golden",
        description="Diff two capture_golden output directories per render mode.",
    )
    p.add_argument("ref", type=Path, nargs="?", help="reference capture directory")
    p.add_argument("cur", type=Path, nargs="?", help="current capture directory")
    p.add_argument("--pixel-threshold", type=int, default=DEFAULT_PIXEL_THRESHOLD,
                   help=f"per-channel delta treated as noise (default {DEFAULT_PIXEL_THRESHOLD})")
    p.add_argument("--frac-tolerance", type=float, default=DEFAULT_FRAC_TOLERANCE,
                   help=f"changed if this fraction exceeded (default {DEFAULT_FRAC_TOLERANCE})")
    p.add_argument("--sheets", type=Path, default=None, help="write contact sheets here")
    p.add_argument("--sheets-for-all", action="store_true",
                   help="contact sheets for unchanged modes too")
    p.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    p.add_argument("--allow-renderer-mismatch", action="store_true",
                   help="diff captures from different GL renderers (noise floor unmeasured)")
    p.add_argument("--measure-noise-floor", nargs=2, metavar=("A", "B"), type=Path,
                   default=None, help="diff two supposedly-identical captures and report the floor")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if args.measure_noise_floor:
        try:
            res = measure_noise_floor(*args.measure_noise_floor)
        except IncomparableCaptures as exc:
            print(f"CANNOT MEASURE: {exc}")
            return 3
        print(f"noise floor: {res['gl_renderer']} / {res['gl_version']} "
              f"@ {res['size']}, {res['modes']} modes, {res['triangles']} triangles")
        print(f"{'mode':<13} {'max':>4} {'mean':>9} {'px!=0':>8} {'px>1':>8} {'px>2':>8}")
        for r in res["per_mode"]:
            print(f"{r['mode']:<13} {r['max_abs']:>4} {r['mean_abs']:>9.5f} "
                  f"{r['pixels_nonzero']:>8} {r['pixels_above_1']:>8} {r['pixels_above_2']:>8}")
        print(f"\nmax_abs over all modes      : {res['max_abs_over_all_modes']}")
        print(f"mean_abs over all modes     : {res['mean_abs_over_all_modes']:.6f}")
        print(f"worst differing-pixel count : {res['max_nonzero_pixels_any_mode']} "
              f"({res['max_frac_nonzero_any_mode'] * 100:.6f}% of a frame)")
        if args.json:
            args.json.write_text(json.dumps(res, indent=2) + "\n")
            print(f"written: {args.json}")
        return 0

    if args.ref is None or args.cur is None:
        p.error("REF and CUR are required (or use --measure-noise-floor A B)")

    try:
        rep = compare(
            args.ref, args.cur,
            pixel_threshold=args.pixel_threshold, frac_tolerance=args.frac_tolerance,
            sheets_dir=args.sheets, sheets_for_all=args.sheets_for_all,
            allow_renderer_mismatch=args.allow_renderer_mismatch,
        )
    except IncomparableCaptures as exc:
        print(f"INCOMPARABLE: {exc}")
        return 3
    print(format_report(rep))
    if args.json:
        args.json.write_text(json.dumps(rep.to_dict(), indent=2) + "\n")
        print(f"written: {args.json}")
    return 1 if rep.changed_modes else 0


if __name__ == "__main__":
    raise SystemExit(main())
