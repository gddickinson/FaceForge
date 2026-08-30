"""Tests for the golden-image diff harness.

These use synthetic capture directories built with PIL, so they need no GL
context and no GPU.  A synthetic source is the right instrument here: it lets a
test state exactly which pixels differ and by how much, which a real render
cannot.

The two cases that matter most:

* **the negative control** -- two identical captures must report zero changed
  modes.  Without it the harness is worthless, because a diff that always
  reports change is indistinguishable from one that never does.
* **single-mode isolation** -- perturbing one mode must flag exactly that mode
  and no other.

Both were also verified against real renders; see ``gpu_validation.md``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from tools.capture_golden import ALL_MODES
from tools.compare_golden import (
    DEFAULT_FRAC_TOLERANCE,
    DEFAULT_PIXEL_THRESHOLD,
    IncomparableCaptures,
    amplified_difference,
    compare,
    contact_sheet,
    diff_images,
    load_manifest,
    measure_noise_floor,
)

SIZE = (64, 48)          # (w, h) -- deliberately non-square to catch axis swaps
CLEAR = (31, 31, 38)     # GLRenderer.CLEAR_COLOR as 8-bit


def synth_frame(seed: int, w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    """A deterministic, non-uniform RGBA frame.

    Seeded rather than random-per-call: a test that cannot reproduce its own
    input cannot be debugged when it fails.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = CLEAR
    img[:, :, 3] = 255
    # A blob of "geometry" in the middle, with structure so a shift is visible.
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx - w / 2) ** 2 / (w / 3) ** 2 + (yy - h / 2) ** 2 / (h / 3) ** 2) < 1.0
    base = rng.integers(60, 240, size=(h, w, 3), dtype=np.uint16)
    img[:, :, :3] = np.where(mask[:, :, None], base.astype(np.uint8), img[:, :, :3])
    return img


def write_capture(
    d,
    *,
    modes=ALL_MODES,
    seed_offset: int = 0,
    commit: str = "abc1234",
    size=SIZE,
    renderer: str = "Apple Software Renderer",
    perturb: dict[str, int] | None = None,
    meshes: int = 3,
    schema_version: int = 1,
    camera_preset: str = "oblique",
) -> "object":
    """Build a synthetic capture directory with a manifest that passes checks."""
    d.mkdir(parents=True, exist_ok=True)
    modes = list(modes)
    for i, m in enumerate(modes):
        img = synth_frame(1000 + i + seed_offset, size[0], size[1])
        if perturb and m in perturb:
            delta = perturb[m]
            # Shift a contiguous block, so the reported bbox is checkable.
            y0, y1 = size[1] // 4, size[1] // 2
            x0, x1 = size[0] // 4, size[0] // 2
            region = img[y0:y1, x0:x1, :3].astype(np.int16) + delta
            img[y0:y1, x0:x1, :3] = np.clip(region, 0, 255).astype(np.uint8)
        Image.fromarray(img, mode="RGBA").save(d / f"{m}.png")
    manifest = {
        "schema_version": schema_version,
        "tool": "synthetic",
        "git_commit": commit,
        "git_dirty": False,
        "gl": {"gl_renderer": renderer, "gl_version": "4.1", "is_software": True},
        "viewport": {"width": size[0], "height": size[1]},
        "clear_color_rgb8": list(CLEAR),
        "camera": {
            "preset": camera_preset, "eye": [1.0, 2.0, 3.0], "target": [0.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0], "fov_deg": 45.0, "near": 1.0, "far": 5000.0,
        },
        "modes": modes,
        "meshes": [
            {"fma_id": f"FMA{i}", "label": f"m{i}", "file": f"FMA{i}.stl",
             "triangles": 100 + i}
            for i in range(meshes)
        ],
        "total_triangles": sum(100 + i for i in range(meshes)),
        "files": {m: f"{m}.png" for m in modes},
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return d


# ---------------------------------------------------------------------------
# The negative control
# ---------------------------------------------------------------------------

def test_identical_captures_report_zero_differences(tmp_path):
    """THE negative control: without this passing, nothing else means anything."""
    a = write_capture(tmp_path / "ref")
    b = write_capture(tmp_path / "cur")

    rep = compare(a, b)

    assert rep.changed_modes == [], f"identical captures flagged {rep.changed_modes}"
    assert len(rep.modes) == 16
    for m in rep.modes:
        assert m.max_abs == 0, f"{m.mode}: max_abs {m.max_abs} on identical input"
        assert m.mean_abs == 0.0
        assert m.frac_above == 0.0
        assert m.pixels_above == 0
        assert m.bbox is None
        assert m.changed is False


def test_identical_captures_emit_no_contact_sheets(tmp_path):
    a = write_capture(tmp_path / "ref")
    b = write_capture(tmp_path / "cur")
    sheets = tmp_path / "sheets"

    rep = compare(a, b, sheets_dir=sheets)

    assert rep.sheets == []
    assert not sheets.exists() or not list(sheets.glob("*.png"))


def test_noise_floor_of_identical_captures_is_zero(tmp_path):
    a = write_capture(tmp_path / "a")
    b = write_capture(tmp_path / "b")

    res = measure_noise_floor(a, b)

    assert res["max_abs_over_all_modes"] == 0
    assert res["mean_abs_over_all_modes"] == 0.0
    assert res["max_nonzero_pixels_any_mode"] == 0


# ---------------------------------------------------------------------------
# Single-mode isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["SOLID", "THERMAL", "XRAY", "ETHEREAL", "POINTS"])
def test_perturbing_one_mode_flags_exactly_that_mode(tmp_path, target):
    a = write_capture(tmp_path / "ref")
    b = write_capture(tmp_path / "cur", perturb={target: 40})

    rep = compare(a, b)

    assert rep.changed_modes == [target], (
        f"expected only {target} to change, got {rep.changed_modes}"
    )
    for m in rep.modes:
        if m.mode == target:
            assert m.changed and m.max_abs > 0 and m.pixels_above > 0
            assert m.bbox is not None
        else:
            assert not m.changed and m.max_abs == 0, f"{m.mode} contaminated"


def test_perturbed_mode_bbox_matches_the_perturbed_region(tmp_path):
    a = write_capture(tmp_path / "ref")
    b = write_capture(tmp_path / "cur", perturb={"THERMAL": 40})

    rep = compare(a, b)
    d = next(m for m in rep.modes if m.mode == "THERMAL")

    w, h = SIZE
    ey0, ey1 = h // 4, h // 2 - 1
    ex0, ex1 = w // 4, w // 2 - 1
    x0, y0, x1, y1 = d.bbox
    # The perturbed block bounds the changed pixels.  It can be tighter than
    # the block (a pixel already at 255 does not move), never looser.
    assert ex0 <= x0 <= x1 <= ex1, f"x bbox {x0}..{x1} outside {ex0}..{ex1}"
    assert ey0 <= y0 <= y1 <= ey1, f"y bbox {y0}..{y1} outside {ey0}..{ey1}"


def test_two_perturbed_modes_are_both_reported(tmp_path):
    a = write_capture(tmp_path / "ref")
    b = write_capture(tmp_path / "cur", perturb={"SOLID": 30, "CARTOON": -30})

    rep = compare(a, b)

    assert sorted(rep.changed_modes) == ["CARTOON", "SOLID"]


def test_contact_sheet_written_only_for_changed_modes(tmp_path):
    a = write_capture(tmp_path / "ref")
    b = write_capture(tmp_path / "cur", perturb={"MEDICAL": 50})
    sheets = tmp_path / "sheets"

    rep = compare(a, b, sheets_dir=sheets)

    assert len(rep.sheets) == 1
    written = sorted(p.name for p in sheets.glob("*.png"))
    assert written == ["MEDICAL_diff.png"]


def test_sheets_for_all_writes_every_mode(tmp_path):
    a = write_capture(tmp_path / "ref")
    b = write_capture(tmp_path / "cur")
    sheets = tmp_path / "sheets"

    compare(a, b, sheets_dir=sheets, sheets_for_all=True)

    assert len(list(sheets.glob("*.png"))) == 16


# ---------------------------------------------------------------------------
# Perceptual metric behaviour: it must be a magnitude, not a hash
# ---------------------------------------------------------------------------

def test_sub_threshold_noise_is_not_reported_as_change():
    """A 1-level per-channel shift on every pixel is driver noise, not a regression."""
    ref = synth_frame(7)[:, :, :3]
    cur = np.clip(ref.astype(np.int16) + 1, 0, 255).astype(np.uint8)

    d = diff_images(ref, cur, mode="SOLID")

    assert d.max_abs == 1, "the fixture did not shift by 1"
    assert d.pixels_above == 0, "a 1-level shift exceeded the noise threshold"
    assert d.changed is False


def test_a_hash_would_have_flagged_the_same_input():
    """Justifies the perceptual metric: bytes differ where perception does not."""
    import hashlib

    ref = synth_frame(7)[:, :, :3]
    cur = np.clip(ref.astype(np.int16) + 1, 0, 255).astype(np.uint8)

    assert hashlib.sha256(ref.tobytes()).digest() != hashlib.sha256(cur.tobytes()).digest()
    assert diff_images(ref, cur, mode="SOLID").changed is False


def test_a_few_stray_pixels_are_below_the_fraction_tolerance():
    ref = synth_frame(11)[:, :, :3]
    cur = ref.copy()
    cur[0, 0] = (255, 255, 255)   # 1 pixel of 64*48 = 3072 -> 0.0326%

    d = diff_images(ref, cur, mode="SOLID")

    assert d.max_abs > DEFAULT_PIXEL_THRESHOLD
    assert d.pixels_above == 1
    # 1/3072 = 0.033% which exceeds the 0.01% default; at capture resolution
    # (512x512 = 262144 px) the same single pixel is 0.0004% and would not.
    big = np.zeros((512, 512, 3), np.uint8)
    big[:, :] = CLEAR
    big2 = big.copy()
    big2[0, 0] = (255, 255, 255)
    d2 = diff_images(big, big2, mode="SOLID")
    assert d2.pixels_above == 1
    assert d2.frac_above < DEFAULT_FRAC_TOLERANCE
    assert d2.changed is False, "one stray pixel in 262144 should not be a regression"


def test_magnitudes_are_reported_per_channel():
    ref = np.zeros((10, 10, 3), np.uint8)
    cur = ref.copy()
    cur[:, :, 1] = 77          # green only

    d = diff_images(ref, cur, mode="SOLID")

    assert d.max_abs_per_channel == [0, 77, 0]
    assert d.max_abs == 77
    assert abs(d.mean_abs - 77 / 3) < 1e-6, "mean is over all three channels"
    assert d.frac_above == 1.0 and d.pixels_above == 100


def test_larger_perturbation_yields_larger_metrics():
    ref = np.zeros((32, 32, 3), np.uint8)
    prev = None
    for delta in (5, 20, 80, 200):
        cur = np.full_like(ref, delta)
        d = diff_images(ref, cur, mode="SOLID")
        assert d.max_abs == delta
        if prev is not None:
            assert d.mean_abs > prev, "metric is not monotonic in perturbation size"
        prev = d.mean_abs


def test_shape_mismatch_is_reported_not_crashed():
    d = diff_images(np.zeros((8, 8, 3), np.uint8), np.zeros((9, 8, 3), np.uint8), mode="SOLID")
    assert d.changed is True
    assert "shape mismatch" in d.note


def test_amplified_difference_is_black_when_identical():
    a = synth_frame(3)[:, :, :3]
    amp = amplified_difference(a, a)
    assert amp.shape == (*a.shape[:2], 3)
    assert amp.max() == 0, "identical frames produced a non-black difference image"


def test_amplified_difference_highlights_the_changed_region():
    a = synth_frame(3)[:, :, :3]
    b = a.copy()
    b[10:20, 10:20] = np.clip(b[10:20, 10:20].astype(np.int16) + 60, 0, 255).astype(np.uint8)

    amp = amplified_difference(a, b)

    assert amp[10:20, 10:20].max() > 0
    outside = amp.copy()
    outside[10:20, 10:20] = 0
    assert outside.max() == 0, "difference leaked outside the changed region"


def test_contact_sheet_is_a_valid_three_panel_image(tmp_path):
    a = synth_frame(5)[:, :, :3]
    b = a.copy()
    b[5:15, 5:15] = 200
    d = diff_images(a, b, mode="SOLID")
    out = contact_sheet(a, b, d, tmp_path / "s.png")

    assert out.is_file()
    with Image.open(out) as im:
        w, h = im.size
    assert w >= SIZE[0] * 3, f"sheet width {w} cannot hold three panels"
    assert h > SIZE[1], "sheet has no annotation band"


# ---------------------------------------------------------------------------
# Comparability refusal
# ---------------------------------------------------------------------------

def test_missing_manifest_is_refused(tmp_path):
    a = write_capture(tmp_path / "ref")
    b = tmp_path / "cur"
    b.mkdir()
    for m in ALL_MODES:
        Image.fromarray(synth_frame(1)).save(b / f"{m}.png")

    with pytest.raises(IncomparableCaptures, match="no manifest.json"):
        compare(a, b)


def test_corrupt_manifest_is_refused(tmp_path):
    d = write_capture(tmp_path / "ref")
    (d / "manifest.json").write_text("{not json")
    with pytest.raises(IncomparableCaptures, match="not valid JSON"):
        load_manifest(d)


def test_nonexistent_directory_is_refused(tmp_path):
    with pytest.raises(IncomparableCaptures, match="not a directory"):
        load_manifest(tmp_path / "nope")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"size": (32, 32)}, "viewport differs"),
        ({"modes": ALL_MODES[:8]}, "mode sets differ"),
        ({"meshes": 5}, "mesh list differs"),
        ({"camera_preset": "anterior"}, "camera differs"),
        ({"schema_version": 2}, "schema_version differs"),
        ({"renderer": "Apple M1 Max"}, "GL_RENDERER differs"),
    ],
)
def test_incomparable_captures_are_refused(tmp_path, kwargs, match):
    a = write_capture(tmp_path / "ref")
    b = write_capture(tmp_path / "cur", **kwargs)

    with pytest.raises(IncomparableCaptures, match=match):
        compare(a, b)


def test_renderer_mismatch_can_be_overridden_explicitly(tmp_path):
    a = write_capture(tmp_path / "ref", renderer="Apple Software Renderer")
    b = write_capture(tmp_path / "cur", renderer="Apple M1 Max")

    rep = compare(a, b, allow_renderer_mismatch=True)

    assert rep.changed_modes == []
    assert any("OVERRIDDEN" in n for n in rep.comparability_notes)


def test_differing_commit_is_a_note_not_a_refusal(tmp_path):
    """Comparing across commits is the point; it must not be blocked."""
    a = write_capture(tmp_path / "ref", commit="aaaaaaa")
    b = write_capture(tmp_path / "cur", commit="bbbbbbb", perturb={"SOLID": 40})

    rep = compare(a, b)

    assert rep.changed_modes == ["SOLID"]
    assert rep.ref_commit == "aaaaaaa" and rep.cur_commit == "bbbbbbb"
    assert any("commit differs" in n for n in rep.comparability_notes)


def test_missing_png_is_reported_as_changed_not_skipped(tmp_path):
    a = write_capture(tmp_path / "ref")
    b = write_capture(tmp_path / "cur")
    # Rename rather than delete: the test must not depend on unlink semantics.
    (b / "SOLID.png").replace(b / "SOLID.png.moved")

    rep = compare(a, b)

    d = next(m for m in rep.modes if m.mode == "SOLID")
    assert d.changed is True
    assert d.present_in_both is False
    assert "missing PNG" in d.note
    assert "SOLID" in rep.changed_modes


def test_report_is_json_serialisable(tmp_path):
    a = write_capture(tmp_path / "ref")
    b = write_capture(tmp_path / "cur", perturb={"XRAY": 25})

    rep = compare(a, b)
    text = json.dumps(rep.to_dict())

    round_tripped = json.loads(text)
    assert round_tripped["changed_modes"] == ["XRAY"]
    assert len(round_tripped["modes"]) == 16
    assert round_tripped["pixel_threshold"] == DEFAULT_PIXEL_THRESHOLD


def test_cli_exit_codes(tmp_path):
    from tools.compare_golden import main

    a = write_capture(tmp_path / "ref")
    same = write_capture(tmp_path / "same")
    diff = write_capture(tmp_path / "diff", perturb={"SEPIA": 40})

    assert main([str(a), str(same)]) == 0, "identical captures should exit 0"
    assert main([str(a), str(diff)]) == 1, "changed captures should exit 1"
    assert main([str(a), str(tmp_path / "missing")]) == 3, "refusal should exit 3"
