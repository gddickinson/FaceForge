# tests/ -- Test Suite

Unit and integration tests for FaceForge, using pytest.

## Running Tests

```bash
export QT_QPA_PLATFORM=offscreen

# Fast tier -- what you run while working. No dataset required.
pytest -m "not slow"

# Everything, including the asset-heavy and GUI tests.
pytest

# A specific module / test
pytest tests/core/test_scene_graph.py
pytest tests/core/test_scene_graph.py::test_node_hierarchy
```

## Two tiers

Measured on 2026-08-29 with `--durations=0`:

| tier | command | result | wall time |
|------|---------|--------|-----------|
| fast | `pytest -m "not slow"` | 1032 passed, 8 skipped, 59 deselected | **16.90 s** |
| full | `pytest`                | 1091 passed, 8 skipped                | 73.75 s |

The fast tier is 95% of the suite in 23% of the time. (When the tiers were
first split the suite was 957 tests / 105.87 s and the fast tier ran in
15.98 s; both moved as other work landed. Full-tier wall time varies between
about 63 s and 106 s depending on how warm the `.npz` mesh cache and the
skin-binding cache are, so treat it as a range, not a constant.)

The eight skips are in `test_shader_gpu_agreement.py` (`points.frag` needs
`gl_PointCoord`, which is undefined when rasterising a triangle) and are
unrelated to assets.

`slow` means exactly one of:

* **the test loads the BodyParts3D asset set.** `test_headless_diagnostic.py`
  alone was 52.08 s of the 104.27 s of measured test time; also
  `test_real_hand_curl.py` (15.65 s), `test_head_rotation_diagnostic.py`
  (4.25 s), `test_fascia_integration.py` (3.30 s), the real-mesh cases in
  `test_stl_parser_equivalence.py` (7.78 s of it one mesh, FMA7163), and the
  real-repo case in `test_fetch_assets.py`.
* **the test runs the whole Qt application** — `tests/ui/test_gui_smoke.py`
  (6.38 s, 789 interactions).
* **the test is a measurement rather than a regression guard** —
  `test_meshinstance_addresses_are_heavily_recycled` (3.15 s of `gc.collect()`
  establishing that CPython recycles object addresses).

The rule is enforced, not just followed. `tests/conftest.py` marks any test
whose fixture closure contains an asset-reading fixture (`real_scene`,
`headless_scene`) as `slow` at collection time, so a new asset-heavy module
cannot enter the fast tier by being written after the markers were placed —
which had already happened once, breaking the dataset-free run with eight
setup errors.

Because everything asset-heavy is in the slow tier, the fast tier runs with no
dataset at all. Verified against an empty asset path: **1032 passed, 8 skipped,
54 deselected, 0 errors in 18.3 s**. That is the condition CI runs in.

Nothing was deleted to make the fast tier fast. The full tier still runs
everything; use it before pushing anything that touches loading, skinning or
the renderer.

## Assets

The asset-heavy modules need the BodyParts3D STL set, reached through
`assets/stl` — a **committed symlink** to a path outside the repo. When that
link is dangling the modules' `skipif` guards turn 33 tests into silent skips
and the suite still reports green; that is what happened before the 2026-08
review. Check it explicitly:

```bash
python -m tools.fetch_assets verify        # diagnose: complete / partial /
                                           # empty / missing / dangling symlink
python -m tools.fetch_assets cache         # build the .npz welded-mesh cache
python -m tools.fetch_assets manifest      # the 932 mesh ids the configs need
```

`verify` exits 0 only when every mesh the configs name is present and passes a
binary-STL header check. A complete install of the published dataset reports
930/932: `expression_muscles.json` names `FMA49041`/`FMA49042` (levator
palpebrae superioris L/R), which are not in the distribution — use
`--allow-missing 2` to gate on that.

## Configuration

Test configuration is in `pyproject.toml`:
- `testpaths = ["tests"]`
- `pythonpath = ["src"]`

Dev dependencies: `pytest >= 7.4`, `pytest-qt >= 4.2` (Qt widget tests),
`ruff >= 0.16.5, < 0.17` (lint). Qt needs `QT_QPA_PLATFORM=offscreen`.

## CI

`.github/workflows/ci.yml` runs four gating jobs and one experiment:

| job | what it gates |
|-----|---------------|
| `lint` | ruff's tuned select: a hard gate on the families that are clean, plus a per-rule ratchet on the 96 pre-existing findings |
| `fast-tests` | `pytest -m "not slow"`, with no dataset |
| `shaders` | glslang 16.5.0 compiles all 35 shader cases; 81 static and 131 CPU-semantics tests |
| `golden-images` | **experiment, non-blocking** — whether a macOS runner can acquire a headless GL context |

Every test job writes a JUnit report and `.github/scripts/check_junit.py`
asserts how many tests actually ran, so a job cannot pass by skipping. The
`shaders` job additionally asserts `glslangValidator` is on PATH before running
anything, because that module skips its whole self when the compiler is absent.

See `ui_responsiveness.md` at the repo root for the per-interaction timing
budget and how it was measured.

## Test Structure

```
tests/
  core/                  # Core infrastructure tests
    test_scene_graph.py  # SceneNode hierarchy, matrix propagation, add/remove
    test_state.py        # FaceState, BodyState, StateManager
    test_events.py       # EventBus subscribe/publish/unsubscribe
    test_math_utils.py   # Vec3, Quat, Mat4 operations, quaternion math
  animation/             # Animation subsystem tests
    test_interpolation.py  # StateInterpolator lerp behavior
    test_auto_blink.py    # AutoBlink timing and blink cycle
  body/                  # Body system tests
    test_skinning_diagnostics.py  # SkinningDiagnostic analysis, cross-region binding prevention
    test_skinning_cache.py  # Binding-solve disk cache: bitwise round-trip, key coverage
  ui/                    # UI tests (headless; runs the real app.main())
    gui_harness.py       # Reusable driver: dialog stubs, startup drain, timed control sweep
    test_gui_smoke.py    # 789 interactions, 6 tabs, 0 exceptions, 1.5 s responsiveness budget
  loaders/               # Loader tests
    test_stl_parser.py   # Binary STL parsing, indexed geometry dedup
  tools/                 # Tool tests
    test_headless_diagnostic.py  # Headless loader and diagnostic integration
  anatomy/               # (placeholder, no tests yet)
  integration/           # (placeholder, no tests yet)
  fixtures/              # Shared test fixtures directory
```

## Test Coverage

### `core/test_scene_graph.py`
Tests SceneNode parent-child relationships, matrix composition, world matrix propagation, visibility, mesh attachment, and Scene traversal.

### `core/test_state.py`
Tests FaceState AU getter/setter, BodyState DOF management, TargetAU/TargetHead, StateManager aggregation.

### `core/test_events.py`
Tests EventBus subscribe, publish with data, unsubscribe, and clear.

### `core/test_math_utils.py`
Tests vector operations, quaternion identity/composition/slerp, matrix inverse, look-at, perspective, euler-to-quaternion conversions.

### `animation/test_interpolation.py`
Tests StateInterpolator convergence behavior for AUs, head rotation, and body state.

### `animation/test_auto_blink.py`
Tests AutoBlink cycle timing, blink amount ramp up/down, random interval generation.

### `body/test_skinning_diagnostics.py`
Tests with realistic joint coordinates matching the actual BodyParts3D model. Verifies:
- Hand/foot chains cannot grab distant vertices (proportional Z margin)
- Arm chains do not bind to leg-region vertices (spatial limit guard)
- Cross-binding anomalies are detected by the diagnostic tool
- Arm raise does not displace leg/torso skin

### `loaders/test_stl_parser.py`
Tests binary STL parsing with synthetic triangle data, indexed geometry deduplication, and error handling for malformed files.

### `tools/test_headless_diagnostic.py`
Integration test for the headless loader: loads scene without Qt/GL, runs diagnostic analysis, verifies binding reports.

## Writing New Tests

- Place tests in the subdirectory matching the source package (e.g., `tests/body/` for `src/faceforge/body/`)
- Use NumPy for constructing test geometry (positions, normals, indices)
- Tests should not require OpenGL or Qt unless specifically testing those integrations
- Use `pytest-qt` for tests that need a QApplication context
