# Headless FaceForge: the Session API and `faceforge-cli`

Until now every render went through the GUI, so a figure could not be
regenerated from a file — not even by its author, who would have to remember a
camera position, a render mode, which structures were loaded and which were
visible. `faceforge.core.scene_state` records all of that in one diffable JSON
file. This page is the other half: a process that renders such a file with no
window, no Qt and no display.

## Two commands, on purpose

| command | what it does | imports Qt |
|---|---|---|
| `faceforge` | launches the GUI (unchanged) | yes |
| `faceforge-cli` | render / batch / scan / export / verify-assets | no |

`faceforge` has always opened a window when run with no arguments. Turning it
into a subcommand dispatcher would either break that or keep the GUI as an
implicit default subcommand — in which case a mistyped subcommand opens a
window instead of printing an error. `faceforge.app` also imports PySide6 at
module scope, so a CLI routed through it could not run on a machine without
Qt, which is exactly the machine the CLI exists for.
`tests/session/test_cli.py::test_importing_the_cli_does_not_import_qt` holds
that line.

`faceforge-cli gui` delegates to the GUI, so one command still leads to the
other. Without an install, `python -m faceforge.cli ...` is equivalent.

> The `faceforge-cli` console script appears after the package is
> (re)installed — `pip install -e .`. An existing editable install that
> predates this change still exposes only `faceforge`; use
> `python -m faceforge.cli` until it is reinstalled.

## Regenerating a figure

```bash
faceforge-cli render --state figures/fig3a.state.json --out fig3a.png
```

Nothing else is needed. A state file records the BodyParts3D `source_id` of
every structure it contains, so the geometry is rebuilt from the file itself.
Run it twice and the two PNGs are byte-identical.

```bash
faceforge-cli render --state fig3a.state.json --out fig3a_print.png --size 2048x2048
faceforge-cli render --state fig3a.state.json --out fig3a_xray.png  --mode XRAY
faceforge-cli batch  --states figures/ --out build/figures/
```

`batch` renders every `*.state.json` in a directory through one GL context and
writes `batch_manifest.json` alongside the images, with a SHA-256 per figure.

## What a state file does *not* record

Two limitations worth knowing before trusting a reproduction:

* **Per-node transforms.** A posed scene's pose lives in the state's `face` and
  `body` blocks, which are replayed by the animation systems rather than by
  geometry loading. Rebuilding from provenance produces an identity-posed
  scene. Use `--tier` for states captured from a posed GUI scene.
* **Whether the BP3D → skull coordinate transform was applied.** `--transform
  none` (the default) reproduces raw STL coordinates, which is what a
  script-built scene has; `--transform bp3d` matches a scene built by the app's
  batch loader.

## Config drift

The anatomy configs decide which STL is loaded under which name, with what
colour and opacity. A state file carries a fingerprint of them. Rendering
against changed configs warns loudly and continues — re-rendering an old state
against updated configs is legitimate, doing it unknowingly is not. Pass
`--require-config-match` to make it exit 4 instead, writing nothing.

## Scanning and exporting need no GL

`scan` and `export` never touch a framebuffer, so they work on a node with no
display at all.

```bash
faceforge-cli scan --state fig3a.state.json --out ct.png \
    --mode ct --orientation coronal --position -0.8 -91.1 1556.6 \
    --width 200 --height 200 --depth 8 --resolution 256 --npy ct.npy
faceforge-cli export --state fig3a.state.json --out fig3a.glb
```

`--position` takes three separate numbers rather than `x,y,z` so that negative
anatomical coordinates parse. `--npy` saves the raw float32 array before
colour mapping, for quantitative use. GLB is the only export format
`faceforge.export` supports today; video export needs a live Qt GL widget to
grab frames from and is therefore not available headlessly.

## The Session API

The CLI is a thin shell over `faceforge.session`:

```python
from faceforge.session import Session

with Session.create(width=512, height=512) as s:
    s.load_anatomy(tier=1)                  # or structures=[...] / layers=[...]
    s.apply_state("fig3a.state.json")
    image = s.render()                      # (H, W, 4) uint8
    s.save_png("fig3a.png", image=image)
    state = s.capture_state()               # for the next paper
```

Importing the module requires neither Qt nor a display; `Session.create`
acquires the context through `tools/glcontext.py` and renders through the same
`GLRenderer` the GUI uses.

**One session per process.** OpenGL object names — VAOs above all — are scoped
to a context, and `MeshInstance.gl_handle` caches an upload on the mesh, which
outlives any renderer. Two live sessions can therefore hand each other stale
handles; that is the failure that broke an earlier benchmark harness here.
`Session.create` refuses while another session is live, and `close()` releases
the framebuffer, the renderer and every mesh's `gl_handle`, so the next session
can reuse the very same `MeshInstance` objects. Sessions are sequential, not
concurrent.

**Never a blank image.** No context is an exception, not a sentinel; a frame
that comes back a single uniform colour raises `BlankFrameError` rather than
being written. `tools/capture_gui_screenshots.py` destroyed 11 tracked README
images by writing blank frames while exiting 0.

## What is proven, and how

| claim | evidence |
|---|---|
| a state file reproduces its render | `test_session_gl.py::test_a_state_file_reproduces_its_render_exactly` — 0 of 16,384 pixels differ after save → load → rebuild → apply → render |
| the comparison can see a change | a +0.05° fov nudge moves >1% of pixels |
| the Session is the same renderer | `test_session_render_equals_capture_golden` — 0 differing pixels vs `tools/capture_golden.py`, SOLID and XRAY, diffed with `tools/compare_golden.py` at threshold 0 |
| the CLI reproduces the golden capture from a state file alone | `test_cli_end_to_end.py::test_render_from_a_state_file_equals_capture_golden` |
| renders are byte-identical between runs | `test_render_twice_produces_byte_identical_files` |
| batching does not change pixels | `test_batch_matches_single_render_byte_for_byte` |
| two sessions cannot corrupt each other | `test_a_second_concurrent_session_is_refused_and_the_first_still_works`, `test_meshes_can_be_reused_by_a_later_session` |

The context reached in a sandbox is Apple's software rasteriser: correct
pixels, meaningless timings. No frame time measured through it is renderer
performance.
