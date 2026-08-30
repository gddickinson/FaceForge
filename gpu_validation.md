# GPU validation: what is proven, what is stubbed

Status of visual (pixel-level) validation of the FaceForge renderer, as of
2026-08-29, commit `750775a` (working tree dirty — see the caveat at the end).

## The headline: headless GPU validation needs no user action

The 2026-08 audit recorded "no OpenGL or Metal context available inside this
sandbox" as a hard constraint, having proven six separate paths fail. That
finding was correct for every path it tested. **A seventh path works.**

`CGLChoosePixelFormat` returns `10017 kCGLBadConnection` because its default
renderer search asks the window server to enumerate displays, and there is no
window-server connection. Naming a renderer explicitly skips that step:

```
kCGLPFARendererID (70) = kCGLRendererGenericFloatID (0x00020400)
```

Apple's software rasteriser is not attached to a display, so it needs no
connection. Measured in the sandbox:

| probe | result |
|---|---|
| `MTLCopyAllDevices()` | non-NULL `NSArray*`, **`count = 0`** — Metal genuinely denied |
| `MTLCreateSystemDefaultDevice()` | `NULL` (confirms the prior finding) |
| CGL, `kCGLPFAAccelerated` + GL3_Core | `CGLChoosePixelFormat err=10017` |
| CGL, `kCGLPFAAccelerated` + offline | `CGLChoosePixelFormat err=10002` |
| **CGL, `kCGLPFARendererID=GenericFloat` + GL3_Core** | **context created** |
| CGL, no attributes at all | `err=10017` |

The resulting context:

```
GL_VERSION            4.1 APPLE-23.1.1
GL_RENDERER           Apple Software Renderer
GL_VENDOR             Apple Inc.
GLSL                  4.10
GL_MAX_CLIP_DISTANCES 8            (default.vert needs 1)
FBO status            GL_FRAMEBUFFER_COMPLETE
```

So `MTLCopyAllDevices` confirms the sandbox has no GPU, and CGL's software
rasteriser gives a real GLSL compiler, a real rasteriser and a real framebuffer
anyway. This is a **security boundary respected, not circumvented**: no GPU is
reached, no display connection is opened. It is CPU rendering.

**Consequence:** every render mode can be captured, diffed and regression-tested
headlessly, with no user in the loop and no remote compute.

## Proven with real pixels

The full `GLRenderer` — not a reimplementation — renders the fixed scene in the
sandbox:

| | |
|---|---|
| Scene | 16 cranial STL meshes, **230,490 triangles** |
| Modes | all 16, 0 GL errors, 0 blank frames |
| Content per mode, 256×256 | 21.15% (POINTS) – 24.81% (WIREFRAME) of pixels non-background |
| Content per mode, 512×512 | 13.83% (POINTS) – 24.62% (SOLID) of pixels non-background |
| Wall time | 9.2 s at 256×256, **14.4 s at 512×512** (all 16 modes) |

The two content ranges differ because POINTS draws distance-scaled point sprites:
at a fixed point size the sprites cover proportionally less of a larger
framebuffer, so its coverage falls from 21.15% to 13.83% while the solid modes
stay near 24.6%. Both ranges are read from the `content_fraction` block of the
respective `manifest.json`.

Every mode renders a recognisable skull and is visually distinct. XRAY,
HOLOGRAM, BLUEPRINT and ETHEREAL visibly blend, which independently confirms
the audit's `_MODE_NEEDS_BLENDING` fix at the pixel level — previously that fix
was only argued from source.

### Measured noise floor: zero

Two back-to-back captures of an unmodified tree, separate processes:

| | |
|---|---|
| `max_abs` over all 16 modes | **0** |
| `mean_abs` over all 16 modes | **0.000000** |
| Differing pixels, worst mode | **0** of 65,536 |

Repeat captures on the same machine and driver are **bit-identical**, so any
non-zero difference is signal. The thresholds in `compare_golden.py`
(`pixel_threshold=2`, `frac_tolerance=0.0001`) therefore sit entirely above a
measured floor of zero — they are headroom against a future OS or driver update
introducing dither, not a guess.

Re-measure on any new machine or driver with:
```
python -m tools.compare_golden --measure-noise-floor A B
```

### Positive and negative control, on real renders

Perturbing one line of `thermal.frag` (`facing * 0.6 + diff * 0.4` →
`0.62 / 0.38`) and re-capturing:

| mode | max_abs | mean_abs | frac > threshold | bbox | verdict |
|---|---|---|---|---|---|
| THERMAL | 11 | 0.6798 | 18.5043% (12,127 px) | 57,46,188,218 | **CHANGED** |
| all 15 others | 0 | 0.00000 | 0.0000% | – | unchanged |

Exactly the perturbed mode was identified, with no contamination of any other
mode. The negative control (two identical captures) reported zero changed modes
and emitted no contact sheets. The shader was then restored and verified
bit-identical to before the edit (`original-vs-restored max_abs = 0`).

## CPU shader semantics, cross-checked against the driver

`tests/rendering/test_shader_compile.py` proves the GLSL compiles and links. It
cannot prove the shader computes the right thing — a shader returning pure
black, or with inverted lighting, compiles perfectly.

`tools/glsl_cpu.py` transliterates all 16 fragment shaders into numpy, and
`tests/rendering/test_shader_semantics.py` asserts behaviour over a swept input
domain (Fibonacci-sphere normals, non-unit lengths, near/far view positions,
several grid and hatch periods): output range, alpha semantics, lighting
monotonicity, silhouette behaviour, clip-plane sign convention.

A transliteration only carries weight if it *is* the shader, so it is verified
rather than assumed. `tests/rendering/test_shader_gpu_agreement.py` links each
real `.frag` against a pass-through vertex shader, renders fragments with exactly
known varyings, and compares the driver's pixels to numpy:

**Worst disagreement across 15 modes × 8 geometric cases × 5 pixels: 0.501 / 255.**

That is half an 8-bit level — pure quantisation rounding. Per-mode worst:

```
HOLOGRAM 0.501  BLUEPRINT 0.500  PEN_INK 0.500  SEPIA   0.497  MEDICAL   0.496
ILLUSTR. 0.488  THERMAL   0.484  COLOR_A 0.481  PORCEL. 0.481  ETHEREAL  0.480
SOLID    0.454  OPAQUE    0.454  XRAY    0.445  WIREFR. 0.400  CARTOON   0.340
```

The test's tolerance is 2/255, four times the observed margin. The invariant
tests are therefore asserting against maths that provably matches the shipped
GLSL.

POINTS is the exception: `points.frag` reads `gl_PointCoord`, undefined for a
triangle primitive, so it is linked but not pixel-compared. Its maths is covered
on the CPU side only — **labelled unverified against the driver**.

## The autonomous loop

```
capture reference  ->  change code  ->  capture again  ->  compare  ->  verdict
```

Every stage runs headless in one process, with no user action:

```
python -m tools.capture_golden --out captures/ref
# ... make a change ...
python -m tools.capture_golden --out captures/cur
python -m tools.compare_golden captures/ref captures/cur --sheets sheets/
```

Exit 0 = nothing changed, 1 = something did (with per-mode magnitudes, bounding
boxes and contact sheets), 3 = the captures are not comparable.

## Not proven / stubbed

- **The hardware Metal driver is untested.** Everything above is the CPU
  rasteriser. `compare_golden.py` *refuses* to diff across different
  `GL_RENDERER` strings, because the cross-driver noise floor is unmeasured —
  hardware references need hardware comparisons. Start
  `tools/render_agent.py` in a desktop session to get them.
- **`points.frag` pixel output is unverified against a driver** (see above).
- **The frame times here are CPU numbers and are not renderer performance.**
  Software rasterisation is orders of magnitude off an M1 Max. Never quote them.
- **`render_agent.py`'s GL path has not run on hardware.** Its validation,
  lifecycle, locking and containment are covered by 95 tests, and its render
  path has run end to end against the software rasteriser. Its behaviour against
  the Metal driver is inferred, not measured.
- **The comparison is per-pixel, not structural.** A change that moves geometry
  by one pixel everywhere reports as a large change; the harness reports
  magnitude and location, it does not classify cause.
- **Captures were taken on a dirty working tree** (`git_dirty: true`), which
  every manifest records. "Same commit" does not mean "same code" here; the
  prior audit's shader work is uncommitted. Nothing above depends on the
  distinction, since the controls compare captures to each other, but a
  reference set stored for future use should be taken from a clean tree.

## The one command the user must run

For correctness validation on this machine: **none.** It is headless.

To validate against the hardware Metal driver the way users see it, in a
terminal in your desktop session:

```
python -m tools.render_agent
```

Ctrl-C stops it. `.render_agent/README.md` covers job submission and the
security model.

## Files

| file | role |
|---|---|
| `tools/glcontext.py` | context acquisition: existing → hardware → software, raises if none |
| `tools/capture_golden.py` | fixed-scene FBO capture → PNG + manifest; `--selftest` |
| `tools/compare_golden.py` | per-mode perceptual diff, contact sheets, `--measure-noise-floor` |
| `tools/render_agent.py` | validated job watcher for a GUI session; `--self-check` |
| `tools/glsl_cpu.py` | numpy transliteration of all 16 fragment shaders |
| `tests/tools/test_render_agent_validation.py` | 95 tests: hostile jobs, lifecycle, containment |
| `tests/tools/test_compare_golden.py` | 35 tests: negative control, isolation, refusals |
| `tests/rendering/test_shader_semantics.py` | CPU invariants over a swept domain |
| `tests/rendering/test_shader_gpu_agreement.py` | numpy vs. driver pixels |
| `.render_agent/README.md` | how to run the agent |
