# FaceForge render agent

Gives automated golden-image validation access to this machine's GPU, without a
human judging images.

## Start it

From the repo root, in a terminal in your logged-in desktop session:

```
python -m tools.render_agent
```

Stop it with **Ctrl-C** (it finishes the job in flight, then exits; a second
Ctrl-C exits immediately). It is not a LaunchAgent and does not install
anything — it runs only while that terminal is open, so it stays visibly
revocable.

## Do you actually need it?

Probably not, for correctness work. `tools/glcontext.py` reaches Apple's
software rasteriser without a window server, so `tools/capture_golden.py` runs
headless — all 16 modes at 512×512 in about 14 s. Measured 2026-08-29:
`GL_RENDERER = Apple Software Renderer`, `GL_VERSION = 4.1 APPLE-23.1.1`.

Start the agent when you want captures from the **hardware Metal driver** —
the one users actually see. `tools/compare_golden.py` refuses to diff a
software capture against a hardware one, because the cross-driver pixel noise
floor is unmeasured, so hardware references need hardware comparisons.

## What it does

Polls `.render_agent/jobs/*.json` about once a second. For each job file:
validate → render → write results → write status → move the job to `done/`.

```
.render_agent/
  jobs/            drop job files here
  jobs/done/       processed job files (MOVED here, never deleted)
  results/<id>/    PNGs + manifest.json for job <id>
  status/<id>.json state, timing, reason-for-rejection
  agent.lock       holds the running instance's pid
```

## Submitting a job

A job file carries **parameters only**. Every key is optional.

```
cat > .render_agent/jobs/baseline.json <<'EOF'
{"modes": ["SOLID", "XRAY"], "meshes": 16,
 "size": "512x512", "camera": "oblique", "label": "baseline"}
EOF
```

| key | type | accepted | default |
|---|---|---|---|
| `modes` | list of strings | any of the 16 `RenderMode` names, exact case | all 16 |
| `meshes` | int | 1–16, an index into a **fixed** mesh list | 16 |
| `size` | string | `WxH`, each 64–4096 | `512x512` |
| `camera` | string | `anterior`, `left_lateral`, `oblique`, `right_lateral`, `superior` | `oblique` |
| `label` | string | `[A-Za-z0-9_-]{1,64}` | `job` |

The job id is the filename stem, which must itself be a slug.

## What it will not do

- read code, paths, filenames, shell strings or format strings from a job file
- `eval`, `exec`, import or subprocess anything a job asked for (an AST check in
  `--self-check` asserts none of those constructs exist in the module)
- write anywhere outside `.render_agent/`
- delete anything — processed jobs are *moved* to `done/`
- make any network call

An **unknown key, wrong type or out-of-range value rejects the whole job** with
a reason in its status file, and nothing is rendered. Unknown keys are rejected
rather than ignored: a job written against a different contract must not be
half-honoured. All render logic lives in the agent; a job selects among
pre-declared options and cannot describe anything new.

## Verify it is working

```
python -m tools.render_agent --self-check      # 8 checks, no GL, no rendering
python -m tools.capture_golden --selftest      # 16 checks, no GL
python -m pytest tests/tools/test_render_agent_validation.py -q   # 95 tests
```

The pytest module feeds the agent 60-odd hostile and malformed job files —
unknown keys, wrong types, huge sizes, path traversal in the label, non-JSON,
NUL bytes, bad UTF-8, symlinks — and asserts every one is rejected, that the
renderer is never invoked, and that nothing is written outside
`.render_agent/`.

End to end, without starting the agent:

```
echo '{"modes":["SOLID"],"meshes":4,"size":"128x128","label":"smoke"}' \
  > .render_agent/jobs/smoke.json
python -m tools.render_agent --once
cat .render_agent/status/smoke.json          # -> "state": "done"
ls .render_agent/results/smoke/              # -> SOLID.png, manifest.json
```

## Comparing captures

```
python -m tools.compare_golden REF_DIR CUR_DIR --sheets sheets/ --json report.json
```

Exit 0 = no mode changed, 1 = at least one changed, 3 = the two captures are not
comparable (different viewport, camera, mesh list, mode set or GL renderer).
Per-mode `max_abs`, `mean_abs`, fraction of pixels over threshold and a bounding
box of the changed region; a reference | current | amplified-difference contact
sheet for each changed mode.

Measured noise floor on this machine: two back-to-back captures of an unmodified
tree were **bit-identical across all 16 modes** (max_abs 0, 0 differing pixels
of 65,536 per mode), so any non-zero difference is signal. See
`gpu_validation.md`.

## If it refuses to start

`another render agent is already running (pid N)` — one instance at a time. Stop
that one, or if the pid is gone the next start takes the lock over
automatically.
