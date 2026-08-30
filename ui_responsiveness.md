# UI responsiveness

The render loop is a 16 ms `QTimer` on the main thread. Any handler that blocks
for one second drops ~60 frames, so per-interaction wall time is a correctness
property of the UI layer, not a nicety. This document records what was measured,
what was changed, and the budget the test suite now enforces.

All numbers below were measured on the reference machine (Apple M1 Max, macOS,
Python 3.11, `QT_QPA_PLATFORM=offscreen`) with the `.npz` welded-geometry mesh
cache warm. No GL context exists in that environment, so the viewport never
paints; everything else in the application constructs and responds normally,
which is what makes a headless sweep possible at all.

---

## 1. The reported defect was misattributed

The earlier headless driver reported one slow interaction: `QCheckBox("Skin")`
at 3.18 s, everything else under 0.5 s. Profiling that interaction shows the
3.18 s is **not** the Skin handler.

`app.py:2242` arms the deferred whole-scene load:

```python
QTimer.singleShot(100, load_assets)
```

`load_assets()` runs `pipeline.load_head()`, `load_body_skeleton()`, the gender
morph and the face features — on the main thread, inside whichever
`processEvents()` call happens to be executing when that 100 ms timer expires.
In the driver's ordering (93 buttons, then 70 checkboxes) the timer fired
partway through the checkbox pass, and the cost landed on whichever control was
being toggled at that moment.

`cProfile` around the Skin toggle in the driver's own ordering, 5.46 s wall
(profiler overhead inflates this; unprofiled it is ~3.2 s):

| cumtime | call |
| --- | --- |
| 5.442 s | `{built-in method processEvents}` |
| 5.442 s | `app.py:1588(load_assets)` |
| 3.059 s | `loading_pipeline.py:159(load_head)` |
| 2.345 s | `loading_pipeline.py:292(load_body_skeleton)` |
| 2.301 s | `anatomy/skull.py:42(build_original_skull)` |
| 1.994 s | `body/gender_morph.py:104(load)` |
| 0.841 s | `body/region_labels.py:229(build_region_kdtrees)` |

`register_skin_mesh` does not appear at all: because `load_assets` had not yet
run, the skinning object had no joints, so binding was skipped entirely.

This startup load is genuine one-time work that must complete before the app is
usable, and the GL-owning thread cannot be relieved of it without a much larger
change. **It was not reduced.** What changed is that it is no longer charged to
an unrelated control: the test harness drains it explicitly
(`gui_harness.drain_deferred_startup`) before any timing is collected, so the
budget measures interactions rather than startup.

## 2. The real Skin toggle cost: 34.54 s

Draining startup first and *then* toggling Skin — the sequence a real user
performs — exposes the actual defect, an order of magnitude worse than the
reported figure:

```
build=1.06s drain=3.91s errors=0
SKIN_TOGGLE_WALL 34.541
re-toggle off: 0.0017s
re-toggle on:  0.0001s
```

`cProfile`, sorted by internal time:

| tottime | cumtime | call |
| --- | --- | --- |
| 12.036 s | 34.143 s | `soft_tissue.py:286(register_skin_mesh)` |
| 11.454 s | 15.767 s | `soft_tissue.py:1167(_geodesic_chain_dists)` |
| 3.827 s | — | `{method 'reduce' of 'numpy.ufunc'}` |
| 2.745 s | — | `{method 'sort' of 'numpy.ndarray'}` |
| 0.335 s | 0.468 s | `soft_tissue.py:1017(_smooth_boundary_weights)` |
| 0.121 s | 0.662 s | `soft_tissue.py:1307(_build_neighbor_data)` |
| 0.068 s | 2.924 s | `soft_tissue.py:1135(_extract_mesh_edges)` |

`register_skin_mesh` accounts for 34.14 s of the 34.48 s. It is called exactly
once, with:

```
mesh=Skin  vertices=791,729  triangles=1,586,498  joints=148  chains=26
is_muscle=False  chain_z_margin=15.0  spatial_limit=25.0
```

The solve builds `(V, S)` and `(V, S, 3)` float64 arrays over 791,729 vertices
and ~140 bone segments — the `(V, S, 3)` intermediates (`ap`, `closest`,
`diff`) are ~2.6 GB each — then runs a Dijkstra pass over 2,379,747 mesh edges
for the geodesic chain separation. The re-toggle timings above (0.0017 s /
0.0001 s) confirm the work is one-time per process, guarded by `_skin_loaded`.

## 3. Fix

The solve is a deterministic function of the mesh geometry, the joint rest
configuration, the call parameters and the tunables. So it is memoised on disk,
following the pattern `faceforge.loaders.stl_parser` already uses for welded
geometry.

**`src/faceforge/body/skinning_cache.py`** (new). Keyed on a blake2b digest of
the rest positions, the index buffer, the per-joint rest translations / chain
IDs / bone segments, the call parameters, and *every* public scalar attribute of
the skinning object (hashed automatically rather than from an enumerated list,
so a future tunable cannot silently serve a stale binding). Only applied above
`MIN_VERTS = 100_000`, so the hundreds of small muscle registrations pay no
hashing cost. Cache directory follows the existing convention
(`$XDG_CACHE_HOME`/`~/.cache/faceforge/skinning`), overridable with
`FACEFORGE_SKIN_CACHE_DIR` and disableable with `FACEFORGE_SKIN_CACHE_OFF`.
Writes are atomic via `os.replace`; a miss, a corrupt entry or a read-only
directory falls back to solving.

**`src/faceforge/body/soft_tissue.py`.** `register_skin_mesh` was split: the
solve moved into `_solve_skin_binding`, which returns
`(joint_indices, secondary_indices, weights, precomputed_edges)` and does the
cache lookup/store; `register_skin_mesh` is now a thin wrapper that turns those
arrays into a `SkinBinding`. The ~300 lines of solve logic between the new head
and tail are untouched.

**`_build_neighbor_data`, same file.** Once the solve was cached this became
the entire remaining cost (0.676 s of a 0.735 s warm toggle), 0.448 s of it in
14 `np.add.at` calls. Every one is a scatter-add of one value per edge endpoint
into a per-vertex total, so they were replaced with `np.bincount`. Concatenating
`[e0, e1]` reproduces the exact accumulation order of the two successive
`np.add.at` calls each one replaces, so the float64 results are bitwise
identical — verified below. The per-vertex Python list comprehension over
791,729 joint lookups was also replaced with a table lookup, and `total_count`
(which is `neighbor_counts` recomputed as float) is now reused rather than
re-scattered.

### Measured, before and after

Same instrumentation in both cases (`build_main_window` → `drain_deferred_startup`
→ toggle the `QCheckBox("Skin")`, unprofiled wall time):

| what | before | after | speedup |
| --- | --- | --- | --- |
| Skin toggle, whole interaction | 34.541 s | 0.332 s | 104x |
| `_solve_skin_binding` alone | 32.610 s | 0.040 s | 815x |
| `_build_neighbor_data` | 0.676 s | ~0.25 s | ~2.7x |
| Skin toggle in the test harness sweep | 34.094 s | 0.315–0.328 s (3 runs) | ~105x |

Cache entry for the skin mesh is 28.5 MB.

### Correctness

Fresh solve versus cache hit, same process, same parameters:

```
SOLVE_NO_CACHE 32.610s
SOLVE_CACHED    0.040s
  IDENTICAL joint_indices      True dtype=int32
  IDENTICAL secondary_indices  True dtype=int32
  IDENTICAL weights            True dtype=float32
  IDENTICAL edges              True dtype=uint32
```

`np.bincount` rewrite versus the `np.add.at` reference it replaced, on the real
skin mesh (V=791,729, E=2,379,747):

```
  BINCOUNT_IDENTICAL counts  True
  BINCOUNT_IDENTICAL sum     True  maxabs=0.000e+00
```

An earlier comparison of two `SkinBinding` objects showed `joint_indices`,
`secondary_indices` and `weights` differing while the neighbour arrays matched.
That was not a cache defect: 764 and 766 vertices respectively are rewritten
after registration by the 766 vertex overrides in
`assets/config/skinning_overrides.json`. Comparing solver output directly, as
above, isolates the cache.

### What was not fixed

The first run on a machine still pays the full ~34 s solve once, to build the
cache entry. Nothing was moved off the main thread and no work was made
incremental — the total CPU cost of a cold solve is unchanged. The startup
`load_assets` freeze (~3.2 s warm) is also unchanged; see §1.

---

## 4. The budget

`tests/ui/test_gui_smoke.py` asserts `INTERACTION_BUDGET_S = 1.5`.

Set from measurement. Three consecutive full sweeps after the fix, 789
interactions each:

| run | interactions | failures | max | p99 | p50 |
| --- | --- | --- | --- | --- | --- |
| 1 | 789 | 0 | 0.363 s | 14.0 ms | 0.03 ms |
| 2 | 789 | 0 | 0.364 s | 8.3 ms | 0.03 ms |
| 3 | 789 | 0 | 0.363 s | 9.3 ms | 0.03 ms |

The worst interaction is `QCheckBox("Stretch Heatmap")` at 0.364 s, consistent
to within 1 ms across runs; `QCheckBox("Skin")` is second at 0.315–0.328 s.
Everything else is under 70 ms.

1.5 s is ~4.1x the worst observed interaction. The headroom absorbs a slower CI
runner (the reference machine is fast single-threaded) and incidental cache
variance, while still sitting ~22x below the 34 s regression the budget exists
to catch. It is deliberately not tightened to, say, 0.5 s: that would sit only
1.4x above a measured value and would flake on slower hardware without catching
anything a 1.5 s bound misses.

Verified that the budget actually detects the defect — with the cache disabled:

```
$ FACEFORGE_SKIN_CACHE_OFF=1 pytest tests/ui/test_gui_smoke.py
E  AssertionError: 1 interaction(s) over the 1.50 s budget
E        33.25s  QCheckBox(Skin)
```

### Cold machines

The very first run on a machine has to build the binding cache and legitimately
spends ~34 s inside the "Skin" interaction. The fixture counts cache entries
before and after the sweep; if the run built one, the budget assertion skips
itself once with an explanatory message and every later run asserts normally.
Verified:

```
$ FACEFORGE_SKIN_CACHE_DIR=/tmp/empty pytest tests/ui/test_gui_smoke.py -rs
SKIPPED [1] tests/ui/test_gui_smoke.py:150: this run built the skin binding
cache for the first time on this machine (~34 s of one-time solve inside the
'Skin' interaction); re-run to assert the steady-state budget
4 passed, 1 skipped in 39.21s
```

The other four assertions run in both cases, so a cold machine still gets the
zero-exception and coverage guarantees.

---

## 5. The test module

| file | role |
| --- | --- |
| `tests/ui/gui_harness.py` | reusable harness: dialog stubs, `build_main_window`, `drain_deferred_startup`, `walk_tabs`, `sweep`, `sweep_open_dialogs`, `summarise` |
| `tests/ui/test_gui_smoke.py` | five assertions over one module-scoped sweep, marked `slow` |
| `tests/body/test_skinning_cache.py` | seven fast unit tests for the cache (bitwise round-trip, key coverage, opt-out, corruption, unwritable directory) |

Assertions:

1. `test_app_constructs_without_exceptions` — the real `app.main()` raises nothing, including inside Qt slots (captured via `sys.excepthook`, since Qt swallows slot exceptions).
2. `test_all_six_tabs_open` — ANIMATE, BODY, LAYERS, ALIGN, DISPLAY, DEBUG all present and selectable.
3. `test_every_control_interacts_without_raising` — 789 interactions, 0 failures.
4. `test_interaction_coverage_has_not_regressed` — floors per control family, so a refactor that makes `findChildren` return nothing fails loudly instead of asserting on an empty list.
5. `test_no_interaction_blocks_the_render_thread` — the 1.5 s budget.

Coverage per sweep: 123 buttons, 71 checkboxes, 1 radio, 231 slider positions,
352 combo selections, 5 spinboxes, 6 tabs.

Runtime: 6.8 s warm, 39.2 s on a cold binding cache. Marked
`pytest.mark.slow`; deselect with `pytest -m "not slow"`.

### Why it cannot hang CI

`gui_harness.stub_blocking_calls()` neutralises `QDialog.exec` (StartupDialog
runs a nested event loop that would never return headless), `QApplication.exec`,
and every *static* dialog helper — `QColorDialog.getColor`,
`QFontDialog.getFont`, the four `QFileDialog` getters, the four `QInputDialog`
getters, and `QMessageBox.{information,warning,critical,about,aboutQt,question}`.
The static helpers matter specifically: they build and run their own dialog
internally and never route through the `QDialog.exec` override, so without them
the first `colorButton` click on the DISPLAY tab blocks forever. Opening a
picker is correct behaviour, not a defect; the stubs exist so the sweep can
continue past it.

`QMessageBox.question` returns `No`, so a destructive confirmation is never
confirmed. Buttons whose label contains quit / exit / close / save / load /
export / import / open / browse / record / screenshot are not clicked
(`gui_harness.SKIP_BUTTON_WORDS`): they would end the process or open a native
file dialog outside Qt's control.

Dialogs the sweep opens — `QuizDialog`, `ComparisonDialog`, `TimelineEditor`,
`StartupDialog` — are swept in a separate pass (`sweep_open_dialogs`), labelled
with their class name. Keeping them out of the main-window pass makes the main
sweep independent of which buttons happened to construct which dialog.
