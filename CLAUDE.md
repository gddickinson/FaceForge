# CLAUDE.md — FaceForge

Project instructions for Claude Code. The global preferences in
`~/.claude/CLAUDE.md` apply (files under 500 lines, modular, INTERFACE.md).

@INTERFACE.md

## What this is

An anatomy visualisation and animation app: BodyParts3D meshes, a FACS face,
a full-body kinematic rig with soft-tissue skinning, a virtual scanner, and
(since 2026-09) exercise demonstrations with a muscle-activation heatmap.

## Environment

- Use `/opt/anaconda3/envs/flika/bin/python` (the editable install); the
  shell's default `python` is a different environment without PyOpenGL.
- `export QT_QPA_PLATFORM=offscreen` before running tests.
- Fast tier: `pytest -m "not slow"` needs no assets and must stay green
  (CI runs it with a dangling `assets/stl`). One pre-existing failure,
  `tests/export/test_mesh_export.py::test_obj_groups_name_the_bodyparts3d_source_ids`,
  predates the exercise work.
- The asset set is reachable through the `assets/stl` symlink on this
  machine (930/932 meshes); `tools/headless_loader.py` loads it in ~40 s.

## Conventions that were measured, not assumed

- Body frame: +Z superior, −Y anterior, +X right. Standing wrapper =
  Rx(−90°) at Y = 203; the soles are then on the floor (Y = 0).
- Positive DOF values: flexion, abduction, external rotation, dorsiflexion,
  supination. Ranges in `faceforge/body/dof_ranges.py`; limits in
  `assets/config/body_joint_limits.json`.
- The arm chains hang off `bodyRoot`, not the thoracic spine: `spine_flex`
  does not move the shoulders. Lean the trunk with a wrapper pitch.
- Never trust a render that exits 0: check pixels (`Session.render` refuses
  blank frames).

## Working rules

- Keep files under 500 lines; one concept per file.
- Poses in the exercise catalogue are authored in degrees through
  `faceforge/exercise/pose_library.py`, never as raw normalised numbers.
- After changing the catalogue run `python -m tools.export_exercise_docs`
  and `pytest tests/exercise`.
- Do not modify the untracked files `bench_influences.json`,
  `verify_state_pixels.py`, `wu_shoulder_attachments.json`; they belong to
  other in-progress work.
- Update `INTERFACE.md` and `SESSION_LOG.md` at the end of a session.
