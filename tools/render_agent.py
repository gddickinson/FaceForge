"""A watcher that renders golden-image jobs on this machine's GPU.

The user starts this in a logged-in GUI session, where a hardware OpenGL
context is reachable.  It then polls ``.render_agent/jobs/`` for JSON job
files, renders each one, and writes PNGs plus a status file back.  That gives
an automated caller GPU access without a human judging images.

Security model
--------------
This process has the user's full filesystem privileges, and the jobs it reads
may be written by an automated caller.  So the agent, not the job file, holds
every piece of authority:

* **A job file carries parameters only.**  Five keys, each with a declared
  type, each validated and clamped:  ``modes`` (subset of the 16 known mode
  names), ``meshes`` (int, clamped to 1..16), ``size`` (``WxH``, clamped to
  64..4096), ``camera`` (one of the named presets), ``label``
  (``[A-Za-z0-9_-]{1,64}``).
* **Any unknown key is a rejection**, not a warning.  A job that carries a key
  this agent does not know is a job written against a different contract, and
  guessing which parts to honour is how an injection succeeds.
* **No code, path, filename, shell string or format string is ever read from a
  job file.**  There is no ``eval``, no ``exec``, no ``subprocess``, no
  ``importlib``, and no ``os.path.join`` on job-supplied text anywhere in this
  module.  The only job value that reaches a filesystem path is ``label``, and
  it reaches it only after passing a strict allowlist regex *and* an explicit
  containment check against the results root.
* **All writes are confined to ``.render_agent/``.**  Output paths are built
  from the job's own id -- which comes from the filename the agent itself
  enumerated, re-validated as a slug -- never from job content.
* **No network calls.**  This module imports no networking library.
* **Nothing is deleted outside ``done/``.**  A processed job file is *moved*
  into ``.render_agent/jobs/done/``, never unlinked.
* **One instance at a time**, enforced by a lock file holding a live pid.

What it deliberately does not do: run arbitrary render parameters (only
presets), accept a mesh list (only a count into the fixed list), write outside
its own directory, or delete anything.  If a caller needs a scene this cannot
express, that belongs in ``capture_golden.py`` as a new named preset, reviewed
as code -- not in a job file.

Usage
-----
    python -m tools.render_agent                 # start; Ctrl-C to stop
    python -m tools.render_agent --once          # drain the queue and exit
    python -m tools.render_agent --self-check    # validate wiring, no render

A job file is JSON, dropped into ``.render_agent/jobs/<id>.json``::

    {"modes": ["SOLID", "XRAY"], "meshes": 16,
     "size": "512x512", "camera": "oblique", "label": "baseline"}

Every key is optional; omitted keys take the documented default.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("render_agent")

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = REPO_ROOT / ".render_agent"
JOBS_DIR = AGENT_DIR / "jobs"
DONE_DIR = JOBS_DIR / "done"
RESULTS_DIR = AGENT_DIR / "results"
STATUS_DIR = AGENT_DIR / "status"
LOCK_FILE = AGENT_DIR / "agent.lock"

POLL_SECONDS = 1.0

# A slug: job ids and labels.  Anchored, bounded, and containing no separator,
# no dot and no NUL, so it cannot climb out of a directory or name a hidden
# file.  This is an allowlist, deliberately -- a denylist of dangerous
# sequences is a losing game.
SLUG_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# The complete set of accepted job keys.  Anything else is a rejection.
ALLOWED_JOB_KEYS = frozenset({"modes", "meshes", "size", "camera", "label"})

MAX_JOB_BYTES = 8192          # a parameters-only job is ~120 bytes
MAX_MODES_PER_JOB = 16


class JobRejected(ValueError):
    """A job file failed validation.  Nothing was rendered."""


class AgentLocked(RuntimeError):
    """Another agent instance holds the lock."""


@dataclass
class ValidatedJob:
    """The only shape the renderer is ever handed.

    Every field here is a value this module produced from a validated
    primitive -- never a string echoed out of the job file into a path.
    """

    job_id: str
    modes: list[str]
    meshes: int
    size: tuple[int, int]
    camera: str
    label: str
    source: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "modes": self.modes,
            "meshes": self.meshes,
            "size": f"{self.size[0]}x{self.size[1]}",
            "camera": self.camera,
            "label": self.label,
            "warnings": self.warnings,
        }


# ----------------------------------------------------------------------------
# Validation.  Pure functions, no filesystem, no GL -- fully testable.
# ----------------------------------------------------------------------------

def _known_modes() -> tuple[str, ...]:
    from tools.capture_golden import ALL_MODES
    return ALL_MODES


def _known_cameras() -> list[str]:
    from tools.capture_golden import CAMERA_PRESETS
    return sorted(CAMERA_PRESETS)


def _limits() -> tuple[int, int, int, str]:
    from tools.capture_golden import DEFAULT_CAMERA, MAX_MESHES, MAX_SIZE, MIN_SIZE
    return MIN_SIZE, MAX_SIZE, MAX_MESHES, DEFAULT_CAMERA


def validate_job_id(name: str) -> str:
    """Validate a job id (derived from a filename the agent enumerated)."""
    if not isinstance(name, str) or not SLUG_RE.match(name):
        raise JobRejected(
            f"job id {name!r} is not a slug matching {SLUG_RE.pattern}. "
            "Job filenames must be plain slugs; this blocks path traversal."
        )
    return name


def validate_job(raw: object, job_id: str) -> ValidatedJob:
    """Turn untrusted parsed JSON into a :class:`ValidatedJob`, or reject it.

    This is the whole trust boundary.  It accepts only a flat JSON object whose
    keys are in :data:`ALLOWED_JOB_KEYS` and whose values pass a per-key type
    and range check.  It never returns a partially-validated job.
    """
    job_id = validate_job_id(job_id)
    min_size, max_size, max_meshes, default_camera = _limits()
    known_modes = _known_modes()
    known_cameras = _known_cameras()
    warnings: list[str] = []

    if not isinstance(raw, dict):
        raise JobRejected(
            f"job must be a JSON object, got {type(raw).__name__}. "
            "Lists and scalars are rejected: only a keyed parameter set is a job."
        )

    unknown = sorted(set(raw) - ALLOWED_JOB_KEYS)
    if unknown:
        raise JobRejected(
            f"unknown key(s) {unknown}; allowed keys are {sorted(ALLOWED_JOB_KEYS)}. "
            "Unknown keys are rejected rather than ignored: a job written against a "
            "different contract must not be half-honoured."
        )

    # --- modes -----------------------------------------------------------
    if "modes" in raw:
        m = raw["modes"]
        if isinstance(m, str):
            raise JobRejected(
                "'modes' must be a list of strings, not a string. A comma-separated "
                "string would need splitting, and parsing job text into a list is "
                "exactly the flexibility this agent refuses."
            )
        if not isinstance(m, list):
            raise JobRejected(f"'modes' must be a list, got {type(m).__name__}")
        if not m:
            raise JobRejected("'modes' must not be empty")
        if len(m) > MAX_MODES_PER_JOB:
            raise JobRejected(f"'modes' has {len(m)} entries, max {MAX_MODES_PER_JOB}")
        for entry in m:
            if not isinstance(entry, str):
                raise JobRejected(
                    f"'modes' entries must be strings, got {type(entry).__name__}: {entry!r}"
                )
            if entry not in known_modes:
                raise JobRejected(
                    f"unknown render mode {entry!r}. Known modes: {list(known_modes)}. "
                    "Modes are matched exactly against the enum, case-sensitively."
                )
        seen = set(m)
        if len(seen) != len(m):
            warnings.append("duplicate modes collapsed")
        # Normalised to enum order so two jobs asking for the same set produce
        # comparable captures.
        modes = [x for x in known_modes if x in seen]
    else:
        modes = list(known_modes)

    # --- meshes ----------------------------------------------------------
    if "meshes" in raw:
        n = raw["meshes"]
        # bool is an int subclass in Python; True must not silently mean 1.
        if isinstance(n, bool) or not isinstance(n, int):
            raise JobRejected(
                f"'meshes' must be an integer, got {type(n).__name__}: {n!r}. "
                "Floats and numeric strings are rejected, not coerced."
            )
        if not 1 <= n <= max_meshes:
            raise JobRejected(f"'meshes' {n} outside the clamp range [1, {max_meshes}]")
        meshes = n
    else:
        meshes = max_meshes

    # --- size ------------------------------------------------------------
    if "size" in raw:
        s = raw["size"]
        if not isinstance(s, str):
            raise JobRejected(f"'size' must be a string like '512x512', got {type(s).__name__}")
        if len(s) > 16:
            raise JobRejected(f"'size' string too long ({len(s)} chars)")
        mm = re.fullmatch(r"(\d{1,4})[xX](\d{1,4})", s)
        if not mm:
            raise JobRejected(
                f"'size' {s!r} must match WxH with 1-4 digits each. Parsed strictly: "
                "no whitespace, signs, units or expressions."
            )
        w, h = int(mm.group(1)), int(mm.group(2))
        for lbl, v in (("width", w), ("height", h)):
            if not min_size <= v <= max_size:
                raise JobRejected(
                    f"'size' {lbl} {v} outside the clamp range [{min_size}, {max_size}]"
                )
        size = (w, h)
    else:
        size = (512, 512)

    # --- camera ----------------------------------------------------------
    if "camera" in raw:
        c = raw["camera"]
        if not isinstance(c, str):
            raise JobRejected(f"'camera' must be a string, got {type(c).__name__}")
        if c not in known_cameras:
            raise JobRejected(
                f"unknown camera preset {c!r}. Known presets: {known_cameras}. "
                "Only named presets are accepted; raw camera coordinates are not "
                "expressible in a job, by design."
            )
        camera = c
    else:
        camera = default_camera

    # --- label -----------------------------------------------------------
    if "label" in raw:
        lab = raw["label"]
        if not isinstance(lab, str):
            raise JobRejected(f"'label' must be a string, got {type(lab).__name__}")
        if not SLUG_RE.match(lab):
            raise JobRejected(
                f"'label' {lab!r} must match {SLUG_RE.pattern}. Rejected because the "
                "label is the one job value recorded alongside output; slashes, dots "
                "and NULs are refused so it can never denote a path."
            )
        label = lab
    else:
        label = "job"

    return ValidatedJob(
        job_id=job_id, modes=modes, meshes=meshes, size=size,
        camera=camera, label=label, warnings=warnings,
    )


def read_job_file(path: Path) -> tuple[object, str]:
    """Read and parse a job file.  Raises :class:`JobRejected` on anything odd.

    Size-capped before parsing, so an enormous or deeply-nested file cannot be
    used to exhaust memory in the JSON parser.
    """
    if path.is_symlink():
        raise JobRejected(
            f"{path.name} is a symlink; refused. A symlinked job could point the "
            "reader at a file outside the jobs directory."
        )
    if not path.is_file():
        raise JobRejected(f"{path.name} is not a regular file")
    st = path.stat()
    if st.st_size > MAX_JOB_BYTES:
        raise JobRejected(f"{path.name} is {st.st_size} bytes, max {MAX_JOB_BYTES}")
    raw_bytes = path.read_bytes()
    if b"\x00" in raw_bytes:
        raise JobRejected(f"{path.name} contains NUL bytes")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JobRejected(f"{path.name} is not valid UTF-8: {exc}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JobRejected(f"{path.name} is not valid JSON: {exc}") from exc
    return parsed, path.stem


# ----------------------------------------------------------------------------
# Filesystem containment
# ----------------------------------------------------------------------------

def assert_contained(path: Path, root: Path) -> Path:
    """Resolve *path* and assert it lies under *root*.

    Belt and braces: :data:`SLUG_RE` already makes traversal unexpressible.
    This is the check that would catch a future edit weakening that regex, and
    it is cheap enough to run on every write.
    """
    rp = Path(path).resolve()
    rr = Path(root).resolve()
    if rp != rr and rr not in rp.parents:
        raise JobRejected(
            f"path escape blocked: {rp} is not under {rr}. "
            "The agent writes only inside .render_agent/."
        )
    return rp


def ensure_dirs() -> None:
    for d in (AGENT_DIR, JOBS_DIR, DONE_DIR, RESULTS_DIR, STATUS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def write_status(job_id: str, payload: dict) -> Path:
    """Write a status file for *job_id*.  Path derived from the validated id."""
    job_id = validate_job_id(job_id)
    p = assert_contained(STATUS_DIR / f"{job_id}.json", AGENT_DIR)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, p)   # atomic: a reader never sees a half-written status
    return p


def move_to_done(path: Path) -> Path:
    """Move a processed job file into ``done/``.  Never deletes.

    A name collision is resolved by suffixing a counter rather than
    overwriting, so resubmitting the same id keeps both records.
    """
    dest = assert_contained(DONE_DIR / path.name, AGENT_DIR)
    if dest.exists():
        stem, suffix, n = dest.stem, dest.suffix, 1
        while dest.exists() and n < 10000:
            dest = assert_contained(DONE_DIR / f"{stem}.{n}{suffix}", AGENT_DIR)
            n += 1
    os.replace(path, dest)
    return dest


# ----------------------------------------------------------------------------
# Locking
# ----------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, owned by someone else
    except OSError:
        return False
    return True


class AgentLock:
    """Refuse to run if another instance holds the lock with a live pid.

    A stale lock (pid gone) is taken over and logged.  Uses O_EXCL so two
    agents starting at the same instant cannot both win.
    """

    def __init__(self, path: Path | None = None):
        # Resolved at call time, not at import: a default argument of
        # ``LOCK_FILE`` binds the module constant once when the class is
        # defined, so any later reassignment of LOCK_FILE (a test redirecting
        # the agent into a temporary directory, or a future --agent-dir flag)
        # was silently ignored and the lock kept pointing at the real repo.
        self.path = Path(path) if path is not None else LOCK_FILE
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = self._read_holder()
            if holder is not None and _pid_alive(holder):
                raise AgentLocked(
                    f"another render agent is already running (pid {holder}, "
                    f"lock {self.path}). Stop it first, or delete the lock file if "
                    "you are certain it is stale."
                ) from None
            logger.warning("stale or blank lock (pid %s) — taking over", holder)
            # Truncate in place rather than unlink: this module never deletes.
            # O_EXCL is deliberately absent here -- the file provably exists and
            # its holder is provably not running.
            fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "started": time.time()}, f)
        self.acquired = True

    def _read_holder(self) -> int | None:
        try:
            data = json.loads(self.path.read_text() or "{}")
            pid = data.get("pid")
            return int(pid) if isinstance(pid, int) else None
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            return None

    def release(self) -> None:
        """Blank the lock rather than removing it, keeping the no-delete rule."""
        if not self.acquired:
            return
        try:
            self.path.write_text("")
        except OSError:
            logger.warning("could not clear lock file %s", self.path)
        self.acquired = False

    def __enter__(self) -> "AgentLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

def default_render_fn(job: ValidatedJob, out_dir: Path) -> dict:
    """Render *job* by calling capture_golden with validated primitives.

    Note what crosses this boundary: a list of enum-member names, two ints, a
    preset name and a slug.  ``out_dir`` is built by the caller from the job
    id, not from job content.  No string from the job file is interpolated
    into a path, a command or a format string.
    """
    from tools.capture_golden import capture

    manifest = capture(
        out_dir,
        modes=job.modes,
        mesh_count=job.meshes,
        size=job.size,
        camera_name=job.camera,
        label=job.label,
        force=True,      # out_dir is the agent's own per-job directory
        prefer="auto",   # hardware in a GUI session; software otherwise
    )
    return manifest


def process_job_file(
    path: Path, render_fn=default_render_fn, *, move_done: bool = True
) -> dict:
    """Validate and run one job file.  Always writes a status file.

    Returns the status payload.  Never raises for a bad job -- rejection is a
    normal outcome and is recorded, so a caller polling status always gets an
    answer.
    """
    started = time.time()
    job_id = path.stem
    status: dict = {
        "job_id": job_id,
        "source_file": path.name,
        "state": "unknown",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    }

    try:
        parsed, stem = read_job_file(path)
        job = validate_job(parsed, stem)
    except JobRejected as exc:
        status.update(state="rejected", reason=str(exc), rendered=False)
        logger.warning("REJECTED %s: %s", path.name, exc)
        try:
            write_status(job_id if SLUG_RE.match(job_id) else "invalid_job_id", status)
        except JobRejected:
            # An id that is not a slug cannot name a status file; record it
            # under a fixed safe name instead of constructing a path from it.
            write_status("invalid_job_id", status)
        if move_done:
            try:
                move_to_done(path)
            except (OSError, JobRejected) as move_exc:
                logger.warning("could not move %s to done/: %s", path.name, move_exc)
        return status

    out_dir = assert_contained(RESULTS_DIR / job.job_id, AGENT_DIR)
    status.update(state="running", job=job.as_dict(), out_dir=str(out_dir))
    write_status(job.job_id, status)
    logger.info(
        "ACCEPTED %s: %d mode(s), %d mesh(es), %dx%d, camera=%s, label=%s",
        job.job_id, len(job.modes), job.meshes, job.size[0], job.size[1],
        job.camera, job.label,
    )

    try:
        manifest = render_fn(job, out_dir)
    except Exception as exc:  # noqa: BLE001 - a bad render must not kill the agent
        status.update(
            state="failed",
            reason=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=12),
            rendered=False,
            duration_s=round(time.time() - started, 3),
        )
        logger.error("FAILED %s: %s: %s", job.job_id, type(exc).__name__, exc)
        write_status(job.job_id, status)
        if move_done:
            move_to_done(path)
        return status

    files = sorted(p.name for p in Path(out_dir).glob("*.png")) if Path(out_dir).is_dir() else []
    status.update(
        state="done",
        rendered=True,
        duration_s=round(time.time() - started, 3),
        png_count=len(files),
        files=files,
        manifest=(manifest if isinstance(manifest, dict) else None),
    )
    write_status(job.job_id, status)
    logger.info(
        "DONE %s: %d PNG(s) in %.1fs -> %s",
        job.job_id, len(files), status["duration_s"], out_dir,
    )
    if move_done:
        move_to_done(path)
    return status


def pending_jobs() -> list[Path]:
    """Job files awaiting processing, oldest first.

    Only ``*.json`` directly in ``jobs/`` -- ``done/`` is a subdirectory and
    ``glob`` does not descend, so a processed job is never re-run.
    """
    if not JOBS_DIR.is_dir():
        return []
    files = [p for p in JOBS_DIR.glob("*.json") if p.is_file()]
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))


BANNER = """
================================================================================
FaceForge render agent
================================================================================
WILL DO
  * poll {jobs} every {poll:.0f}s for *.json job files
  * accept ONLY these keys: modes, meshes, size, camera, label
      modes   subset of the 16 RenderMode names, matched exactly
      meshes  integer, clamped to 1..{maxmesh} (index into a FIXED mesh list)
      size    "WxH", each clamped to {minsz}..{maxsz}
      camera  one of the named presets: {cams}
      label   slug [A-Za-z0-9_-]{{1,64}}
  * render with capture_golden.py and write PNGs + manifest.json to
      {results}/<job_id>/
  * write a status file to {status}/<job_id>.json
  * move each processed job file into {done}/

WILL NOT DO
  * read code, paths, filenames, shell strings or format strings from a job
  * eval, exec, import or subprocess anything a job asked for
  * write anywhere outside {agent}/
  * delete anything (processed jobs are MOVED to done/)
  * make any network call

A job with an unknown key, a wrong type or an out-of-range value is REJECTED
with a reason in its status file, and nothing is rendered.

Stop with Ctrl-C.
================================================================================
"""


def print_banner() -> None:
    min_size, max_size, max_meshes, _ = _limits()
    print(BANNER.format(
        jobs=JOBS_DIR, poll=POLL_SECONDS, results=RESULTS_DIR, status=STATUS_DIR,
        done=DONE_DIR, agent=AGENT_DIR, maxmesh=max_meshes,
        minsz=min_size, maxsz=max_size, cams=", ".join(_known_cameras()),
    ), flush=True)


class _Stopper:
    """Ctrl-C sets a flag; the loop finishes its current job and exits."""

    def __init__(self) -> None:
        self.stop = False

    def install(self) -> None:
        def handler(signum, _frame):
            if self.stop:
                logger.warning("second signal %s — exiting immediately", signum)
                raise SystemExit(130)
            self.stop = True
            logger.info(
                "signal %s received — finishing the current job then stopping "
                "(Ctrl-C again to exit now)", signum,
            )
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass   # not the main thread; --once path does not need it


def run(once: bool = False, render_fn=default_render_fn) -> int:
    """Poll and process until stopped.  Returns a process exit code."""
    ensure_dirs()
    print_banner()
    stopper = _Stopper()
    stopper.install()
    processed = 0
    try:
        with AgentLock():
            logger.info("agent ready (pid %d); waiting for jobs", os.getpid())
            idle_logged = False
            while not stopper.stop:
                jobs = pending_jobs()
                if not jobs:
                    if once:
                        if not processed:
                            logger.info("no pending jobs; --once so exiting")
                        break
                    if not idle_logged:
                        logger.debug("idle")
                        idle_logged = True
                    time.sleep(POLL_SECONDS)
                    continue
                idle_logged = False
                for p in jobs:
                    if stopper.stop:
                        break
                    process_job_file(p, render_fn=render_fn)
                    processed += 1
                if once and not pending_jobs():
                    break
    except AgentLocked as exc:
        logger.error("%s", exc)
        return 4
    logger.info("agent stopped after processing %d job(s)", processed)
    return 0


def self_check() -> int:
    """Verify wiring and the validation contract without rendering."""
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - self-check reports, never crashes
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")

    print("render_agent --self-check (no GL, no rendering)")

    def defaults_valid() -> None:
        j = validate_job({}, "job1")
        assert len(j.modes) == 16, f"default should be all 16 modes, got {len(j.modes)}"
        assert j.size == (512, 512) and j.camera == "oblique" and j.label == "job"
        assert j.meshes == 16
    check("an empty job takes documented defaults", defaults_valid)

    def unknown_key_rejected() -> None:
        for bad in ({"command": "ls"}, {"path": "/etc/passwd"}, {"modes": ["SOLID"], "x": 1}):
            try:
                validate_job(bad, "job1")
            except JobRejected:
                continue
            raise AssertionError(f"accepted unknown key: {bad}")
    check("unknown keys are rejected", unknown_key_rejected)

    def traversal_rejected() -> None:
        for bad in ("../../etc/passwd", "a/b", "a.b", ".hidden", "", "x" * 65, "a b"):
            try:
                validate_job({"label": bad}, "job1")
            except JobRejected:
                pass
            else:
                raise AssertionError(f"accepted label {bad!r}")
            try:
                validate_job_id(bad)
            except JobRejected:
                pass
            else:
                raise AssertionError(f"accepted job id {bad!r}")
    check("path traversal in label and job id is rejected", traversal_rejected)

    def containment_holds() -> None:
        assert_contained(RESULTS_DIR / "ok", AGENT_DIR)
        for bad in (Path("/etc/passwd"), AGENT_DIR / ".." / "src", Path("/tmp")):
            try:
                assert_contained(bad, AGENT_DIR)
            except JobRejected:
                continue
            raise AssertionError(f"assert_contained allowed {bad}")
    check("assert_contained blocks paths outside .render_agent/", containment_holds)

    def no_dangerous_calls() -> None:
        # Parsed, not grepped.  A substring scan cannot tell a call from a
        # string literal -- it flagged this very check's own token list -- and
        # it also misses `getattr(os, "system")`-shaped indirection that the
        # AST makes visible as a call to `getattr`.
        import ast

        forbidden_calls = {
            "eval", "exec", "compile", "__import__", "getattr", "setattr",
            "system", "popen", "spawn", "spawnl", "spawnv", "fork", "execv",
            "unlink", "remove", "rmtree", "rmdir", "removedirs",
            "loads_pickle", "load_pickle",
        }
        forbidden_imports = {
            "subprocess", "socket", "urllib", "urllib2", "urllib3", "requests",
            "http", "httplib", "ftplib", "telnetlib", "smtplib", "asyncio",
            "pickle", "shelve", "marshal", "importlib", "ctypes", "shutil",
        }
        # `re.compile` builds a regex, not code.  Allowlisted by its qualified
        # name so that a bare `compile(...)` is still caught.
        allowed_qualified = {"re.compile"}

        tree = ast.parse(Path(__file__).read_text())
        problems: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    name, qualified = fn.id, fn.id
                elif isinstance(fn, ast.Attribute):
                    base = fn.value.id if isinstance(fn.value, ast.Name) else "?"
                    name, qualified = fn.attr, f"{base}.{fn.attr}"
                else:
                    continue
                if name in forbidden_calls and qualified not in allowed_qualified:
                    problems.append(f"line {node.lineno}: call to {qualified}()")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    root = a.name.split(".")[0]
                    if root in forbidden_imports:
                        problems.append(f"line {node.lineno}: import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in forbidden_imports:
                    problems.append(f"line {node.lineno}: from {node.module} import ...")
        assert not problems, "forbidden constructs: " + "; ".join(problems)
    check("AST shows no eval/exec/subprocess/network/delete constructs", no_dangerous_calls)

    def clamps_hold() -> None:
        min_size, max_size, max_meshes, _ = _limits()
        for bad in ({"size": f"{max_size + 1}x64"}, {"size": f"64x{min_size - 1}"},
                    {"size": "99999x99999"}, {"meshes": 0}, {"meshes": max_meshes + 1},
                    {"meshes": -1}):
            try:
                validate_job(bad, "job1")
            except JobRejected:
                continue
            raise AssertionError(f"accepted out-of-range {bad}")
        assert validate_job({"size": f"{max_size}x{min_size}"}, "job1").size == (max_size, min_size)
    check("size and mesh clamps hold at the boundaries", clamps_hold)

    def capture_contract_matches() -> None:
        import inspect

        from tools.capture_golden import capture
        params = set(inspect.signature(capture).parameters)
        for needed in ("modes", "mesh_count", "size", "camera_name", "label", "force", "prefer"):
            assert needed in params, f"capture() has no {needed} parameter"
    check("capture_golden.capture() accepts the parameters the agent passes", capture_contract_matches)

    def dirs_creatable() -> None:
        ensure_dirs()
        for d in (JOBS_DIR, DONE_DIR, RESULTS_DIR, STATUS_DIR):
            assert d.is_dir(), f"{d} was not created"
    check("agent directories can be created", dirs_creatable)

    print()
    if failures:
        print(f"SELF-CHECK FAILED: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SELF-CHECK PASSED. Start the agent with:  python -m tools.render_agent")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="render_agent",
        description="Poll .render_agent/jobs/ and render validated golden-image jobs.",
    )
    p.add_argument("--once", action="store_true",
                   help="process pending jobs and exit instead of polling")
    p.add_argument("--self-check", action="store_true",
                   help="verify wiring and the validation contract, then exit")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    if args.self_check:
        return self_check()
    return run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
