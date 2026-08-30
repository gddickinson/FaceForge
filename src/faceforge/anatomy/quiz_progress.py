"""Per-user quiz progress: attempt history plus SM-2 card state, on disk.

Where the file lives
--------------------
Not in the repository.  A learner's history is user data, and writing it into
``assets/`` or the source tree would put it in ``git status``, lose it on a
reinstall, and break a read-only or shared install.  The directory follows the
platformdirs convention for a user *data* dir (not cache -- this is not
regenerable):

===========  ===================================================
macOS        ``~/Library/Application Support/FaceForge``
Windows      ``%LOCALAPPDATA%\\FaceForge``
other        ``$XDG_DATA_HOME/faceforge`` or ``~/.local/share/faceforge``
===========  ===================================================

``platformdirs`` itself is not a dependency of this project, and one module
that needs three ``sys.platform`` branches is not a reason to add a runtime
dependency to a GUI application, so the convention is implemented here in
:func:`user_data_dir`.  ``FACEFORGE_DATA_DIR`` overrides it -- that is what the
tests use, and it is also the escape hatch for a portable install.

File format
-----------
One JSON object per user at ``<data dir>/progress/<user>.json``, carrying
``schema_version``.  Versioning is the whole point of writing a header: this
file outlives the build that wrote it, and a future field rename has to be
able to tell a v1 file from a v2 one instead of guessing from which keys are
present.  :meth:`ProgressStore.load` migrates forward through
:data:`_MIGRATIONS` and refuses (loudly, returning empty state) a version from
the future rather than silently discarding fields it does not understand.

Writes are atomic: a temporary file in the same directory followed by
``os.replace``.  A quiz is interrupted by closing a window, and a half-written
JSON file would lose the entire history rather than the last answer.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from faceforge.anatomy.spaced_repetition import Scheduler, utc_clock

logger = logging.getLogger(__name__)

#: Bump when the on-disk shape changes, and add a migration below.
SCHEMA_VERSION = 1

APP_NAME = "FaceForge"

_SAFE_USER = re.compile(r"[^A-Za-z0-9._-]+")


def user_data_dir(app_name: str = APP_NAME) -> Path:
    """Per-user data directory, platformdirs convention.

    ``FACEFORGE_DATA_DIR`` takes precedence when set and non-empty.
    """
    override = os.environ.get("FACEFORGE_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / app_name
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    root = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return root / app_name.lower()


def progress_path(user: str = "default", app_name: str = APP_NAME) -> Path:
    """Path of one user's progress file.

    The user name is slugified because it reaches the filesystem: a learner
    called ``../../etc`` must not be able to choose where the file lands.
    """
    slug = _SAFE_USER.sub("_", user).strip("._-") or "default"
    return user_data_dir(app_name) / "progress" / f"{slug}.json"


@dataclass(frozen=True)
class Attempt:
    """One answered (or skipped) question.

    ``item_id`` is the stable structure id -- the BodyParts3D mesh id where
    known -- and ``display_name`` is what the learner was shown, kept so a
    history stays readable after a config rewording.
    """

    item_id: str
    display_name: str
    timestamp: str            # ISO-8601, UTC, timezone-aware
    correct: bool
    grade: int                # SM-2 0..5, see spaced_repetition.grade_from_outcome
    given_answer: str = ""
    mode: str = "identify"
    curriculum: str = ""
    tier: str = ""
    elapsed_s: float = 0.0
    skipped: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "Attempt":
        known = {f for f in cls.__dataclass_fields__}          # noqa: F821
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ProgressSummary:
    """Aggregate view of a history, for the results page and reporting."""

    attempts: int = 0
    correct: int = 0
    items_seen: int = 0
    items_due: int = 0
    first_attempt: Optional[str] = None
    last_attempt: Optional[str] = None

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempts if self.attempts else 0.0


@dataclass(frozen=True)
class ItemStats:
    """Raw per-item counters, for empirical calibration *later*.

    These are counts and latencies, nothing derived.  A difficulty index needs
    a cohort: with one learner and single-digit attempt counts, a "facility"
    of 0.5 from two attempts is indistinguishable from noise, and a
    discrimination index needs a distribution of total scores across
    candidates that a single-user file does not contain.  So the store keeps
    the raw material and says so; computing p-values or point-biserial
    correlations from this and presenting them as item statistics would be
    misleading.  Export a cohort's files and calibrate outside.
    """

    item_id: str
    times_seen: int = 0
    times_correct: int = 0
    times_skipped: int = 0
    mean_latency_s: float = 0.0
    min_latency_s: float = 0.0
    max_latency_s: float = 0.0
    last_grade: Optional[int] = None
    interval_days: int = 0
    easiness: float = 0.0
    lapses: int = 0

    @property
    def proportion_correct(self) -> Optional[float]:
        """Times correct / times seen, or None when never seen.

        Not a difficulty index: see the class docstring.
        """
        if not self.times_seen:
            return None
        return self.times_correct / self.times_seen


class ProgressStore:
    """Loads, mutates and saves one user's quiz history.

    Parameters
    ----------
    user:
        Profile name; selects the file. Slugified for the filesystem.
    path:
        Explicit file path, overriding ``user``/data-dir resolution.
    clock:
        Zero-argument callable returning a timezone-aware ``datetime``.  The
        scheduler shares it, so a test fixes time once and both the attempt
        timestamps and the SM-2 due dates follow.
    """

    def __init__(
        self,
        user: str = "default",
        path: Optional[Path] = None,
        clock: Callable[[], datetime] = utc_clock,
    ):
        self.user = user
        self.path = Path(path) if path is not None else progress_path(user)
        self.clock = clock
        self.attempts: list[Attempt] = []
        self.scheduler = Scheduler(clock=clock)
        self._created: Optional[str] = None
        self._dirty = False

    # -- persistence -------------------------------------------------------

    def load(self) -> bool:
        """Read the file if present.  Returns True when state was loaded.

        A missing file is normal (first run) and not an error.  An unreadable
        or future-versioned file logs and leaves the store empty: losing a
        history is bad, but refusing to start a quiz is worse.
        """
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        except (OSError, ValueError):
            logger.exception("Quiz progress at %s is unreadable; starting empty",
                             self.path)
            return False

        if not isinstance(raw, dict):
            logger.error("Quiz progress at %s is not a JSON object", self.path)
            return False

        version = raw.get("schema_version")
        if not isinstance(version, int):
            logger.error("Quiz progress at %s has no schema_version; ignoring",
                         self.path)
            return False
        if version > SCHEMA_VERSION:
            logger.error(
                "Quiz progress at %s is schema v%d but this build understands "
                "v%d; not loading (and not overwriting until you answer a "
                "question)", self.path, version, SCHEMA_VERSION)
            return False
        while version < SCHEMA_VERSION:
            migrate = _MIGRATIONS.get(version)
            if migrate is None:
                logger.error("No migration from quiz-progress schema v%d", version)
                return False
            raw = migrate(raw)
            version += 1

        self._created = raw.get("created")
        self.attempts = [
            Attempt.from_dict(a) for a in raw.get("attempts", [])
            if isinstance(a, dict) and "item_id" in a
        ]
        self.scheduler.load_dict(raw.get("cards", {}))
        return True

    def save(self) -> Path:
        """Write the file atomically.  Creates parent directories."""
        now = self.clock().isoformat()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "app": APP_NAME,
            "user": self.user,
            "created": self._created or now,
            "updated": now,
            "attempts": [asdict(a) for a in self.attempts],
            "cards": self.scheduler.to_dict(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Same directory as the target: os.replace is only atomic within a
        # filesystem, and /tmp is frequently a different one.
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent),
                                   prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            # Leave no half-written temp file behind on failure.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._created = payload["created"]
        self._dirty = False
        return self.path

    # -- mutation ----------------------------------------------------------

    def record(
        self,
        item_id: str,
        display_name: str,
        correct: bool,
        grade: int,
        *,
        given_answer: str = "",
        mode: str = "identify",
        curriculum: str = "",
        tier: str = "",
        elapsed_s: float = 0.0,
        skipped: bool = False,
    ) -> Attempt:
        """Append an attempt and advance that item's SM-2 card."""
        attempt = Attempt(
            item_id=item_id,
            display_name=display_name,
            timestamp=self.clock().isoformat(),
            correct=correct,
            grade=int(grade),
            given_answer=given_answer,
            mode=mode,
            curriculum=curriculum,
            tier=tier,
            elapsed_s=float(elapsed_s),
            skipped=skipped,
        )
        self.attempts.append(attempt)
        self.scheduler.record(item_id, grade)
        self._dirty = True
        return attempt

    # -- queries -----------------------------------------------------------

    def attempts_for(self, item_id: str) -> list[Attempt]:
        return [a for a in self.attempts if a.item_id == item_id]

    def accuracy_for(self, item_id: str) -> Optional[float]:
        seen = self.attempts_for(item_id)
        if not seen:
            return None
        return sum(1 for a in seen if a.correct) / len(seen)

    def weakest_items(self, limit: int = 10) -> list[tuple[str, float]]:
        """Items with the worst accuracy, ties broken by most attempts.

        Only items answered at least once appear.  Deterministic ordering.
        """
        by_item: dict[str, list[Attempt]] = {}
        for a in self.attempts:
            by_item.setdefault(a.item_id, []).append(a)
        scored = [
            (item, sum(1 for a in rows if a.correct) / len(rows), len(rows))
            for item, rows in by_item.items()
        ]
        scored.sort(key=lambda t: (t[1], -t[2], t[0]))
        return [(item, acc) for item, acc, _ in scored[:limit]]

    def item_stats(self, item_id: str) -> "ItemStats":
        """Raw per-item counters for one structure.

        Deliberately raw.  Times seen, times correct and mean latency are what
        the history actually contains; a difficulty or discrimination index
        computed from one learner's handful of attempts would look like a
        calibrated psychometric statistic and would not be one.  See
        :class:`ItemStats`.
        """
        rows = self.attempts_for(item_id)
        latencies = [a.elapsed_s for a in rows if a.elapsed_s > 0]
        card = self.scheduler.card(item_id)
        return ItemStats(
            item_id=item_id,
            times_seen=len(rows),
            times_correct=sum(1 for a in rows if a.correct),
            times_skipped=sum(1 for a in rows if a.skipped),
            mean_latency_s=(sum(latencies) / len(latencies)) if latencies else 0.0,
            min_latency_s=min(latencies) if latencies else 0.0,
            max_latency_s=max(latencies) if latencies else 0.0,
            last_grade=card.last_grade,
            interval_days=card.interval_days,
            easiness=card.easiness,
            lapses=card.lapses,
        )

    def all_item_stats(self) -> list["ItemStats"]:
        """Per-item counters for every structure with a history, id-ordered."""
        return [self.item_stats(i) for i in sorted({a.item_id for a in self.attempts})]

    def summary(self) -> ProgressSummary:
        stamps = [a.timestamp for a in self.attempts]
        seen = {a.item_id for a in self.attempts}
        now = self.clock()
        return ProgressSummary(
            attempts=len(self.attempts),
            correct=sum(1 for a in self.attempts if a.correct),
            items_seen=len(seen),
            items_due=sum(
                1 for item in seen if self.scheduler.card(item).is_due(now)
            ),
            first_attempt=min(stamps) if stamps else None,
            last_attempt=max(stamps) if stamps else None,
        )


# -- migrations ------------------------------------------------------------
#
# Each entry takes a payload at version N and returns one at N+1.  There are
# none yet (v1 is the first shipped shape); the dict exists so the loader has
# somewhere to look, and so the next schema change is a one-function edit
# rather than a rewrite of load().
_MIGRATIONS: dict[int, Callable[[dict], dict]] = {}
