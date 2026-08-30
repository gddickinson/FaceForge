"""Progress persistence: location, versioning, atomicity, round-trip.

Nothing here writes to the real user data directory: every test either sets
``FACEFORGE_DATA_DIR`` via monkeypatch or passes an explicit ``path``.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from faceforge.anatomy import quiz_progress as qp
from faceforge.anatomy.quiz_progress import (
    SCHEMA_VERSION,
    Attempt,
    ProgressStore,
    progress_path,
    user_data_dir,
)

T0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def clock():
    state = {"now": T0}
    now = lambda: state["now"]                                  # noqa: E731
    now.advance = lambda **kw: state.__setitem__("now", state["now"] + timedelta(**kw))
    return now


@pytest.fixture
def store(tmp_path, clock):
    return ProgressStore(user="tester", path=tmp_path / "tester.json", clock=clock)


# ── location ─────────────────────────────────────────────────────────────

def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("FACEFORGE_DATA_DIR", str(tmp_path / "elsewhere"))
    assert user_data_dir() == tmp_path / "elsewhere"
    assert progress_path("bob") == tmp_path / "elsewhere" / "progress" / "bob.json"


def test_default_location_is_outside_the_repository(monkeypatch):
    monkeypatch.delenv("FACEFORGE_DATA_DIR", raising=False)
    repo_root = Path(__file__).resolve().parents[2]
    assert repo_root not in user_data_dir().resolve().parents


@pytest.mark.parametrize("platform,expected_tail", [
    ("darwin", ("Library", "Application Support", "FaceForge")),
    ("win32", ("FaceForge",)),
    ("linux", ("faceforge",)),
])
def test_platform_conventions(monkeypatch, platform, expected_tail):
    monkeypatch.delenv("FACEFORGE_DATA_DIR", raising=False)
    monkeypatch.setattr(qp.sys, "platform", platform)
    parts = user_data_dir().parts
    assert parts[-len(expected_tail):] == expected_tail


def test_user_name_cannot_escape_the_progress_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("FACEFORGE_DATA_DIR", str(tmp_path))
    path = progress_path("../../etc/passwd")
    assert path.parent == tmp_path / "progress"
    assert path.name == "etc_passwd.json"


def test_empty_user_name_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("FACEFORGE_DATA_DIR", str(tmp_path))
    assert progress_path("...").name == "default.json"


# ── round trip ───────────────────────────────────────────────────────────

def test_missing_file_is_not_an_error(store):
    assert store.load() is False
    assert store.attempts == []


def test_record_save_load_round_trip(tmp_path, clock, store):
    store.record("FMA52734", "Frontal Bone", True, 5, given_answer="Frontal bone",
                 curriculum="skull_bones", tier="foundation", elapsed_s=2.5)
    clock.advance(seconds=30)
    store.record("FMA52735", "Occipital Bone", False, 1, given_answer="wrong")
    path = store.save()
    assert path.exists()

    reloaded = ProgressStore(user="tester", path=path, clock=clock)
    assert reloaded.load() is True
    assert [a.item_id for a in reloaded.attempts] == ["FMA52734", "FMA52735"]
    assert reloaded.attempts[0] == store.attempts[0]
    assert reloaded.scheduler.cards.keys() == store.scheduler.cards.keys()
    assert reloaded.scheduler.card("FMA52734").interval_days == 1


def test_file_carries_a_schema_version(store):
    store.record("a", "A", True, 5)
    payload = json.loads(store.save().read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["user"] == "tester"
    assert payload["created"] == T0.isoformat()


def test_save_creates_missing_parent_directories(tmp_path, clock):
    target = tmp_path / "deep" / "nested" / "p.json"
    s = ProgressStore(path=target, clock=clock)
    s.record("a", "A", True, 5)
    assert s.save().exists()


def test_save_leaves_no_temporary_files_behind(tmp_path, clock, store):
    store.record("a", "A", True, 5)
    store.save()
    store.save()
    assert sorted(p.name for p in store.path.parent.iterdir()) == ["tester.json"]


def test_created_timestamp_survives_a_resave(tmp_path, clock, store):
    store.record("a", "A", True, 5)
    store.save()
    clock.advance(days=3)
    store.record("b", "B", True, 5)
    payload = json.loads(store.save().read_text())
    assert payload["created"] == T0.isoformat()
    assert payload["updated"] != payload["created"]


# ── version and corruption handling ──────────────────────────────────────

def test_future_schema_is_refused_rather_than_misread(tmp_path, clock, caplog):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 99,
                                "attempts": [{"item_id": "a"}]}))
    s = ProgressStore(path=path, clock=clock)
    assert s.load() is False
    assert s.attempts == []


def test_file_without_a_version_is_refused(tmp_path, clock):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"attempts": [{"item_id": "a"}]}))
    assert ProgressStore(path=path, clock=clock).load() is False


def test_corrupt_json_does_not_raise(tmp_path, clock):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    s = ProgressStore(path=path, clock=clock)
    assert s.load() is False
    assert s.attempts == []


def test_attempt_rows_missing_required_fields_are_dropped(tmp_path, clock):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "attempts": [
            {"item_id": "a", "display_name": "A", "timestamp": T0.isoformat(),
             "correct": True, "grade": 5},
            {"display_name": "no id"},
            "not a dict",
        ],
        "cards": {},
    }))
    s = ProgressStore(path=path, clock=clock)
    assert s.load() is True
    assert [a.item_id for a in s.attempts] == ["a"]


def test_unknown_attempt_fields_from_a_newer_build_are_ignored():
    a = Attempt.from_dict({
        "item_id": "a", "display_name": "A", "timestamp": T0.isoformat(),
        "correct": True, "grade": 5, "some_future_field": 1,
    })
    assert a.item_id == "a"


# ── queries ──────────────────────────────────────────────────────────────

def test_summary_counts_and_accuracy(store, clock):
    store.record("a", "A", True, 5)
    store.record("b", "B", False, 1)
    store.record("a", "A", True, 5)
    summary = store.summary()
    assert (summary.attempts, summary.correct, summary.items_seen) == (3, 2, 2)
    assert summary.accuracy == pytest.approx(2 / 3)
    assert summary.first_attempt == T0.isoformat()


def test_items_due_uses_the_scheduler_and_the_clock(store, clock):
    store.record("a", "A", True, 5)          # due in 1 day
    assert store.summary().items_due == 0
    clock.advance(days=2)
    assert store.summary().items_due == 1


def test_accuracy_and_weakest_items(store):
    store.record("good", "G", True, 5)
    store.record("bad", "B", False, 1)
    store.record("bad", "B", False, 1)
    store.record("mixed", "M", True, 5)
    store.record("mixed", "M", False, 1)
    assert store.accuracy_for("bad") == 0.0
    assert store.accuracy_for("never seen") is None
    assert store.weakest_items(2) == [("bad", 0.0), ("mixed", 0.5)]


def test_attempts_for_filters_by_item(store):
    store.record("a", "A", True, 5)
    store.record("b", "B", True, 5)
    assert len(store.attempts_for("a")) == 1
