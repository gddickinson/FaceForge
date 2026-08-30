"""Exam sessions: assembly, the four formats, timing, and recording.

The station clock is driven by a fixed injected clock, so "the candidate ran
out of time" is a deterministic assertion rather than a sleep.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from faceforge.anatomy.answer_explanations import ExplanationBuilder
from faceforge.anatomy.curricula import build_curricula
from faceforge.anatomy.exam_items import ExamItem, Option, Provenance
from faceforge.anatomy.exam_session import (
    DEFAULT_STATION_SECONDS,
    ExamConfig,
    ExamSession,
)
from faceforge.anatomy.fma_taxonomy import SCHEMA_VERSION, Taxonomy
from faceforge.anatomy.item_generators import ItemGenerator
from faceforge.anatomy.quiz_progress import ProgressStore

T0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

PAYLOAD = {
    "schema_version": SCHEMA_VERSION,
    "_source": "synthetic",
    "nodes": {
        "1": {"label": "Bone organ", "parent": ""},
        "2": {"label": "Flat bone", "parent": "1"},
        "3": {"label": "Irregular bone", "parent": "1"},
        "4": {"label": "Frontal bone", "parent": "2"},
        "5": {"label": "Parietal bone", "parent": "2"},
        "6": {"label": "Mandible", "parent": "3"},
        "7": {"label": "Occipital bone", "parent": "2"},
        "8": {"label": "Liver", "parent": "12"},
        "9": {"label": "Right temporalis", "parent": "10"},
        "10": {"label": "Temporalis", "parent": "11"},
        "11": {"label": "Muscle of head", "parent": "1"},
        "12": {"label": "Lobular organ", "parent": "1"},
    },
    "labels": {"FMA100": "neurocranium", "FMA101": "skull",
               "FMA102": "abdomen", "FMA103": "musculature of head"},
    "part_of": {"FMA4": ["FMA100"], "FMA5": ["FMA100"], "FMA7": ["FMA100"],
                "FMA6": ["FMA101"], "FMA8": ["FMA102"], "FMA9": ["FMA103"]},
    "composite_of": {},
}
FMA = {
    f"FMA{n}": {"display_name": name, "preferred_label": label,
                "system": "skeletal", "category": "skull_bones"}
    for n, name, label in (
        (4, "Frontal Bone", "Frontal bone"),
        (5, "Parietal Bone", "Parietal bone"),
        (6, "Mandible", "Mandible"),
        (7, "Occipital Bone", "Occipital bone"),
    )
}
FMA["FMA8"] = {"display_name": "Liver", "preferred_label": "Liver",
               "system": "digestive", "category": "organs"}
FMA["FMA9"] = {"display_name": "Temporalis R",
               "preferred_label": "Right temporalis",
               "system": "muscular", "category": "muscles"}


@pytest.fixture
def clock():
    state = {"now": T0}
    now = lambda: state["now"]                                  # noqa: E731
    now.advance = lambda **kw: state.__setitem__("now", state["now"] + timedelta(**kw))
    return now


@pytest.fixture
def curricula(tmp_path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "skull_bones.json").write_text(json.dumps([
        {"name": name, "stl": mesh_id}
        for mesh_id, name in (("FMA4", "Frontal Bone"), ("FMA5", "Parietal Bone"),
                              ("FMA6", "Mandible"), ("FMA7", "Occipital Bone"),
                              ("FMA8", "Liver"), ("FMA9", "Temporalis R"))
    ]))
    return build_curricula(fma=FMA, config_dir=cfg)


@pytest.fixture
def store(tmp_path, clock):
    return ProgressStore(user="t", path=tmp_path / "t.json", clock=clock)


@pytest.fixture
def session(curricula, store, clock):
    return ExamSession(
        progress=store, curricula=curricula, clock=clock,
        generator=ItemGenerator(fma=FMA, taxonomy=Taxonomy(payload=PAYLOAD)),
        explanations=ExplanationBuilder(fma=FMA),
    )


CONFIG = ExamConfig(levels=("L1", ), curriculum="skull_bones", count=3,
                    options=3, seed=1)


# ── assembly ─────────────────────────────────────────────────────────────

def test_builds_the_requested_number_of_items(session):
    items = session.build(CONFIG)
    assert len(items) == 3
    assert {i.level for i in items} == {"L1"}
    assert all(len(i.options) == 3 for i in items)


def test_unknown_curriculum_builds_nothing(session):
    assert session.build(ExamConfig(curriculum="nope")) == []


def test_explicit_focus_ids_override_the_curriculum(session):
    items = session.build(ExamConfig(levels=("L1", ), focus_ids=("FMA6", ),
                                     count=5, options=3))
    assert [i.focus_id for i in items] == ["FMA6"]


def test_levels_are_tried_in_order_until_the_count_is_met(session):
    items = session.build(ExamConfig(levels=("L3", "L1"),
                                     curriculum="skull_bones", count=20,
                                     options=3))
    levels = [i.level for i in items]
    assert "L3" in levels and "L1" in levels
    assert levels.index("L3") < levels.index("L1")


def test_items_lacking_provenance_are_refused_not_asked(session, monkeypatch):
    bad = ExamItem(level="L1", fmt="sba", stem="Unsourced?",
                   options=(Option(text="A"), Option(text="B")),
                   answer_index=0)
    monkeypatch.setattr(session.generator, "generate",
                        lambda *a, **kw: [bad])
    assert session.build(CONFIG) == []
    assert session.refused and "no provenance" in session.refused[0][1]


def test_audit_lists_every_item_and_every_refusal(session):
    session.build(CONFIG)
    report = session.audit()
    assert "fma_label" in report
    assert report.startswith("Exam: 3 item(s)")


# ── running an SBA paper ─────────────────────────────────────────────────

def test_answering_correctly_scores_and_grades(session, clock):
    item = session.start(CONFIG)
    clock.advance(seconds=2)
    outcome = session.answer(item.answer_index)
    assert outcome.correct is True
    assert outcome.grade == 5
    assert outcome.elapsed_s == pytest.approx(2.0)
    assert session.score == (1, 1)


def test_a_wrong_answer_gets_a_data_derived_explanation(session, clock):
    item = session.start(CONFIG)
    wrong = 1 if item.answer_index == 0 else 0
    outcome = session.answer(wrong)
    assert outcome.correct is False
    assert outcome.explanation
    assert item.options[wrong].text in outcome.explanation
    assert item.answer.text in outcome.explanation


def test_a_wrong_answer_in_the_same_system_grades_two(session):
    session.build(ExamConfig(levels=("L1", ), focus_ids=("FMA4", ), count=1,
                             options=3, seed=1))
    item = session.start()
    wrong = 1 if item.answer_index == 0 else 0
    outcome = session.answer(wrong)
    # The frontal bone's distractors are its is-a siblings, all skeletal, so a
    # wrong option is a same-system confusion: SM-2 grade 2, not 1.
    assert outcome.grade == 2


def test_skipping_grades_zero(session):
    session.start(CONFIG)
    outcome = session.answer(None)
    assert (outcome.skipped, outcome.correct, outcome.grade) == (True, False, 0)


def test_an_out_of_range_index_is_a_skip_not_a_crash(session):
    session.start(CONFIG)
    assert session.answer(99).skipped is True


def test_progress_through_the_paper(session):
    session.start(CONFIG)
    assert session.remaining == 2
    session.answer(0)
    session.next_item()
    assert session.remaining == 1
    session.next_item()
    session.next_item()
    assert session.current is None
    assert session.finished is True


def test_there_is_no_way_back(session):
    assert not hasattr(session, "previous")
    assert not hasattr(session, "back")


# ── station mode ─────────────────────────────────────────────────────────

STATION = ExamConfig(levels=("L1", ), curriculum="skull_bones", count=2,
                     options=3, fmt="station", seed=1)


def test_station_mode_applies_a_default_time_limit(session):
    items = session.build(STATION)
    assert all(i.seconds == DEFAULT_STATION_SECONDS for i in items)


def test_station_time_limit_is_configurable(session):
    items = session.build(ExamConfig(levels=("L1", ),
                                     curriculum="skull_bones", count=2,
                                     options=3, fmt="station",
                                     seconds_per_item=30.0, seed=1))
    assert all(i.seconds == 30.0 for i in items)


def test_time_left_counts_down_on_the_injected_clock(session, clock):
    session.start(STATION)
    assert session.time_left() == pytest.approx(DEFAULT_STATION_SECONDS)
    clock.advance(seconds=20)
    assert session.time_left() == pytest.approx(DEFAULT_STATION_SECONDS - 20)
    assert session.expired() is False


def test_an_answer_after_the_station_clock_expires_is_a_skip(session, clock):
    item = session.start(STATION)
    clock.advance(seconds=DEFAULT_STATION_SECONDS + 1)
    assert session.expired() is True
    outcome = session.answer(item.answer_index)
    assert (outcome.expired, outcome.skipped, outcome.correct) == (True, True, False)
    assert outcome.grade == 0


def test_the_clock_restarts_on_the_next_station(session, clock):
    session.start(STATION)
    clock.advance(seconds=30)
    session.answer(0)
    session.next_item()
    assert session.time_left() == pytest.approx(DEFAULT_STATION_SECONDS)


def test_untimed_items_report_no_time_left(session):
    session.start(CONFIG)
    assert session.time_left() is None
    assert session.expired() is False


# ── extended matching ────────────────────────────────────────────────────

def test_emq_session_shares_one_option_list(session):
    items = session.build(ExamConfig(levels=("L3", ),
                                     curriculum="skull_bones", count=5,
                                     fmt="emq", seed=1))
    assert len(items) >= 2
    assert all(i.options == items[0].options for i in items)
    assert {i.fmt for i in items} == {"emq"}


# ── spot format ──────────────────────────────────────────────────────────

def test_spot_items_keep_their_format_inside_a_station(session):
    items = session.build(ExamConfig(levels=("L2", ),
                                     curriculum="skull_bones", count=4,
                                     options=3, fmt="station", seed=1))
    # L2 laterality items are spot items; reframing must not turn a tagged
    # stimulus into a written stem.
    for item in items:
        if "laterality" in item.tags:
            assert item.fmt == "spot"
            assert item.seconds == DEFAULT_STATION_SECONDS


# ── recording ────────────────────────────────────────────────────────────

def test_outcomes_are_recorded_against_the_structure_id(session, store, clock):
    item = session.start(CONFIG)
    clock.advance(seconds=4)
    session.answer(item.answer_index)
    attempt = store.attempts[-1]
    assert attempt.item_id == item.focus_id
    assert attempt.mode == f"{item.level}/{item.fmt}"
    assert attempt.curriculum == "skull_bones"
    assert attempt.elapsed_s == pytest.approx(4.0)


def test_finish_persists_the_file(session, store):
    session.start(CONFIG)
    session.answer(0)
    assert session.finish() == session.score
    assert store.path.exists()
    assert json.loads(store.path.read_text())["attempts"]


def test_a_session_without_a_store_still_runs(curricula, clock):
    session = ExamSession(
        progress=None, curricula=curricula, clock=clock,
        generator=ItemGenerator(fma=FMA, taxonomy=Taxonomy(payload=PAYLOAD)),
        explanations=ExplanationBuilder(fma=FMA))
    item = session.start(CONFIG)
    assert item is not None
    assert session.answer(item.answer_index).correct is True
    assert session.finish() == (1, 1)


def test_scheduling_orders_the_focus_structures(session, store, clock):
    """A lapsed structure is due tomorrow; a twice-passed one is due in six
    days.  Two days later the exam should be built over the lapsed one."""
    for _ in range(2):
        for mesh_id in ("FMA4", "FMA5", "FMA6"):
            clock.advance(seconds=1)
            store.record(mesh_id, mesh_id, True, 5)
    store.record("FMA7", "Occipital Bone", False, 1)
    clock.advance(days=2)
    assert session.focus_order(CONFIG)[0] == "FMA7"


# ── per-item statistics ──────────────────────────────────────────────────

def test_item_stats_are_raw_counts(session, store, clock):
    item = session.start(CONFIG)
    clock.advance(seconds=3)
    session.answer(item.answer_index)
    stats = store.item_stats(item.focus_id)
    assert (stats.times_seen, stats.times_correct) == (1, 0 + 1)
    assert stats.mean_latency_s == pytest.approx(3.0)
    assert stats.proportion_correct == 1.0
    assert stats.last_grade == 5


def test_item_stats_for_an_unseen_structure_are_empty(store):
    stats = store.item_stats("FMA999")
    assert stats.times_seen == 0
    assert stats.proportion_correct is None


def test_all_item_stats_is_id_ordered(session, store, clock):
    for _ in range(2):
        item = session.start(CONFIG) if session.current is None else session.current
        session.answer(0)
        session.next_item()
    ids = [s.item_id for s in store.all_item_stats()]
    assert ids == sorted(ids)
