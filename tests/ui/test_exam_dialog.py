"""The exam dialog, driven headlessly.

Visibility is asserted with ``isHidden()`` rather than ``isVisible()``: the
dialog is never shown, and Qt reports every descendant of an unshown widget as
not visible whatever its own flag says.  ``isHidden()`` is the widget's own
setting, which is what these tests are about.

``QT_QPA_PLATFORM=offscreen`` (set below before any Qt import) gives real
widgets with no window server, so every control is clickable.  The dialog is
built over a synthetic session -- fixed clock, temp progress file, synthetic
FMA graph -- so nothing here touches the user's data directory or the 1.2 GB
mesh dataset.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                             # noqa: E402
from PySide6.QtWidgets import QApplication                     # noqa: E402

from faceforge.anatomy.answer_explanations import ExplanationBuilder  # noqa: E402
from faceforge.anatomy.curricula import build_curricula        # noqa: E402
from faceforge.anatomy.exam_session import ExamSession         # noqa: E402
from faceforge.anatomy.fma_taxonomy import SCHEMA_VERSION, Taxonomy  # noqa: E402
from faceforge.anatomy.item_generators import ItemGenerator    # noqa: E402
from faceforge.anatomy.quiz_progress import ProgressStore      # noqa: E402
from faceforge.ui.exam_dialog import ExamDialog, _scan_pixmap  # noqa: E402

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
    },
    "labels": {"FMA100": "neurocranium"},
    "part_of": {"FMA4": ["FMA100"], "FMA5": ["FMA100"], "FMA7": ["FMA100"]},
    "composite_of": {},
}
FMA = {
    f"FMA{n}": {"display_name": display, "preferred_label": label,
                "system": "skeletal", "category": "skull_bones"}
    for n, display, label in (
        (4, "Frontal Bone", "Frontal bone"),
        (5, "Parietal Bone", "Parietal bone"),
        (6, "Mandible", "Mandible"),
        (7, "Occipital Bone", "Occipital bone"),
    )
}


@pytest.fixture(scope="module")
def qapp():
    """A QApplication that is destroyed again on teardown.

    Load-bearing: ``faceforge.app.main()`` calls ``QApplication(sys.argv)``
    unconditionally, and shiboken refuses to build a second instance while one
    is alive ("Please destroy the QApplication singleton before creating a new
    QApplication instance").  A session-scoped app left running here therefore
    breaks tests/ui/test_gui_smoke.py, which runs the real ``main()``.  So this
    fixture creates an app only if none exists and shuts down the one it
    created, leaving a pre-existing app alone.
    """
    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    app = QApplication([])
    try:
        yield app
    finally:
        app.processEvents()
        app.shutdown()


@pytest.fixture
def clock():
    state = {"now": T0}
    now = lambda: state["now"]                                  # noqa: E731
    now.advance = lambda **kw: state.__setitem__(
        "now", state["now"] + timedelta(**kw))
    return now


@pytest.fixture
def dialog(qapp, tmp_path, clock):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "skull_bones.json").write_text(json.dumps([
        {"name": display, "stl": mesh_id}
        for mesh_id, display in (("FMA4", "Frontal Bone"),
                                 ("FMA5", "Parietal Bone"),
                                 ("FMA6", "Mandible"),
                                 ("FMA7", "Occipital Bone"))
    ]))
    session = ExamSession(
        progress=ProgressStore(user="t", path=tmp_path / "t.json", clock=clock),
        curricula=build_curricula(fma=FMA, config_dir=cfg),
        generator=ItemGenerator(fma=FMA, taxonomy=Taxonomy(payload=PAYLOAD)),
        explanations=ExplanationBuilder(fma=FMA),
        clock=clock,
    )
    widget = ExamDialog(session=session)
    yield widget
    widget.deleteLater()


# ── setup page ───────────────────────────────────────────────────────────

def test_dialog_constructs_with_the_setup_page_showing(dialog):
    assert dialog._pages.currentIndex() == 0


def test_levels_come_from_the_schema_not_a_hardcoded_list(dialog):
    from faceforge.anatomy.exam_items import LEVEL_TITLES
    assert set(dialog._level_boxes) == set(LEVEL_TITLES)


def test_each_level_tooltip_names_its_data_source(dialog):
    for box in dialog._level_boxes.values():
        assert "Source:" in box.toolTip()


def test_l4_is_disabled_until_a_scene_is_supplied(dialog):
    box = dialog._level_boxes["L4"]
    assert box.isEnabled() is False
    assert "requires a loaded scene" in box.toolTip()


def test_l5_is_disabled_because_no_vignette_content_ships(dialog):
    box = dialog._level_boxes["L5"]
    assert box.isEnabled() is False
    assert "no vignette content is installed" in box.toolTip()


def test_curriculum_combo_is_populated_from_the_session(dialog):
    assert dialog._curriculum_combo.count() >= 1
    assert dialog._curriculum_keys[0] == "skull_bones"


def test_config_reflects_the_controls(dialog):
    dialog._count_spin.setValue(7)
    dialog._options_spin.setValue(4)
    dialog._seed_spin.setValue(42)
    dialog._format_combo.setCurrentIndex(1)          # timed station
    config = dialog.build_config()
    assert config.count == 7
    assert config.options == 4
    assert config.seed == 42
    assert config.fmt == "station"
    assert config.seconds_per_item > 0
    assert config.curriculum == "skull_bones"


def test_sba_format_has_no_per_item_clock(dialog):
    dialog._format_combo.setCurrentIndex(0)
    assert dialog.build_config().seconds_per_item == 0.0


def test_disabled_levels_are_never_selected(dialog):
    dialog._level_boxes["L4"].setChecked(True)       # disabled: must not count
    assert "L4" not in dialog.selected_levels()


# ── running a paper ──────────────────────────────────────────────────────

def test_begin_builds_a_paper_and_shows_the_first_item(dialog):
    dialog._count_spin.setValue(3)
    dialog._options_spin.setValue(3)
    assert dialog.begin() is True
    assert dialog._pages.currentIndex() == 1
    assert dialog._stem_label.text()
    assert len(dialog._option_buttons) == 3
    assert dialog._progress_label.text().startswith("Item 1 of 3")


def test_option_buttons_are_lettered(dialog):
    dialog._count_spin.setValue(1)
    dialog._options_spin.setValue(3)
    dialog.begin()
    assert dialog._option_buttons[0].text().startswith("A. ")
    assert dialog._option_buttons[2].text().startswith("C. ")


def test_a_correct_answer_is_acknowledged(dialog):
    dialog._count_spin.setValue(1)
    dialog._options_spin.setValue(3)
    dialog.begin()
    dialog._option_buttons[dialog._session.current.answer_index].setChecked(True)
    dialog.submit()
    assert dialog._feedback_label.text() == "Correct."
    assert dialog._next_button.isEnabled() is True
    assert dialog._submit_button.isEnabled() is False


def test_a_wrong_answer_shows_the_data_derived_explanation(dialog):
    dialog._count_spin.setValue(1)
    dialog._options_spin.setValue(3)
    dialog.begin()
    item = dialog._session.current
    wrong = 1 if item.answer_index == 0 else 0
    dialog._option_buttons[wrong].setChecked(True)
    dialog.submit()
    text = dialog._feedback_label.text()
    assert text.startswith("Incorrect.")
    assert item.options[wrong].text in text
    assert item.answer.text in text


def test_submitting_nothing_is_a_skip(dialog):
    dialog._count_spin.setValue(1)
    dialog._options_spin.setValue(3)
    dialog.begin()
    dialog.submit()
    assert dialog._feedback_label.text().startswith("Skipped.")


def test_submit_is_idempotent_within_an_item(dialog):
    dialog._count_spin.setValue(2)
    dialog._options_spin.setValue(3)
    dialog.begin()
    dialog.submit(skip=True)
    dialog.submit(skip=True)
    assert len(dialog._session.outcomes) == 1


def test_advancing_through_the_paper_reaches_the_results_page(dialog):
    dialog._count_spin.setValue(2)
    dialog._options_spin.setValue(3)
    dialog.begin()
    for _ in range(2):
        dialog.submit(skip=True)
        dialog.advance()
    assert dialog._pages.currentIndex() == 2
    assert "Examination complete." in dialog._results_label.text()
    assert "Score: 0/2" in dialog._results_label.text()


def test_results_report_lifetime_progress(dialog):
    dialog._count_spin.setValue(1)
    dialog._options_spin.setValue(3)
    dialog.begin()
    dialog._option_buttons[dialog._session.current.answer_index].setChecked(True)
    dialog.submit()
    dialog.advance()
    assert "All time:" in dialog._results_label.text()


def test_the_audit_view_shows_provenance_on_demand(dialog):
    dialog._count_spin.setValue(1)
    dialog._options_spin.setValue(3)
    dialog.begin()
    dialog.submit(skip=True)
    dialog.advance()
    assert dialog._audit_view.isHidden() is True
    dialog._audit_button.setChecked(True)
    assert "fma_label" in dialog._audit_view.toPlainText()
    assert dialog._audit_button.text() == "Hide provenance"


def test_an_impossible_selection_reports_rather_than_starting(dialog):
    dialog._curriculum_combo.setCurrentIndex(-1)
    assert dialog.begin() is False
    assert "No items could be built" in dialog._setup_status.text()
    assert dialog._pages.currentIndex() == 0


# ── station clock ────────────────────────────────────────────────────────

def test_station_mode_shows_and_counts_down_the_clock(dialog, clock):
    dialog._count_spin.setValue(2)
    dialog._options_spin.setValue(3)
    dialog._format_combo.setCurrentIndex(1)
    dialog._seconds_spin.setValue(60)
    dialog.begin()
    assert dialog._clock_bar.isHidden() is False
    assert "60 s remaining" in dialog._clock_label.text()
    clock.advance(seconds=25)
    dialog._on_tick()
    assert "35 s remaining" in dialog._clock_label.text()


def test_running_out_of_time_submits_a_skip(dialog, clock):
    dialog._count_spin.setValue(1)
    dialog._options_spin.setValue(3)
    dialog._format_combo.setCurrentIndex(1)
    dialog._seconds_spin.setValue(30)
    dialog.begin()
    clock.advance(seconds=31)
    dialog._on_tick()
    assert dialog._feedback_label.text().startswith("Out of time.")
    assert dialog._session.outcomes[-1].expired is True


def test_untimed_items_hide_the_clock(dialog):
    dialog._count_spin.setValue(1)
    dialog._options_spin.setValue(3)
    dialog._format_combo.setCurrentIndex(0)
    dialog.begin()
    assert dialog._clock_bar.isHidden() is True


# ── L4 wiring ────────────────────────────────────────────────────────────

def _cube_scene():
    from tests.anatomy.test_radiology_items import IDENTITY, cube
    return [
        (cube((0.0, 0.0, 0.0), 6.0, "Frontal Bone"), IDENTITY, "FMA4"),
        (cube((20.0, 0.0, 0.0), 6.0, "Parietal Bone"), IDENTITY, "FMA5"),
    ]


def test_setting_a_radiology_source_enables_l4(dialog):
    dialog.set_radiology_source(_cube_scene)
    box = dialog._level_boxes["L4"]
    assert box.isEnabled() is True
    assert "requires a loaded scene" not in box.toolTip()


def test_clearing_the_radiology_source_disables_l4_again(dialog):
    dialog.set_radiology_source(_cube_scene)
    dialog.set_radiology_source(None)
    box = dialog._level_boxes["L4"]
    assert box.isEnabled() is False
    assert box.isChecked() is False


def test_an_l4_paper_shows_the_slice_image(dialog):
    dialog.set_radiology_source(_cube_scene)
    dialog._level_boxes["L1"].setChecked(False)
    dialog._level_boxes["L4"].setChecked(True)
    dialog._count_spin.setValue(1)
    dialog._options_spin.setValue(3)
    assert dialog.begin() is True
    assert dialog._session.current.level == "L4"
    assert dialog._image_label.isHidden() is False
    assert dialog._image_label.pixmap() is not None
    assert not dialog._image_label.pixmap().isNull()


def test_a_failing_radiology_source_does_not_crash_the_dialog(dialog):
    def boom():
        raise RuntimeError("no scene")
    dialog.set_radiology_source(boom)
    dialog._level_boxes["L1"].setChecked(False)
    dialog._level_boxes["L4"].setChecked(True)
    dialog._count_spin.setValue(1)
    assert dialog.begin() is False
    assert "No items could be built" in dialog._setup_status.text()


def test_non_radiology_items_hide_the_image(dialog):
    dialog.set_radiology_source(_cube_scene)
    dialog._count_spin.setValue(1)
    dialog._options_spin.setValue(3)
    dialog.begin()
    assert dialog._image_label.isHidden() is True


# ── the tag overlay ──────────────────────────────────────────────────────

def test_scan_pixmap_renders_and_scales(qapp):
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:40, 20:40] = 0.9
    pixmap = _scan_pixmap(image, (30, 30), size=128)
    assert not pixmap.isNull()
    assert max(pixmap.width(), pixmap.height()) == 128


def test_scan_pixmap_handles_an_all_zero_image(qapp):
    pixmap = _scan_pixmap(np.zeros((32, 32), dtype=np.float32), (16, 16))
    assert not pixmap.isNull()


def test_scan_pixmap_accepts_a_colour_array(qapp):
    pixmap = _scan_pixmap(np.ones((32, 32, 3), dtype=np.float32), (5, 5))
    assert not pixmap.isNull()
