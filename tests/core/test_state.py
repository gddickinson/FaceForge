"""Tests for state management."""

import pytest

from faceforge.core.state import (
    FaceState, BodyState, TargetAU, TargetHead,
    ConstraintState, StateManager, AU_IDS,
)


def test_face_state_defaults():
    fs = FaceState()
    assert fs.AU1 == 0.0
    assert fs.head_yaw == 0.0
    assert fs.auto_blink is False
    assert fs.current_expression == "neutral"


def test_face_state_get_set_au():
    fs = FaceState()
    fs.set_au("AU12", 0.8)
    assert fs.get_au("AU12") == 0.8


def test_face_state_au_clamping():
    fs = FaceState()
    fs.set_au("AU1", 1.5)
    assert fs.get_au("AU1") == 1.0
    fs.set_au("AU1", -0.5)
    assert fs.get_au("AU1") == 0.0


def test_face_state_au_dict():
    fs = FaceState()
    fs.set_au("AU6", 0.5)
    fs.set_au("AU12", 0.9)
    d = fs.get_au_dict()
    assert d["AU6"] == 0.5
    assert d["AU12"] == 0.9
    assert d["AU1"] == 0.0


def test_face_state_set_aus_from_dict():
    fs = FaceState()
    fs.set_aus_from_dict({"AU1": 0.3, "AU4": 0.7})
    assert fs.AU1 == 0.3
    assert fs.AU4 == 0.7


def test_target_au():
    t = TargetAU()
    t.set("AU12", 0.9)
    assert t.get("AU12") == 0.9
    d = t.to_dict()
    assert d["AU12"] == 0.9


def test_body_state_defaults():
    bs = BodyState()
    assert bs.spine_flex == 0.0
    assert bs.breath_depth == 0.3
    assert bs.auto_breath_body is False


def test_body_state_js_key_map():
    bs = BodyState()
    bs.set_from_js_dict({"spineFlex": 0.5, "kneeRFlex": 0.7})
    assert bs.spine_flex == 0.5
    assert bs.knee_r_flex == 0.7


def test_body_state_to_dict():
    bs = BodyState()
    bs.spine_flex = 0.3
    d = bs.to_dict()
    assert d["spine_flex"] == 0.3
    assert "spine_flex" in d


def test_constraint_state():
    cs = ConstraintState()
    assert cs.total_excess == 0.0
    assert cs.spine_compensation_yaw == 0.0


def test_state_manager():
    sm = StateManager()
    assert isinstance(sm.face, FaceState)
    assert isinstance(sm.body, BodyState)
    assert isinstance(sm.target_au, TargetAU)
    assert isinstance(sm.target_head, TargetHead)
    assert sm.frame_count == 0


def test_au_ids():
    assert len(AU_IDS) == 12
    assert "AU1" in AU_IDS
    assert "AU26" in AU_IDS
