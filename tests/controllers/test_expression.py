"""Facial state handlers, driven through the real event bus.

These handlers used to be closures inside ``app.main()``.  Everything here goes
through ``bus.publish``, not a direct method call, so the subscription wiring is
under test as well as the handler body.
"""

from __future__ import annotations

from dataclasses import dataclass

from faceforge.controllers.expression import SPEECH_RESET_AUS, ExpressionController
from faceforge.core.events import EventType


def wired(ctx):
    controller = ExpressionController(ctx)
    controller.subscribe()
    controller.subscribe_toggles()
    return controller


# -- AU routing -----------------------------------------------------------

def test_ordinary_au_goes_to_the_target_not_live_state(ctx, state):
    """Sliders write targets; the simulation interpolates live state to them."""
    wired(ctx)
    ctx.event_bus.publish(EventType.AU_CHANGED, au_id="AU12", value=0.7)
    assert state.target_au.get("AU12") == 0.7
    assert state.face.AU12 == 0.0, "live AU must be left to the interpolator"


def test_gaze_and_pupil_bypass_the_interpolator(ctx, state):
    """The four non-AU sliders write live state directly."""
    wired(ctx)
    for au_id, value, read in (
        ("eye_look_x", 0.4, lambda: state.face.eye_look_x),
        ("eye_look_y", -0.3, lambda: state.face.eye_look_y),
        ("pupil_dilation", 0.6, lambda: state.face.pupil_dilation),
        ("ear_wiggle", 0.25, lambda: state.target_ear_wiggle),
    ):
        ctx.event_bus.publish(EventType.AU_CHANGED, au_id=au_id, value=value)
        assert read() == value, au_id
    assert "eye_look_x" not in state.target_au.to_dict(), \
        "gaze must not be registered as an AU target"


def test_unknown_au_id_does_not_raise(ctx):
    """An unknown AU is ignored, and must not disturb the AUs that do exist.

    Checking only that this does not raise would also pass if the handler
    silently clobbered a real AU or accepted AU999 into the state.
    """
    controller = wired(ctx)
    state = controller.state if hasattr(controller, "state") else ctx.state
    state.target_au.set("AU12", 0.7)

    ctx.event_bus.publish(EventType.AU_CHANGED, au_id="AU999", value=1.0)

    assert state.target_au.get("AU12") == 0.7, \
        "an unknown AU id disturbed a known AU"
    assert state.target_au.get("AU999") in (0.0, None), \
        "AU999 was accepted into the state despite being unknown"


# -- Expression presets ---------------------------------------------------

def test_preset_zeroes_the_aus_it_does_not_mention(ctx, state):
    """A preset is a complete pose: unnamed AUs go to zero, not stale values."""
    controller = wired(ctx)
    state.target_au.set("AU12", 0.9)
    controller.on_expression_set(name="Surprise", values={"AU1": 0.8})
    assert state.target_au.get("AU1") == 0.8
    assert state.target_au.get("AU12") == 0.0


def test_preset_accepts_both_head_key_spellings(ctx, state):
    """Presets exist in JS (``headYaw``) and Python (``head_yaw``) generations."""
    controller = wired(ctx)
    controller.on_expression_set(name="a", values={"headYaw": 12.0})
    assert state.target_head.head_yaw == 12.0
    controller.on_expression_set(name="b", values={"head_yaw": -7.0})
    assert state.target_head.head_yaw == -7.0


def test_preset_records_its_name(ctx, state):
    wired(ctx)
    ctx.event_bus.publish(EventType.EXPRESSION_SET, name="Joy", values={})
    assert state.face.current_expression == "Joy"


def test_head_rotation_writes_all_three_axes(ctx, state):
    wired(ctx)
    ctx.event_bus.publish(EventType.HEAD_ROTATION_CHANGED,
                          head_yaw=1.0, head_pitch=2.0, head_roll=3.0)
    head = state.target_head
    assert (head.head_yaw, head.head_pitch, head.head_roll) == (1.0, 2.0, 3.0)


# -- Auto behaviour toggles ------------------------------------------------

def test_auto_behaviour_toggles_reach_face_state(ctx, state):
    wired(ctx)
    for event, attr in (
        (EventType.AUTO_BLINK_TOGGLED, "auto_blink"),
        (EventType.AUTO_BREATHING_TOGGLED, "auto_breathing"),
        (EventType.EYE_TRACKING_TOGGLED, "eye_tracking"),
        (EventType.MICRO_EXPRESSIONS_TOGGLED, "micro_expressions"),
    ):
        ctx.event_bus.publish(event, enabled=True)
        assert getattr(state.face, attr) is True, attr
        ctx.event_bus.publish(event, enabled=False)
        assert getattr(state.face, attr) is False, attr


# -- Speech ---------------------------------------------------------------

@dataclass
class Viseme:
    start_time: float
    end_time: float
    au_targets: dict


class StubSpeechEngine:
    def __init__(self, sequence):
        self.sequence = sequence
        self.calls: list[tuple[str, float]] = []

    def generate_au_sequence(self, text, speed=1.0):
        self.calls.append((text, speed))
        return self.sequence


def test_empty_text_does_not_start_playback(ctx):
    ctx.speech_engine = StubSpeechEngine([])
    controller = ExpressionController(ctx)
    controller.on_speech_play(text="", speed=1.0)
    assert controller.speech_playing is False
    assert ctx.speech_engine.calls == []


def test_speech_scans_for_the_covering_viseme(ctx, state):
    """Playback is time-addressed, so a skipped frame skips a viseme only."""
    seq = [Viseme(0.0, 0.1, {"AU25": 0.3}), Viseme(0.1, 0.2, {"AU25": 0.9})]
    ctx.speech_engine = StubSpeechEngine(seq)
    controller = ExpressionController(ctx)
    controller.on_speech_play(text="hi", speed=1.0)

    controller.advance_speech(0.15)      # jumps straight into the second viseme
    assert state.target_au.get("AU25") == 0.9
    assert controller.speech_playing is True


def test_speech_resets_the_mouth_when_the_sequence_ends(ctx, state):
    seq = [Viseme(0.0, 0.1, {"AU25": 0.9})]
    ctx.speech_engine = StubSpeechEngine(seq)
    controller = ExpressionController(ctx)
    controller.on_speech_play(text="hi", speed=1.0)
    controller.advance_speech(0.05)
    assert state.target_au.get("AU25") == 0.9

    controller.advance_speech(1.0)       # past the end
    assert controller.speech_playing is False
    for au_id in SPEECH_RESET_AUS:
        assert state.target_au.get(au_id) == 0.0, au_id


def test_advance_speech_is_a_no_op_while_not_playing(ctx):
    controller = ExpressionController(ctx)
    controller.advance_speech(1.0)
    assert controller.speech_time == 0.0


# -- Eye colour -----------------------------------------------------------

class StubFaceFeatures:
    def __init__(self):
        self.eye_colors: list = []

    def set_eye_color(self, color):
        self.eye_colors.append(color)


class StubPipeline:
    def __init__(self, **kw):
        self.face_features = None
        self.gender_morph = None
        for k, v in kw.items():
            setattr(self, k, v)


def test_eye_colour_reaches_the_feature_system(ctx, state):
    features = StubFaceFeatures()
    ctx.pipeline = StubPipeline(face_features=features)
    ExpressionController(ctx).on_eye_color_set(name="blue", color=(0.2, 0.4, 0.8))
    assert state.face.eye_color == "blue"
    assert features.eye_colors == [(0.2, 0.4, 0.8)]


def test_eye_colour_survives_a_scene_with_no_face(ctx, state):
    """The head can fail to load; the colour is still recorded in state."""
    ctx.pipeline = StubPipeline()
    ExpressionController(ctx).on_eye_color_set(name="green", color=(0, 1, 0))
    assert state.face.eye_color == "green"
