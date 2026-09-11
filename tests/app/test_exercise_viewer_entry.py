"""The standalone exercise viewer enters its mode when the load *sequence* completes.

``LOADING_COMPLETE`` is published by the skeleton pipeline several stages
before body animation, the rib pivots and the skinning are wired; the viewer
must not enter (and start an exercise) on it.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from faceforge.coordination.asset_load_sequence import LoadStage       # noqa: E402
from faceforge.exercise_viewer import watch_load_sequence               # noqa: E402


class _Sequence:
    def __init__(self, on_stage=None):
        self.on_stage = on_stage


def test_enters_once_deferred_when_the_sequence_reaches_complete():
    seen: list = []
    seq = _Sequence(on_stage=seen.append)
    deferred: list = []
    entered: list = []
    watch_load_sequence(seq, lambda: entered.append(True),
                        defer=lambda _ms, fn: deferred.append(fn))

    for stage in (LoadStage.LOAD_HEAD, LoadStage.LOAD_BODY_SKELETON,
                  LoadStage.WIRE_BODY_ANIMATION, LoadStage.BUILD_SKINNING):
        seq.on_stage(stage)
    assert seen[-1] is LoadStage.BUILD_SKINNING     # the previous observer still runs
    assert not deferred and not entered              # nothing before the last stage

    seq.on_stage(LoadStage.COMPLETE)
    assert len(deferred) == 1 and not entered        # queued for the event loop
    deferred[0]()
    assert entered == [True]

    seq.on_stage(LoadStage.COMPLETE)                 # a second completion does not re-enter
    assert len(deferred) == 1


def test_works_without_a_previous_observer():
    seq = _Sequence()
    calls: list = []
    watch_load_sequence(seq, lambda: calls.append(1), defer=lambda _ms, fn: fn())
    seq.on_stage(LoadStage.COMPLETE)
    assert calls == [1]
