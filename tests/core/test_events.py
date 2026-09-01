"""Tests for event bus."""

from faceforge.core.events import EventBus, EventType


def test_subscribe_publish():
    bus = EventBus()
    received = []
    bus.subscribe(EventType.AU_CHANGED, lambda **kw: received.append(kw))
    bus.publish(EventType.AU_CHANGED, au_id="AU1", value=0.5)
    assert len(received) == 1
    assert received[0] == {"au_id": "AU1", "value": 0.5}


def test_unsubscribe():
    bus = EventBus()
    received = []
    handler = lambda **kw: received.append(kw)
    bus.subscribe(EventType.AU_CHANGED, handler)
    bus.unsubscribe(EventType.AU_CHANGED, handler)
    bus.publish(EventType.AU_CHANGED, au_id="AU1", value=0.5)
    assert len(received) == 0


def test_multiple_subscribers():
    bus = EventBus()
    a, b = [], []
    bus.subscribe(EventType.EXPRESSION_SET, lambda **kw: a.append(1))
    bus.subscribe(EventType.EXPRESSION_SET, lambda **kw: b.append(1))
    bus.publish(EventType.EXPRESSION_SET, name="happy")
    assert len(a) == 1
    assert len(b) == 1


def test_different_events_independent():
    bus = EventBus()
    received = []
    bus.subscribe(EventType.AU_CHANGED, lambda **kw: received.append("au"))
    bus.publish(EventType.EXPRESSION_SET, name="happy")
    assert len(received) == 0


def test_clear():
    """clear() must actually detach subscribers, not merely survive a publish.

    The original test published after clear() and checked only that nothing
    raised -- which would also have passed if clear() did nothing at all, since
    the handler was a no-op lambda.
    """
    bus = EventBus()
    received = []
    bus.subscribe(EventType.AU_CHANGED, lambda **kw: received.append(kw))

    bus.publish(EventType.AU_CHANGED, au_id="AU1", value=1.0)
    assert len(received) == 1, "the subscriber was not wired up to begin with"

    bus.clear()
    bus.publish(EventType.AU_CHANGED, au_id="AU1", value=1.0)
    assert len(received) == 1, "clear() left the subscriber attached"
