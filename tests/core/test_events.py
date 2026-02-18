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
    bus = EventBus()
    bus.subscribe(EventType.AU_CHANGED, lambda **kw: None)
    bus.clear()
    # Should not raise
    bus.publish(EventType.AU_CHANGED)
