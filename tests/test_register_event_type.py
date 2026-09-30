"""Tests for `register_event_type`'s duplicate checks."""

import pytest

from simulator.events import EVENT_TYPES, BindingSimEvent, register_event_type


@pytest.fixture(autouse=True)
def _restore_registry():
    snapshot = list(EVENT_TYPES)
    yield
    EVENT_TYPES[:] = snapshot


def test_registering_same_class_twice_is_rejected():
    n_before = len(EVENT_TYPES)
    with pytest.raises(ValueError, match="already registered"):
        register_event_type(BindingSimEvent)
    assert len(EVENT_TYPES) == n_before
    assert BindingSimEvent.class_index == EVENT_TYPES.index(BindingSimEvent)


def test_duplicate_name_is_rejected():
    class ImpostorSimEvent(BindingSimEvent):
        name = BindingSimEvent.name

    n_before = len(EVENT_TYPES)
    with pytest.raises(ValueError, match="already used by BindingSimEvent"):
        register_event_type(ImpostorSimEvent)
    assert len(EVENT_TYPES) == n_before


def test_new_unique_type_gets_next_class_index():
    class BleachingSimEvent(BindingSimEvent):
        name = "bleaching"

    n_before = len(EVENT_TYPES)
    register_event_type(BleachingSimEvent)
    assert BleachingSimEvent.class_index == n_before
    assert EVENT_TYPES[-1] is BleachingSimEvent
