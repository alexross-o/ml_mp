"""Regression test for the Step 1 event-registry refactor (simulator/events.py).

Checks that the registry-driven `gen_events` reproduces the pre-refactor,
hardcoded binding/unbinding/movement sampling bit-for-bit, and that the
registry itself matches today's class layout. This is a one-time migration
safety net -- safe to delete once the refactor is trusted -- not a test of
ongoing behavior (a future 4th event type is expected to change the numbers
this compares against).
"""

import numpy as np
import pytest

from simulator.events import (
    NM_PER_PX,
    BindingSimEvent,
    EVENT_TYPES,
    MovementSimEvent,
    UnbindingSimEvent,
    density_to_n_events,
    gen_events,
)

_BORDER_MASK = 5
_MOV_SHAPE = (500, 64, 64)
_CONTRAST_RANGE = (0.01, 0.03)
_DISTANCE_RANGE = (1.5, 300)
_EVENT_DENSITY = 0.5
_NAVG = 5

# Frozen constants mirroring the pre-refactor module-level ones, so
# `_legacy_gen_events` below doesn't depend on the (now-removed) originals.
_MOVEMENT_SMALL_DISTANCE_FRACTION = 0.6
_MOVEMENT_MEDIUM_DISTANCE_FRACTION = 0.3
_MOVEMENT_SMALL_DISTANCE_MAX_NM = 20.0
_MOVEMENT_MEDIUM_DISTANCE_MAX_NM = 100.0


def _legacy_gen_events(
    event_density, mov_shape, contrast_range, distance_range, event_type_weight, navg, rng
):
    """Frozen copy of the pre-refactor `gen_events` body (binding/unbinding/
    movement hardcoded inline), kept only so this test can assert the new
    registry-driven dispatcher draws identically from a shared `rng`."""
    weights = np.array(event_type_weight) / np.sum(event_type_weight)

    mov_shape_masked = (
        mov_shape[0],
        mov_shape[1] - 2 * _BORDER_MASK,
        mov_shape[2] - 2 * _BORDER_MASK,
    )

    n_bindings, n_unbindings, n_movements = (
        density_to_n_events(density=density, shape=mov_shape_masked)
        for density in weights * event_density
    )

    x_low, x_high = _BORDER_MASK - 0.5, mov_shape[2] - (_BORDER_MASK - 0.5)
    y_low, y_high = _BORDER_MASK - 0.5, mov_shape[1] - (_BORDER_MASK - 0.5)
    i_low, i_high = navg, mov_shape[0] - navg

    binding_evs = [
        BindingSimEvent(x=x, y=y, i=i, c=c)
        for x, y, i, c in zip(
            rng.uniform(x_low, x_high, n_bindings),
            rng.uniform(y_low, y_high, n_bindings),
            rng.uniform(i_low, i_high, n_bindings),
            rng.uniform(contrast_range[0], contrast_range[1], n_bindings),
        )
    ]
    unbinding_evs = [
        UnbindingSimEvent(x=x, y=y, i=i, c=c)
        for x, y, i, c in zip(
            rng.uniform(x_low, x_high, n_unbindings),
            rng.uniform(y_low, y_high, n_unbindings),
            rng.uniform(i_low, i_high, n_unbindings),
            -rng.uniform(contrast_range[0], contrast_range[1], n_unbindings),
        )
    ]

    n_small_movements = int(_MOVEMENT_SMALL_DISTANCE_FRACTION * n_movements)
    n_medium_movements = int(_MOVEMENT_MEDIUM_DISTANCE_FRACTION * n_movements)
    n_large_movements = n_movements - n_small_movements - n_medium_movements

    movement_evs = [
        MovementSimEvent(x=x, y=y, i=i, c=c, distance=distance / NM_PER_PX, theta=theta)
        for x, y, i, c, distance, theta in zip(
            rng.uniform(x_low, x_high, n_movements),
            rng.uniform(y_low, y_high, n_movements),
            rng.uniform(i_low, i_high, n_movements),
            rng.uniform(contrast_range[0], contrast_range[1], n_movements),
            np.concatenate(
                [
                    rng.uniform(
                        distance_range[0], _MOVEMENT_SMALL_DISTANCE_MAX_NM, n_small_movements
                    ),
                    rng.uniform(
                        _MOVEMENT_SMALL_DISTANCE_MAX_NM,
                        _MOVEMENT_MEDIUM_DISTANCE_MAX_NM,
                        n_medium_movements,
                    ),
                    rng.uniform(
                        _MOVEMENT_MEDIUM_DISTANCE_MAX_NM, distance_range[1], n_large_movements
                    ),
                ]
            ),
            rng.uniform(0, 2 * np.pi, n_movements),
        )
    ]

    return binding_evs + unbinding_evs + movement_evs


def test_registry_sanity():
    assert len(EVENT_TYPES) == 3
    assert [event_type.name for event_type in EVENT_TYPES] == ["binding", "unbinding", "movement"]
    assert (BindingSimEvent.class_index, UnbindingSimEvent.class_index, MovementSimEvent.class_index) == (
        0,
        1,
        2,
    )
    assert BindingSimEvent.has_orientation is False
    assert UnbindingSimEvent.has_orientation is False
    assert MovementSimEvent.has_orientation is True


@pytest.mark.parametrize("seed", [0, 12345, 999])
def test_sampling_parity_with_legacy_algorithm(seed):
    new_events = gen_events(
        event_density=_EVENT_DENSITY,
        mov_shape=_MOV_SHAPE,
        contrast_range=_CONTRAST_RANGE,
        distance_range=_DISTANCE_RANGE,
        navg=_NAVG,
        rng=np.random.default_rng(seed),
    )
    legacy_events = _legacy_gen_events(
        event_density=_EVENT_DENSITY,
        mov_shape=_MOV_SHAPE,
        contrast_range=_CONTRAST_RANGE,
        distance_range=_DISTANCE_RANGE,
        event_type_weight=(1, 1, 1),
        navg=_NAVG,
        rng=np.random.default_rng(seed),
    )

    assert len(new_events) == len(legacy_events)
    for new_event, legacy_event in zip(new_events, legacy_events):
        assert type(new_event) is type(legacy_event)
        assert new_event.x == legacy_event.x
        assert new_event.y == legacy_event.y
        assert new_event.i == legacy_event.i
        assert new_event.c == legacy_event.c
        if isinstance(new_event, MovementSimEvent):
            assert new_event.distance == legacy_event.distance
            assert new_event.theta == legacy_event.theta


def test_event_type_weight_rejects_unrecognized_name():
    with pytest.raises(ValueError):
        gen_events(
            event_density=_EVENT_DENSITY,
            mov_shape=_MOV_SHAPE,
            contrast_range=_CONTRAST_RANGE,
            event_type_weight={"binding": 1.0, "bleaching": 1.0},
            navg=_NAVG,
            rng=np.random.default_rng(0),
        )
