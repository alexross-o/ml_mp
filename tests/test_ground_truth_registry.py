"""Regression test for the Step 2 registry refactor (simulator/ground_truth.py).

Checks that the registry-driven `gen_ground_truth` reproduces the
pre-refactor, hardcoded isinstance-chain implementation bit-for-bit on a
fixed synthetic event list. This is a one-time migration safety net --
safe to delete once the refactor is trusted -- not a test of ongoing
behavior (a future 4th event type is expected to change the heatmap shape
this compares against).
"""

import numpy as np
import torch

from model.model import EventDetector
from simulator.constants import HEATMAP_GAUSSIAN_SIGMA_PX, HEATMAP_GAUSSIAN_THUMBNAIL_SIZE
from simulator.events import EVENT_TYPES, BindingSimEvent, MovementSimEvent, UnbindingSimEvent
from simulator.ground_truth import gen_ground_truth

_MOV_SHAPE = (20, 64, 64)

_EVENTS = [
    BindingSimEvent(x=10.3, y=15.7, i=5, c=0.02),
    UnbindingSimEvent(x=30.1, y=40.2, i=8, c=-0.015),
    MovementSimEvent(x=32.0, y=32.0, i=10, c=0.02, distance=3.0, theta=0.7),
    BindingSimEvent(x=50.0, y=50.0, i=12, c=0.01),
    UnbindingSimEvent(x=20.0, y=20.0, i=1, c=-0.02),
]


def _legacy_gen_ground_truth(events, mov_shape, sigma_px=HEATMAP_GAUSSIAN_SIGMA_PX):
    """Frozen copy of the pre-refactor `gen_ground_truth` body (hardcoded
    isinstance chain into `EventDetector.CLASS_*`), kept only so this test
    can assert the new registry-driven version renders identically."""
    t_size, h_size, w_size = mov_shape

    heatmap = np.zeros((3, t_size, h_size, w_size), dtype=np.float32)
    offset = np.zeros((2, t_size, h_size, w_size), dtype=np.float32)
    orientation = np.zeros((2, t_size, h_size, w_size), dtype=np.float32)

    half = HEATMAP_GAUSSIAN_THUMBNAIL_SIZE // 2
    patch_yy, patch_xx = np.mgrid[-half : half + 1, -half : half + 1]
    gaussian_patch = np.exp(-(patch_xx**2 + patch_yy**2) / (2 * sigma_px**2))

    for event in events:
        if isinstance(event, BindingSimEvent):
            channel = EventDetector.CLASS_BINDING
        elif isinstance(event, UnbindingSimEvent):
            channel = EventDetector.CLASS_UNBINDING
        elif isinstance(event, MovementSimEvent):
            channel = EventDetector.CLASS_MOVEMENT
        else:
            raise TypeError(f"unrecognized event type: {type(event).__name__}")

        frame = int(np.clip(np.round(event.i), 0, t_size - 1))
        x_px, y_px = event.hot_px
        dx, dy = event.offset

        heatmap[
            channel,
            frame,
            y_px - half : y_px + half + 1,
            x_px - half : x_px + half + 1,
        ] += gaussian_patch

        offset[:, frame, y_px, x_px] = (dy, dx)

        if isinstance(event, MovementSimEvent):
            orientation[:, frame, y_px, x_px] = (np.cos(event.theta), np.sin(event.theta))

    heatmap = np.clip(heatmap, 0.0, 1.0)

    return {
        "heatmap": torch.from_numpy(heatmap),
        "offset": torch.from_numpy(offset),
        "orientation": torch.from_numpy(orientation),
    }


def test_ground_truth_parity_with_legacy_algorithm():
    new_gt = gen_ground_truth(events=_EVENTS, mov_shape=_MOV_SHAPE)
    legacy_gt = _legacy_gen_ground_truth(events=_EVENTS, mov_shape=_MOV_SHAPE)

    assert new_gt["heatmap"].shape[0] == len(EVENT_TYPES) == 3
    assert torch.equal(new_gt["heatmap"], legacy_gt["heatmap"])
    assert torch.equal(new_gt["offset"], legacy_gt["offset"])
    assert torch.equal(new_gt["orientation"], legacy_gt["orientation"])


def test_binding_unbinding_have_no_orientation():
    assert BindingSimEvent(x=1, y=1, i=0, c=0.01).orientation is None
    assert UnbindingSimEvent(x=1, y=1, i=0, c=-0.01).orientation is None


def test_movement_orientation_matches_theta():
    event = MovementSimEvent(x=32, y=32, i=5, c=0.02, distance=3.0, theta=0.7)
    cos, sin = event.orientation
    assert cos == np.cos(event.theta)
    assert sin == np.sin(event.theta)
