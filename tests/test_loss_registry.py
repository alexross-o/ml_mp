"""Regression test for the Step 4 registry refactor (model/loss.py).

Checks that loss_fn's new orientation_channels default (derived from
simulator.events.EVENT_TYPES's has_orientation flag) reproduces the
pre-refactor movement_channel=EventDetector.CLASS_MOVEMENT behavior
bit-for-bit on a fixed synthetic batch. One-time migration safety net --
safe to delete once the refactor is trusted -- not a test of ongoing
behavior (a future orientation-bearing type changes what this compares
against).
"""

import torch

from model.loss import (
    DEFAULT_NON_HEATMAP_LOSS,
    DEFAULT_ORIENTATION_CHANNELS,
    loss_fn,
    loss_fn_heatmap,
    loss_fn_offset,
    loss_fn_orientation,
)
from model.model import EventDetector


def _make_batch(seed=0):
    generator = torch.Generator().manual_seed(seed)
    n, c, t, h, w = 1, 3, 4, 8, 8
    predictions = {
        "heatmap": torch.randn(n, c, t, h, w, generator=generator),
        "offset": torch.randn(n, 2, t, h, w, generator=generator),
        "orientation": torch.randn(n, 2, t, h, w, generator=generator),
    }
    ground_truth = {
        "heatmap": torch.zeros(n, c, t, h, w),
        "offset": torch.randn(n, 2, t, h, w, generator=generator),
        "orientation": torch.randn(n, 2, t, h, w, generator=generator),
    }
    # scatter a peak voxel (value 1) per class so both offset and
    # orientation masks are non-degenerate
    for cls in range(c):
        ground_truth["heatmap"][0, cls, cls % t, cls % h, cls % w] = 1.0
    return predictions, ground_truth


def _legacy_loss_fn(predictions, ground_truth, movement_channel):
    """Frozen copy of the pre-refactor `loss_fn` body (single movement_channel
    slice instead of orientation_channels), kept only so this test can assert
    the new version renders identically."""
    loss_heatmap, hm_mask = loss_fn_heatmap(predictions["heatmap"], ground_truth["heatmap"])

    offset_mask = hm_mask.any(dim=1, keepdim=True).float()
    loss_offset = loss_fn_offset(
        predictions["offset"], ground_truth["offset"], offset_mask, DEFAULT_NON_HEATMAP_LOSS
    )

    orientation_mask = hm_mask[:, movement_channel : movement_channel + 1]
    loss_orientation = loss_fn_orientation(
        predictions["orientation"],
        ground_truth["orientation"],
        orientation_mask,
        DEFAULT_NON_HEATMAP_LOSS,
    )

    return loss_heatmap + 1.0 * loss_offset + 0.8 * loss_orientation


def test_default_orientation_channels_matches_movement_class():
    assert list(DEFAULT_ORIENTATION_CHANNELS) == [EventDetector.CLASS_MOVEMENT]


def test_loss_parity_with_legacy_movement_channel_slicing():
    predictions, ground_truth = _make_batch()

    new_loss = loss_fn(predictions, ground_truth)
    legacy_loss = _legacy_loss_fn(
        predictions, ground_truth, movement_channel=EventDetector.CLASS_MOVEMENT
    )

    assert torch.equal(new_loss, legacy_loss)


def test_orientation_channels_accepts_explicit_subset():
    predictions, ground_truth = _make_batch()

    # binding (0) and unbinding (1) have no orientation ground truth
    # scattered as non-zero in this batch's construction, but the API
    # should still accept an arbitrary explicit subset without raising.
    loss = loss_fn(predictions, ground_truth, orientation_channels=(0, 1))
    assert torch.isfinite(loss)


def test_empty_orientation_channels_does_not_crash():
    predictions, ground_truth = _make_batch()

    loss = loss_fn(predictions, ground_truth, orientation_channels=())
    assert torch.isfinite(loss)
