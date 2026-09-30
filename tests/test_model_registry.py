"""Regression test for the Step 3 registry refactor (model/model.py).

Checks that EventDetector's default instantiation (n_event_classes derived
from simulator.events.EVENT_TYPES) produces identical head shapes and
bias-init values to the pre-refactor hardcoded-3-classes version, and that
a non-default n_event_classes without an explicit class_prior is rejected.
One-time migration safety net -- safe to delete once the refactor is
trusted -- not a test of ongoing behavior (a future 4th event type is
expected to change these numbers).
"""

import pytest
import torch

from model.model import EventDetector
from simulator.events import EVENT_TYPES


def test_class_aliases_match_registry():
    assert EventDetector.CLASS_BINDING == 0
    assert EventDetector.CLASS_UNBINDING == 1
    assert EventDetector.CLASS_MOVEMENT == 2


def test_default_head_shapes_match_legacy():
    model = EventDetector()
    assert model.heatmap_head.temporal_conv.out_channels == len(EVENT_TYPES) == 3
    assert model.offset_head.temporal_conv.out_channels == 2
    assert model.orientation_head.temporal_conv.out_channels == 2


def test_default_bias_init_matches_legacy_values():
    model = EventDetector()
    legacy_pi = torch.tensor([37 / 2_007_040, 37 / 2_007_040, 29 / 2_007_040])
    legacy_bias = -torch.log((1 - legacy_pi) / legacy_pi)
    assert torch.equal(model.heatmap_head.temporal_conv.bias.detach(), legacy_bias)


def test_forward_pass_shapes():
    model = EventDetector()
    model.eval()
    x = torch.zeros(1, 1, 8, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out["heatmap"].shape == (1, 3, 8, 64, 64)
    assert out["offset"].shape == (1, 2, 8, 64, 64)
    assert out["orientation"].shape == (1, 2, 8, 64, 64)


def test_non_default_n_event_classes_requires_class_prior():
    with pytest.raises(ValueError):
        EventDetector(n_event_classes=4)


def test_non_default_n_event_classes_with_class_prior_works():
    model = EventDetector(n_event_classes=4, class_prior=[0.1, 0.1, 0.1, 0.1])
    assert model.heatmap_head.temporal_conv.out_channels == 4


def test_class_prior_length_mismatch_rejected():
    with pytest.raises(ValueError):
        EventDetector(n_event_classes=4, class_prior=[0.1, 0.1])
