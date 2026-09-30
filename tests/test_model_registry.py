"""Regression test for the Step 3 registry refactor (model/model.py).

Checks that EventDetector's default instantiation (n_event_classes derived
from simulator.events.EVENT_TYPES) produces the expected head shapes and
bias-init values, and that the uniform DEFAULT_CLASS_PRIOR fallback works
at any n_event_classes without requiring an explicit class_prior.
"""

import pytest
import torch

from model.model import DEFAULT_CLASS_PRIOR, EventDetector
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


def test_default_bias_init_uses_uniform_prior():
    model = EventDetector()
    expected_pi = torch.tensor([DEFAULT_CLASS_PRIOR] * len(EVENT_TYPES))
    expected_bias = -torch.log((1 - expected_pi) / expected_pi)
    assert torch.equal(model.heatmap_head.temporal_conv.bias.detach(), expected_bias)


def test_forward_pass_shapes():
    model = EventDetector()
    model.eval()
    x = torch.zeros(1, 1, 8, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out["heatmap"].shape == (1, 3, 8, 64, 64)
    assert out["offset"].shape == (1, 2, 8, 64, 64)
    assert out["orientation"].shape == (1, 2, 8, 64, 64)


def test_non_default_n_event_classes_uses_uniform_prior_by_default():
    model = EventDetector(n_event_classes=4)
    assert model.heatmap_head.temporal_conv.out_channels == 4
    expected_pi = torch.tensor([DEFAULT_CLASS_PRIOR] * 4)
    expected_bias = -torch.log((1 - expected_pi) / expected_pi)
    assert torch.equal(model.heatmap_head.temporal_conv.bias.detach(), expected_bias)


def test_non_default_n_event_classes_with_explicit_class_prior_works():
    model = EventDetector(n_event_classes=4, class_prior=[0.1, 0.1, 0.1, 0.1])
    assert model.heatmap_head.temporal_conv.out_channels == 4


def test_class_prior_length_mismatch_rejected():
    with pytest.raises(ValueError):
        EventDetector(n_event_classes=4, class_prior=[0.1, 0.1])
