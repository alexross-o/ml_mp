from typing import Sequence

import torch
import torch.nn as nn
from torch.nn import functional

from simulator.events import (
    EVENT_TYPES,
    BindingSimEvent,
    MovementSimEvent,
    UnbindingSimEvent,
)

from .custom_unet import CustomUNet
from .unet_parts import Conv2Plus1D

DEFAULT_CLASS_PRIOR: float = 33 / 2_007_040


class EventDetector(nn.Module):
    """Per-pixel event detection head on top of a `CustomUNet` backbone.

    Predicts, at every spatiotemporal location: a per-class heatmap (one
    channel per registered event type, see `simulator.events.EVENT_TYPES`),
    a sub-pixel (dy, dx) centering offset, and a dipole orientation encoded
    as a unit vector.
    """

    # Aliases onto today's registered event types, kept for convenience/
    # backward compatibility -- a newly registered type doesn't get one of
    # these unless explicitly added.
    CLASS_BINDING: int = BindingSimEvent.class_index
    CLASS_UNBINDING: int = UnbindingSimEvent.class_index
    CLASS_MOVEMENT: int = MovementSimEvent.class_index

    def __init__(
        self,
        n_channels: int = 1,
        feat_channels: int = 3,
        min_channels: int = 8,
        trilinear: bool = False,
        n_event_classes: int = len(EVENT_TYPES),
        class_prior: Sequence[float] | None = None,
    ) -> None:
        """Initialize the event detector.

        Args:
            n_channels: Number of input channels.
            feat_channels: Number of channels in the backbone's output, i.e.
                `backbone.n_classes`.
            min_channels: Minimum number of channels in the backbone.
            trilinear: Whether to use trilinear interpolation for upsampling.
            n_event_classes: Number of heatmap output channels, one per
                event class. Defaults to `len(EVENT_TYPES)`, i.e. every
                currently registered event type.
            class_prior: Per-class expected positive-voxel fraction, used
                for `heatmap_head`'s bias init (see below); must have length
                `n_event_classes`. Defaults to a uniform prior
                (`DEFAULT_CLASS_PRIOR`) applied to every class -- a rough
                order-of-magnitude approximation rather than exact per-class
                counts, but one that works at any `n_event_classes`.

        Raises:
            ValueError: If `class_prior`'s length doesn't match
                `n_event_classes`.
        """
        super().__init__()
        self.n_event_classes = n_event_classes
        self.backbone = CustomUNet(
            n_channels=n_channels,
            n_classes=feat_channels,
            min_channels=min_channels,
            trilinear=trilinear,
        )

        # one independent channel per event class — sigmoid'd independently, not softmax
        self.heatmap_head = Conv2Plus1D(
            in_channels=feat_channels,
            mid_channels=16,
            out_channels=n_event_classes,
            temporal_kernel_size=1,
            temporal_padding=0,
            bias=True,
        )

        # shared across all classes — sub-pixel centering doesn't depend on class
        self.offset_head = Conv2Plus1D(
            in_channels=feat_channels,
            mid_channels=16,
            out_channels=2,  # dy, dx
            temporal_kernel_size=1,
            temporal_padding=0,
            bias=True,
        )

        # computed everywhere, but only ever supervised/read at locations of
        # classes with has_orientation = True (see EVENT_TYPES, model.loss.loss_fn)
        self.orientation_head = Conv2Plus1D(
            in_channels=feat_channels,
            mid_channels=16,
            out_channels=2,  # cos, sin — L2-normalized to a unit vector in forward()
            temporal_kernel_size=1,
            temporal_padding=0,
            bias=True,
        )

        # Per-class bias init (RetinaNet/CornerNet-style): near-zero weights
        # make a fresh conv output sigmoid(bias) everywhere, so setting bias
        # to the true class prior starts the network close to correct almost
        # everywhere instead of an overconfident 50/50 guess at every voxel.
        if class_prior is None:
            # Uniform prior applied to every class -- an order-of-magnitude
            # approximation that works at any n_event_classes, rather than
            # exact per-class counts. DEFAULT_CLASS_PRIOR is derived from
            # simulator.event_generator at OPTIMUM_EVENT_DENSITY (0.5
            # events/um^2/s) on a (500, 64, 64) movie, cropped to (490, 64,
            # 64) by gen_data's navg=5 dead-zone crop, where today's 3
            # registered types see roughly 33 events each out of 2,007,040
            # voxels/channel.
            class_prior = [DEFAULT_CLASS_PRIOR] * n_event_classes
        if len(class_prior) != n_event_classes:
            raise ValueError(
                f"class_prior must have length n_event_classes ({n_event_classes}), "
                f"got {len(class_prior)}"
            )

        pi = torch.tensor(class_prior)
        bias_init = -torch.log((1 - pi) / pi)
        with torch.no_grad():
            self.heatmap_head.temporal_conv.bias.copy_(bias_init)

        # offset/orientation targets are both symmetric about zero (offsets
        # are sub-pixel-uniform, movement angles are uniform over
        # [0, 2*pi)), so the origin is the MSE-optimal constant prediction
        # for either — start there instead of an arbitrary random direction.
        # For orientation this also avoids feeding functional.normalize a
        # small-but-nonzero vector, which is where its 1/||x|| gradient is
        # worst-conditioned; an exact-zero input lands cleanly on its
        # eps-clamped branch instead.
        nn.init.zeros_(self.offset_head.temporal_conv.weight)
        nn.init.zeros_(self.offset_head.temporal_conv.bias)
        nn.init.zeros_(self.orientation_head.temporal_conv.weight)
        nn.init.zeros_(self.orientation_head.temporal_conv.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the backbone and all three detection heads.

        Args:
            x: Input tensor of shape (N, n_channels, T, H, W).

        Returns:
            Dict with keys:
                "heatmap": Per-class logits, shape (N, n_event_classes, T, H,
                    W). Apply a sigmoid per channel (not softmax) — classes
                    aren't mutually exclusive.
                "offset": Sub-pixel (dy, dx) centering offset, shape
                    (N, 2, T, H, W).
                "orientation": Dipole orientation as a unit (cos, sin) vector,
                    shape (N, 2, T, H, W). Recover the angle via
                    `torch.atan2(orientation[:, 1], orientation[:, 0])`.
        """
        features = self.backbone(x)

        return {
            "heatmap": self.heatmap_head(features),
            "offset": self.offset_head(features),
            "orientation": functional.normalize(
                self.orientation_head(features), dim=1, eps=1e-6
            ),
        }
