import torch
import torch.nn as nn
from torch.nn import functional

from .custom_unet import CustomUNet
from .unet_parts import Conv2Plus1D


class EventDetector(nn.Module):
    """Per-pixel event detection head on top of a `CustomUNet` backbone.

    Predicts, at every spatiotemporal location: a per-class binding/unbinding/
    movement heatmap, a sub-pixel (dy, dx) centering offset, and a dipole
    orientation encoded as a unit vector.
    """

    CLASS_BINDING: int = 0
    CLASS_UNBINDING: int = 1
    CLASS_MOVEMENT: int = 2

    def __init__(
        self,
        n_channels: int = 1,
        feat_channels: int = 3,
        min_channels: int = 8,
        trilinear: bool = False,
    ) -> None:
        """Initialize the event detector.

        Args:
            n_channels: Number of input channels.
            feat_channels: Number of channels in the backbone's output, i.e.
                `backbone.n_classes`.
            min_channels: Minimum number of channels in the backbone.
            trilinear: Whether to use trilinear interpolation for upsampling.

        Raises:
            ValueError: If `feat_channels` doesn't match `backbone.n_classes`.
        """
        super().__init__()
        self.backbone = CustomUNet(
            n_channels=n_channels,
            n_classes=feat_channels,
            min_channels=min_channels,
            trilinear=trilinear,
        )

        if feat_channels != self.backbone.n_classes:
            raise ValueError(
                f"feat_channels ({feat_channels}) must match backbone.n_classes "
                f"({self.backbone.n_classes})"
            )

        # 3 independent channels, one per class — sigmoid'd independently, not softmax
        self.heatmap_head = Conv2Plus1D(
            in_channels=feat_channels,
            mid_channels=16,
            out_channels=3,
            temporal_kernel_size=1,
            temporal_padding=0,
            bias=True,
        )

        # shared across all 3 classes — sub-pixel centering doesn't depend on class
        self.offset_head = Conv2Plus1D(
            in_channels=feat_channels,
            mid_channels=16,
            out_channels=2,  # dy, dx
            temporal_kernel_size=1,
            temporal_padding=0,
            bias=True,
        )

        # computed everywhere, but only ever supervised/read at dipole locations
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
        # pi derived from simulator.event_generator at OPTIMUM_EVENT_DENSITY
        # (0.5 events/um^2/s) on a (500, 64, 64) movie, cropped to (490, 64,
        # 64) by gen_data's navg=5 dead-zone crop: 37 binding, 37 unbinding,
        # 29 movement events, out of 2,007,040 voxels/channel.
        pi = torch.tensor([37 / 2_007_040, 37 / 2_007_040, 29 / 2_007_040])
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
                "heatmap": Per-class logits, shape (N, 3, T, H, W). Apply a
                    sigmoid per channel (not softmax) — classes aren't mutually
                    exclusive.
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
