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

    def __init__(self, backbone: CustomUNet, feat_channels: int) -> None:
        """Initialize the event detector.

        Args:
            backbone: Feature-extracting `CustomUNet` producing `feat_channels`
                channels of output.
            feat_channels: Number of channels in the backbone's output, i.e.
                `backbone.n_classes`.

        Raises:
            ValueError: If `feat_channels` doesn't match `backbone.n_classes`.
        """
        super().__init__()
        self.backbone = backbone

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

        # TODO: adjust to real class frequencies once data is simulated
        # # per-class bias init — don't assume black lobe, white lobe, and dipole
        # # occur at the same rate. Set pi per channel from your simulator's actual
        # # class frequencies rather than broadcasting one scalar to all 3.
        # pi = torch.tensor([0.01, 0.01, 0.005])   # example: dipoles rarer than singles
        # bias_init = -torch.log((1 - pi) / pi)
        # with torch.no_grad():
        #     self.heatmap_head.temporal_conv.bias.copy_(bias_init)

        # nn.init.zeros_(self.offset_head.temporal_conv.weight)
        # nn.init.zeros_(self.offset_head.temporal_conv.bias)

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
