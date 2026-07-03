"""Full assembly of the U-Net parts into the complete network."""

import torch
import torch.nn as nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from .unet_parts import Conv2Plus1D, DoubleConv, Down, OutConv, Up


class CustomUNet(nn.Module):
    """Encoder-decoder segmentation network with skip connections."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        min_channels: int = 8,
        trilinear: bool = False,
    ) -> None:
        """Initialize the U-Net.

        Args:
            n_channels: Number of channels in the input image.
            n_classes: Number of output classes/channels.
            min_channels: Number of channels in the first convolutional layer.
                The number of channels doubles after each downsampling step.
            trilinear: If True, use trilinear upsampling in the decoder.
                If False, use learned transposed convolutions.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        # PSF is about 12-16 px in width, so 4 2x downsamples
        # is about right to create one hot pixel
        self.inc = DoubleConv(
            n_channels,
            min_channels,
            kernel_size_1=5,  # larger kernel + dilation to learn/detect large features
            dilation_1=2,
            padding_1=4,
            kernel_size_2=5,  # footprint shrinks to 5 with no dilation
            padding_2=2,
        )
        self.down1 = Down(min_channels, min_channels * 2)
        self.down2 = Down(min_channels * 2, min_channels * 4)
        self.down3 = Down(min_channels * 4, min_channels * 8)
        factor = 2 if trilinear else 1
        self.down4 = Down(min_channels * 8, min_channels * 16 // factor)
        self.up1 = Up(min_channels * 16, min_channels * 8 // factor, trilinear)
        self.up2 = Up(min_channels * 8, min_channels * 4 // factor, trilinear)
        self.up3 = Up(min_channels * 4, min_channels * 2 // factor, trilinear)
        self.up4 = Up(min_channels * 2, min_channels, trilinear)
        self.outc = OutConv(min_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder-decoder forward pass.

        Args:
            x: Input tensor of shape (N, n_channels, T, H, W).

        Returns:
            Per-class logits of shape (N, n_classes, T, H, W).
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self) -> None:
        """Wrap each stage in gradient checkpointing to reduce training memory.

        Trades extra forward-pass compute during backprop for lower peak
        memory usage, since intermediate activations are recomputed instead
        of stored.
        """
        self.inc = checkpoint(self.inc)
        self.down1 = checkpoint(self.down1)
        self.down2 = checkpoint(self.down2)
        self.down3 = checkpoint(self.down3)
        self.down4 = checkpoint(self.down4)
        self.up1 = checkpoint(self.up1)
        self.up2 = checkpoint(self.up2)
        self.up3 = checkpoint(self.up3)
        self.up4 = checkpoint(self.up4)
        self.outc = checkpoint(self.outc)


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
        """
        super().__init__()
        self.backbone = backbone

        # 3 independent channels, one per class — sigmoid'd independently, not softmax
        self.heatmap_head = Conv2Plus1D(
            in_channels=feat_channels,
            out_channels=3,
            temporal_kernel_size=1,
            temporal_padding=0,
            bias=True,
        )

        # shared across all 3 classes — sub-pixel centering doesn't depend on class
        self.offset_head = Conv2Plus1D(
            in_channels=feat_channels,
            out_channels=2,  # dy, dx
            temporal_kernel_size=1,
            temporal_padding=0,
            bias=True,
        )

        # computed everywhere, but only ever supervised/read at dipole locations
        self.orientation_head = Conv2Plus1D(
            in_channels=feat_channels,
            out_channels=2,  # cos, sin — L2-normalized to a unit vector in forward()
            temporal_kernel_size=1,
            temporal_padding=0,
            bias=True,
        )

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
