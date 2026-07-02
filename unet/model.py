"""Full assembly of the U-Net parts into the complete network."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """Encoder-decoder segmentation network with skip connections."""

    def __init__(
        self, n_channels: int, n_classes: int, bilinear: bool = False
    ) -> None:
        """Initialize the U-Net.

        Args:
            n_channels: Number of channels in the input image.
            n_classes: Number of output classes/channels.
            bilinear: If True, use bilinear upsampling in the decoder.
                If False, use learned transposed convolutions.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder-decoder forward pass.

        Args:
            x: Input tensor of shape (N, n_channels, H, W).

        Returns:
            Per-class logits of shape (N, n_classes, H, W).
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
