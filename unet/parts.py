"""Building blocks of the U-Net architecture: DoubleConv, Down, Up, OutConv."""

import torch
import torch.nn as nn
from torch.nn import functional


class DoubleConv(nn.Module):
    """Two consecutive (convolution -> BatchNorm -> ReLU) blocks.

    The two convolutions can use independent dilation rates, which allows
    this block to also serve as a dilated/atrous convolution stage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation1: int = 1,
        dilation2: int = 1,
        mid_channels: int | None = None,
    ) -> None:
        """Initialize the double convolution block.

        Args:
            in_channels: Number of channels in the input tensor.
            out_channels: Number of channels produced by the block.
            dilation1: Dilation rate for the first convolution.
            dilation2: Dilation rate for the second convolution.
            mid_channels: Number of channels between the two convolutions.
                Defaults to `out_channels` when not given.
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                dilation=dilation1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                dilation=dilation2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two convolution blocks.

        Args:
            x: Input tensor of shape (N, in_channels, H, W).

        Returns:
            Output tensor of shape (N, out_channels, H, W).
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: max-pool followed by a `DoubleConv`."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the downscaling block.

        Args:
            in_channels: Number of channels in the input tensor.
            out_channels: Number of channels produced by the block.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Halve the spatial resolution and apply a double convolution.

        Args:
            x: Input tensor of shape (N, in_channels, H, W).

        Returns:
            Output tensor of shape (N, out_channels, H // 2, W // 2).
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block: upsample, concatenate skip connection, `DoubleConv`."""

    def __init__(
        self, in_channels: int, out_channels: int, bilinear: bool = True
    ) -> None:
        """Initialize the upscaling block.

        Args:
            in_channels: Number of channels in the concatenated input
                (decoder feature map + skip connection).
            out_channels: Number of channels produced by the block.
            bilinear: If True, upsample with bilinear interpolation followed
                by a channel-reducing convolution. If False, upsample with a
                learned transposed convolution.
        """
        super().__init__()
        self.up: nn.Module

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Upsample `x1`, align it to `x2`, and merge via concatenation.

        Args:
            x1: Decoder feature map to be upsampled, shape (N, C1, H1, W1).
            x2: Encoder skip-connection feature map to concatenate with,
                shape (N, C2, H2, W2).

        Returns:
            Output tensor produced by the double convolution over the
            concatenated feature maps.
        """
        x1 = self.up(x1)
        # Input is (N, C, H, W); pad x1 so its spatial size matches x2.
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = functional.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        # If you have padding issues, see:
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping features to per-class logits."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the output convolution.

        Args:
            in_channels: Number of channels in the input tensor.
            out_channels: Number of output classes/channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 1x1 convolution.

        Args:
            x: Input tensor of shape (N, in_channels, H, W).

        Returns:
            Output tensor of shape (N, out_channels, H, W).
        """
        return self.conv(x)
