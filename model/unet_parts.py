"""Building blocks of the U-Net architecture: Conv2Plus1D, DoubleConv, Down, Up, OutConv."""

import torch
import torch.nn as nn
from torch.nn import functional


def _group_norm_groups(num_channels: int, preferred: int = 8) -> int:
    """Largest group count no greater than `preferred` that divides `num_channels`.

    `nn.GroupNorm` requires `num_channels % num_groups == 0`. Channel counts
    that come from a formula (rather than a fixed architecture choice) aren't
    guaranteed to be a multiple of a hard-coded group count, so this searches
    downward from `preferred` for a valid, stability-favoring group count
    rather than risking a crash or jumping up to a noisier, more granular one.

    Args:
        num_channels: Number of channels to be normalized.
        preferred: Upper bound on the number of groups.

    Returns:
        A group count in `[1, min(preferred, num_channels)]` that evenly
        divides `num_channels`.
    """
    for groups in range(min(preferred, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class Conv2Plus1D(nn.Module):
    """Spatial-then-temporal decomposition of a 3D convolution.

    Factorizes a 3D convolution into a spatial 2D convolution (kernel
    `(1, k, k)`) followed by a temporal 1D convolution (kernel `(t, 1, 1)`),
    as in the R(2+1)D architecture (Tran et al., 2018,
    https://arxiv.org/abs/1711.11248). `mid_channels` is derived from
    `in_channels`/`out_channels` so the factorized block has roughly the
    same parameter count as an equivalent full 3D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        spatial_dilation: int = 1,
        temporal_dilation: int = 1,
        spatial_padding: int = 1,
        temporal_padding: int = 1,
        bias: bool = False,
        mid_channels: int | None = None
    ) -> None:
        """Initialize the (2+1)D convolution block.

        Args:
            in_channels: Number of channels in the input tensor.
            out_channels: Number of channels produced by the block.
            spatial_kernel_size: Kernel size of the spatial (H, W) convolution.
            temporal_kernel_size: Kernel size of the temporal (T) convolution.
            spatial_dilation: Dilation rate of the spatial convolution.
            temporal_dilation: Dilation rate of the temporal convolution.
            spatial_padding: Padding of the spatial convolution.
            temporal_padding: Padding of the temporal convolution.
            bias: Whether the temporal convolution learns a bias term.
                Typically False when followed by a normalization layer.
        """
        super().__init__()

        if mid_channels is None:
            # Floored at 1: for small out_channels (e.g. single-channel heads) the
            # parameter-matching formula can round down to 0, which would build a
            # zero-channel spatial conv.
            mid_channels = max(
                1,
                int(
                    (
                        temporal_kernel_size
                        * spatial_kernel_size**2
                        * in_channels
                        * out_channels
                    )
                    / (
                        spatial_kernel_size**2 * in_channels
                        + temporal_kernel_size * out_channels
                    )
                ),
            )

        self.spatial_conv = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, spatial_kernel_size, spatial_kernel_size),
            padding=(0, spatial_padding, spatial_padding),
            dilation=(1, spatial_dilation, spatial_dilation),
            bias=False,  # immediately followed by a normalization
        )
        self.temporal_conv = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            padding=(temporal_padding, 0, 0),
            dilation=(temporal_dilation, 1, 1),
            bias=bias,
        )

        self.conv2plus1d = nn.Sequential(
            self.spatial_conv,
            nn.GroupNorm(
                num_groups=_group_norm_groups(mid_channels), num_channels=mid_channels
            ),
            nn.ReLU(inplace=True),
            self.temporal_conv,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the spatial convolution followed by the temporal convolution.

        Args:
            x: Input tensor of shape (N, in_channels, T, H, W).

        Returns:
            Output tensor of shape (N, out_channels, T', H', W').
        """
        return self.conv2plus1d(x)


class DoubleConv(nn.Module):
    """Two consecutive (convolution -> BatchNorm -> ReLU) blocks.

    The two convolutions can use independent dilation rates, which allows
    this block to also serve as a dilated/atrous convolution stage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_1: int = 3,
        dilation_1: int = 1,
        padding_1: int = 1,
        kernel_size_2: int = 3,
        dilation_2: int = 1,
        padding_2: int = 1,
        mid_channels: int | None = None,
    ) -> None:
        """Initialize the double convolution block.

        Args:
            in_channels: Number of channels in the input tensor.
            out_channels: Number of channels produced by the block.
            kernel_size_N: Spatial kernel size for convolution 1 or 2. Only
                scopes the spatial half of each `Conv2Plus1D` block; the
                temporal kernel size stays at `Conv2Plus1D`'s default.
            padding_N: Spatial padding for convolution 1 or 2. Only scopes
                the spatial half of each `Conv2Plus1D` block; the temporal
                padding stays at `Conv2Plus1D`'s default.
            dilation_N: Spatial dilation rate for convolution 1 or 2. Only
                scopes the spatial half of each `Conv2Plus1D` block; the
                temporal dilation stays at `Conv2Plus1D`'s default.
            mid_channels: Number of channels between the two convolutions.
                Defaults to `out_channels` when not given.
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Conv2Plus1D(
                in_channels,
                mid_channels,
                spatial_kernel_size=kernel_size_1,
                spatial_padding=padding_1,
                spatial_dilation=dilation_1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=_group_norm_groups(mid_channels), num_channels=mid_channels
            ),
            nn.ReLU(inplace=True),
            Conv2Plus1D(
                mid_channels,
                out_channels,
                spatial_kernel_size=kernel_size_2,
                spatial_padding=padding_2,
                spatial_dilation=dilation_2,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=_group_norm_groups(out_channels), num_channels=out_channels
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two convolution blocks.

        Args:
            x: Input tensor of shape (N, in_channels, T, H, W).

        Returns:
            Output tensor of shape (N, out_channels, T, H, W).
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
            nn.MaxPool3d(kernel_size=(1, 2, 2)), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Halve the spatial resolution and apply a double convolution.

        Args:
            x: Input tensor of shape (N, in_channels, T, H, W).

        Returns:
            Output tensor of shape (N, out_channels, T, H // 2, W // 2).
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block: upsample, concatenate skip connection, `DoubleConv`."""

    def __init__(
        self, in_channels: int, out_channels: int, trilinear: bool = True
    ) -> None:
        """Initialize the upscaling block.

        Args:
            in_channels: Number of channels in the concatenated input
                (decoder feature map + skip connection).
            out_channels: Number of channels produced by the block.
            trilinear: If True, upsample with trilinear interpolation followed
                by a channel-reducing convolution. If False, upsample with a
                learned transposed convolution.
        """
        super().__init__()
        self.up: nn.Module

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(
                scale_factor=(1, 2, 2), mode="trilinear", align_corners=True
            )
            self.conv = DoubleConv(
                in_channels, out_channels, mid_channels=in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2)
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Upsample `x1`, align it to `x2`, and merge via concatenation.

        Args:
            x1: Decoder feature map to be upsampled, shape (N, C1, T1, H1, W1).
            x2: Encoder skip-connection feature map to concatenate with,
                shape (N, C2, T2, H2, W2).

        Returns:
            Output tensor produced by the double convolution over the
            concatenated feature maps.
        """
        x1 = self.up(x1)
        # Input is (N, C, T, H, W); pad x1 so its spatial size matches x2.
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]

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
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 1x1 convolution.

        Args:
            x: Input tensor of shape (N, in_channels, T, H, W).

        Returns:
            Output tensor of shape (N, out_channels, T, H, W).
        """
        return self.conv(x)
