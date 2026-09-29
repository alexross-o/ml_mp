"""Ground truth heatmap/offset/orientation targets, compatible with `model.loss.loss_fn`."""

from typing import Sequence

import numpy as np
import torch

from model.model import EventDetector
from simulator.constants import HEATMAP_GAUSSIAN_SIGMA_PX, HEATMAP_GAUSSIAN_THUMBNAIL_SIZE
from simulator.events import (
    AbstractSimEvent,
    BindingSimEvent,
    MovementSimEvent,
    UnbindingSimEvent,
)


def gen_ground_truth(
    events: Sequence[AbstractSimEvent],
    mov_shape: tuple[int, int, int],
    sigma_px: float = HEATMAP_GAUSSIAN_SIGMA_PX,
) -> dict[str, torch.Tensor]:
    """Render simulated events into ground truth compatible with `loss_fn`.

    For each event, adds a small Gaussian thumbnail (peak 1, truncated to
    `HEATMAP_GAUSSIAN_THUMBNAIL_SIZE` px) onto its class's heatmap centered
    at (frame, hot_px) -- the CornerNet/CenterNet-style target
    `loss_fn_heatmap` expects -- and records its sub-pixel (dy, dx) offset at
    that same peak voxel. Movement events additionally record their (cos,
    sin) orientation there. Overlapping events of the same class are summed,
    then the heatmap is clipped to [0, 1] so nearby peaks still saturate to
    1 rather than exceeding it.

    Args:
        events: Simulated events, as returned by `gen_events`.
        mov_shape: Movie array shape (T, H, W) the events were placed within.
        sigma_px: Standard deviation, in px, of the heatmap Gaussian.

    Returns:
        Dict with keys "heatmap" (3, T, H, W), "offset" (2, T, H, W), and
        "orientation" (2, T, H, W) -- matching `predictions` from
        `EventDetector.forward` up to the batch dimension.

    Raises:
        TypeError: If `events` contains a type other than `BindingSimEvent`,
            `UnbindingSimEvent`, or `MovementSimEvent`.
    """
    t_size, h_size, w_size = mov_shape

    heatmap = np.zeros((3, t_size, h_size, w_size), dtype=np.float32)
    offset = np.zeros((2, t_size, h_size, w_size), dtype=np.float32)
    orientation = np.zeros((2, t_size, h_size, w_size), dtype=np.float32)

    half = HEATMAP_GAUSSIAN_THUMBNAIL_SIZE // 2
    patch_yy, patch_xx = np.mgrid[-half : half + 1, -half : half + 1]
    gaussian_patch = np.exp(-(patch_xx**2 + patch_yy**2) / (2 * sigma_px**2))

    for event in events:
        if isinstance(event, BindingSimEvent):
            channel = EventDetector.CLASS_BINDING
        elif isinstance(event, UnbindingSimEvent):
            channel = EventDetector.CLASS_UNBINDING
        elif isinstance(event, MovementSimEvent):
            channel = EventDetector.CLASS_MOVEMENT
        else:
            raise TypeError(f"unrecognized event type: {type(event).__name__}")

        frame = int(np.clip(np.round(event.i), 0, t_size - 1))
        x_px, y_px = event.hot_px
        dx, dy = event.offset

        heatmap[
            channel,
            frame,
            y_px - half : y_px + half + 1,
            x_px - half : x_px + half + 1,
        ] += gaussian_patch

        # last write wins: if another event already claimed this (frame, hot_px)
        # voxel -- same class or not -- its offset/orientation is silently
        # overwritten rather than averaged. Unlike the heatmap, a single voxel
        # can't represent two distinct sub-pixel positions/angles, and with
        # continuous-valued placement over a large voxel grid this is rare
        # enough not to be worth tracking/averaging.
        offset[:, frame, y_px, x_px] = (dy, dx)

        if isinstance(event, MovementSimEvent):
            orientation[:, frame, y_px, x_px] = (np.cos(event.theta), np.sin(event.theta))

    heatmap = np.clip(heatmap, 0.0, 1.0)

    return {
        "heatmap": torch.from_numpy(heatmap),
        "offset": torch.from_numpy(offset),
        "orientation": torch.from_numpy(orientation),
    }
