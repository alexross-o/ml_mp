"""Loss functions for the heatmap/offset/orientation event detection heads."""

import torch
import torch.nn as nn
from torch.nn import functional
from torch.nn.modules.loss import _Loss
from .model import EventDetector

DEFAULT_NON_HEATMAP_LOSS = nn.MSELoss(reduction="none")


def loss_fn_heatmap(
    heatmap_pred: torch.Tensor,
    heatmap_gt: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalty-reduced focal loss for the per-class event heatmap.

    Ground truth is expected to be a Gaussian-splatted heatmap with an exact
    peak of 1 at true event locations, decaying towards 0 away from it (as in
    CornerNet/CenterNet). Peak voxels are pushed towards 1, non-peak voxels
    are pushed towards 0 with a penalty reduced by their distance from a peak.

    Args:
        heatmap_pred: Predicted per-class heatmap logits, shape (N, C, T, H, W).
        heatmap_gt: Ground truth heatmap in [0, 1], same shape as `heatmap_pred`.
        alpha: Down-weights the loss at voxels the model already predicts
            confidently, focusing training on hard voxels.
        beta: Down-weights the negative-voxel penalty near a peak, since
            ground truth decays smoothly rather than dropping to 0 sharply.

    Returns:
        A tuple of:
            - Scalar focal loss, averaged over the number of true peak voxels.
            - Boolean-valued mask (as float) of true peak voxels (`heatmap_gt == 1`),
              for reuse when masking the offset/orientation losses.
    """
    pred_prob = torch.sigmoid(heatmap_pred)

    pos_mask = (heatmap_gt == 1).float()
    neg_mask = (heatmap_gt < 1).float()

    pos_loss = functional.logsigmoid(heatmap_pred) * (1 - pred_prob) ** alpha * pos_mask
    neg_loss = (
        functional.logsigmoid(-heatmap_pred)
        * pred_prob**alpha
        * (1 - heatmap_gt) ** beta
        * neg_mask
    )

    num_pos = pos_mask.sum().clamp(min=1)
    return (-(pos_loss.sum() + neg_loss.sum()) / num_pos, pos_mask)


def _masked_mean_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: _Loss,
) -> torch.Tensor:
    """Mean of a per-voxel loss, restricted to masked-in locations.

    Args:
        pred: Predicted tensor.
        gt: Ground truth tensor, same shape as `pred`.
        mask: Per-voxel weight (0 or 1), broadcastable against `pred`.
        loss_fn: Elementwise loss module with `reduction="none"`.

    Returns:
        Scalar loss, averaged over the number of masked-in voxels.
    """
    loss_per_voxel = loss_fn(pred, gt)
    num_pos = mask.sum().clamp(min=1)
    return (loss_per_voxel * mask).sum() / num_pos


def loss_fn_offset(
    offset_pred: torch.Tensor,
    offset_gt: torch.Tensor,
    offset_gt_mask: torch.Tensor,
    loss_fn: _Loss,
) -> torch.Tensor:
    """Masked regression loss for the sub-pixel (dy, dx) offset head.

    Args:
        offset_pred: Predicted offset, shape (N, 2, T, H, W).
        offset_gt: Ground truth offset, same shape as `offset_pred`.
        offset_gt_mask: Per-voxel weight (0 or 1) selecting voxels with a true
            event of any class, shape (N, 1, T, H, W).
        loss_fn: Elementwise loss module with `reduction="none"` (e.g. L1/L2).

    Returns:
        Scalar loss, averaged over the number of masked-in voxels.
    """
    return _masked_mean_loss(offset_pred, offset_gt, offset_gt_mask, loss_fn)


def loss_fn_orientation(
    orientation_pred: torch.Tensor,
    orientation_gt: torch.Tensor,
    orientation_gt_mask: torch.Tensor,
    loss_fn: _Loss,
) -> torch.Tensor:
    """Masked regression loss for the dipole (cos, sin) orientation head.

    Args:
        orientation_pred: Predicted orientation unit vector, shape (N, 2, T, H, W).
        orientation_gt: Ground truth orientation, same shape as `orientation_pred`.
        orientation_gt_mask: Per-voxel weight (0 or 1) selecting voxels with a
            true dipole/movement event, shape (N, 1, T, H, W).
        loss_fn: Elementwise loss module with `reduction="none"` (e.g. L1/L2).

    Returns:
        Scalar loss, averaged over the number of masked-in voxels.
    """
    return _masked_mean_loss(
        orientation_pred, orientation_gt, orientation_gt_mask, loss_fn
    )


def loss_fn(
    predictions: dict[str, torch.Tensor],
    ground_truth: dict[str, torch.Tensor],
    alpha_heatmap: float = 2.0,
    beta_heatmap: float = 4.0,
    lambda_offset: float = 1.0,
    lambda_orientation: float = 1.0,
    offset_loss_fn: _Loss = DEFAULT_NON_HEATMAP_LOSS,
    orientation_loss_fn: _Loss = DEFAULT_NON_HEATMAP_LOSS,
    movement_channel: int = EventDetector.CLASS_MOVEMENT,
) -> torch.Tensor:
    """Combined loss for `EventDetector`'s heatmap, offset, and orientation heads.

    Args:
        predictions: Dict with keys "heatmap", "offset", "orientation" as
            returned by `EventDetector.forward`.
        ground_truth: Dict with the same keys/shapes as `predictions`.
        alpha_heatmap: `alpha` passed to `loss_fn_heatmap`.
        beta_heatmap: `beta` passed to `loss_fn_heatmap`.
        lambda_offset: Weight of the offset loss in the combined loss.
        lambda_orientation: Weight of the orientation loss in the combined loss.
        offset_loss_fn: Elementwise loss module for the offset head; must
            have `reduction="none"`.
        orientation_loss_fn: Elementwise loss module for the orientation
            head; must have `reduction="none"`.
        movement_channel: Heatmap channel index corresponding to
            `EventDetector.CLASS_MOVEMENT` (dipole events), used to mask the
            orientation loss to voxels with a true movement event.

    Returns:
        Scalar combined loss: heatmap loss plus the weighted offset and
        orientation losses.

    Raises:
        ValueError: If `offset_loss_fn` or `orientation_loss_fn` don't use
            `reduction="none"` — masking assumes a per-voxel loss.
    """
    if (offset_loss_fn.reduction != "none") or (
        orientation_loss_fn.reduction != "none"
    ):
        raise ValueError('Reduction for loss functions must be "none"')

    loss_heatmap, hm_mask = loss_fn_heatmap(
        predictions["heatmap"],
        ground_truth["heatmap"],
        alpha=alpha_heatmap,
        beta=beta_heatmap,
    )

    offset_mask = hm_mask.any(dim=1, keepdim=True).float()
    loss_offset = loss_fn_offset(
        predictions["offset"], ground_truth["offset"], offset_mask, offset_loss_fn
    )

    orientation_mask = hm_mask[:, movement_channel : movement_channel + 1]
    loss_orientation = loss_fn_orientation(
        predictions["orientation"],
        ground_truth["orientation"],
        orientation_mask,
        orientation_loss_fn,
    )

    return (
        loss_heatmap
        + lambda_offset * loss_offset
        + lambda_orientation * loss_orientation
    )
