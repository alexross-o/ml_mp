import torch

def loss_fn_heatmap(
    heatmap_pred: torch.Tensor,
    heatmap_gt: torch.Tensor,
) -> float:
    return torch.inf


def loss_fn_offset(
    offset_pred: torch.Tensor,
    offset_gt: torch.Tensor,
) -> float:
    """L1 loss for this
    """
    return torch.inf


def loss_fn_orientation(
    orientation_pred: torch.Tensor,
    orientation_gt: torch.Tensor,
) -> float:
    return torch.inf


def loss_fn(
    predictions: dict[str, torch.Tensor],
    ground_truth: dict[str, torch.Tensor],
    lambda_offset: float = 1.0,
    lambda_orientation: float = 1.0,
) -> float:
    """Loss function relative to ground truth simulations

    Args:
        predictions: ...
        ground_truth: ...
        lambda_offset: weight of offset loss in the loss function
        lambda_orientation: weight of orientation loss in the loss function

    Returns:
        loss value
    """

    loss_heatmap = loss_fn_heatmap(predictions["heatmap"], ground_truth["heatmap"])
    loss_offset = loss_fn_offset(predictions["offset"], ground_truth["offset"])
    loss_orientation = loss_fn_orientation(
        predictions["orientation"], ground_truth["orientation"]
    )

    return (
        loss_heatmap
        + lambda_offset * loss_offset
        + lambda_orientation * loss_orientation
    )
