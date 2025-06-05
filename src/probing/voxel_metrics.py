"""
Voxel-based metrics: IoU, precision, recall, F1 for 3D voxel grids.
"""

import torch
from typing import Dict


def compute_voxel_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Computes Intersection over Union (IoU) for 3D voxel grids.
    """
    if predictions.dim() == 5 and predictions.shape[1] == 1:
        predictions = predictions.squeeze(1)
    if targets.dim() == 5 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} must match target shape {targets.shape}"
        )
    pred_binary = torch.sigmoid(predictions) > threshold
    target_binary = targets.bool()
    pred_flat = pred_binary.view(pred_binary.shape[0], -1)
    target_flat = target_binary.view(target_binary.shape[0], -1)

    intersection = (pred_flat & target_flat).sum(dim=1).float()
    union = (pred_flat | target_flat).sum(dim=1).float()
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def compute_voxel_precision_recall_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1-score for 3D voxel grids.
    """
    if predictions.dim() == 5 and predictions.shape[1] == 1:
        predictions = predictions.squeeze(1)
    if targets.dim() == 5 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} must match target shape {targets.shape}"
        )
    pred_binary = torch.sigmoid(predictions) > threshold
    target_binary = targets.bool()
    pred_flat = pred_binary.view(pred_binary.shape[0], -1)
    target_flat = target_binary.view(target_binary.shape[0], -1)

    tp = (pred_flat & target_flat).sum(dim=1).float()
    fp = (pred_flat & ~target_flat).sum(dim=1).float()
    fn = (~pred_flat & target_flat).sum(dim=1).float()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)

    return {
        "voxel_precision": precision.mean().item(),
        "voxel_recall": recall.mean().item(),
        "voxel_f1": f1.mean().item(),
    }


def compute_voxel_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes all relevant voxel metrics: IoU, Precision, Recall, F1.
    """
    metrics = {}
    metrics["voxel_iou"] = compute_voxel_iou(predictions, targets, threshold, smooth)
    metrics.update(
        compute_voxel_precision_recall_f1(predictions, targets, threshold, smooth)
    )
    metrics.update(
        compute_voxel_precision_recall_f1(predictions, targets, threshold, smooth)
    )
    return metrics
