"""
Evaluation metrics for probing experiments

This module provides evaluation metrics for both regression
and classification tasks in the context of probing the SSL vision models
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    r2_score,
)
import logging

logger = logging.getLogger(__name__)


def compute_regression_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, return_per_dim: bool = False
) -> Dict[str, float]:
    """
    Compute regression metrics

    Args:
        predictions: Predicted values [batch_size, output_dim]
        targets: Target values [batch_size, output_dim]
        return_per_dim: If True, return metrics for each output dimension

    Returns:
        Dictionary of metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    metrics = {}

    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    metrics["mae"] = mae

    # Median Absolute Error
    medae = np.median(np.abs(predictions - targets))
    metrics["medae"] = medae

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    metrics["rmse"] = rmse

    # R-squared
    r2 = r2_score(targets, predictions, multioutput="uniform_average")
    metrics["r2"] = r2

    # Mean Absolute Percentage Error
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        if np.isfinite(mape):
            metrics["mape"] = mape

    # calculate metrics for each dimension when we have multiple output dims
    if return_per_dim and predictions.shape[1] > 1:
        for dim in range(predictions.shape[1]):
            dim_pred = predictions[:, dim]
            dim_target = targets[:, dim]

            metrics[f"mae_dim_{dim}"] = np.mean(np.abs(dim_pred - dim_target))
            metrics[f"rmse_dim_{dim}"] = np.sqrt(np.mean((dim_pred - dim_target) ** 2))
            metrics[f"r2_dim_{dim}"] = r2_score(dim_target, dim_pred)

    return metrics


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics

    Args:
        predictions: Predicted logits or probabilities [batch_size, num_classes]
        targets: Target class indices [batch_size]
        num_classes: Number of classes (inferred if not provided)
        class_names: Names of classes for detailed reporting

    Returns:
        Dictionary of metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Convert predictions to class indices if they are probabilities/logits
    if predictions.ndim == 2:
        predicted_classes = np.argmax(predictions, axis=1)
    else:
        predicted_classes = predictions

    metrics = {}

    # Accuracy
    accuracy = accuracy_score(targets, predicted_classes)
    metrics["accuracy"] = accuracy

    # F1 scores
    f1_macro = f1_score(targets, predicted_classes, average="macro", zero_division=0)
    f1_micro = f1_score(targets, predicted_classes, average="micro", zero_division=0)
    f1_weighted = f1_score(
        targets, predicted_classes, average="weighted", zero_division=0
    )

    metrics.update(
        {"f1_macro": f1_macro, "f1_micro": f1_micro, "f1_weighted": f1_weighted}
    )

    # Per-class metrics
    if num_classes is None:
        num_classes = max(np.max(targets), np.max(predicted_classes)) + 1

    f1_per_class = f1_score(
        targets,
        predicted_classes,
        average=None,
        labels=range(num_classes),
        zero_division=0,
    )

    for i, f1 in enumerate(f1_per_class):
        class_name = (
            class_names[i] if class_names and i < len(class_names) else f"class_{i}"
        )
        metrics[f"f1_{class_name}"] = f1

    # Confusion matrix statistics
    cm = confusion_matrix(targets, predicted_classes, labels=range(num_classes))

    # Top-k accuracy
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        for k in [3, 5]:
            if k < predictions.shape[1]:
                top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
                top_k_acc = np.mean(
                    [targets[i] in top_k_preds[i] for i in range(len(targets))]
                )
                metrics[f"top_{k}_accuracy"] = top_k_acc

    return metrics


def compute_voxel_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Computes Intersection over Union (IoU) for 3D voxel grids.

    Args:
        predictions: Predicted voxel logits [Batch, 1, D, H, W] or [Batch, D, H, W].
        targets: Ground truth voxel occupancy (binary) [Batch, 1, D, H, W] or [Batch, D, H, W].
        threshold: Threshold to binarize predictions.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Mean IoU over the batch.
    """
    if predictions.dim() == 5 and predictions.shape[1] == 1:
        predictions = predictions.squeeze(1)  # [B, D, H, W]
    if targets.dim() == 5 and targets.shape[1] == 1:
        targets = targets.squeeze(1)  # [B, D, H, W]

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} must match target shape {targets.shape}"
        )

    pred_binary = (torch.sigmoid(predictions) > threshold).byte()
    targets_binary = targets.byte()

    pred_flat = pred_binary.view(pred_binary.shape[0], -1)
    targets_flat = targets_binary.view(targets_binary.shape[0], -1)

    intersection = (pred_flat & targets_flat).sum(dim=1).float()
    union = (pred_flat | targets_flat).sum(dim=1).float()

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

    Args:
        predictions: Predicted voxel logits [Batch, 1, D, H, W] or [Batch, D, H, W].
        targets: Ground truth voxel occupancy (binary) [Batch, 1, D, H, W] or [Batch, D, H, W].
        threshold: Threshold to binarize predictions.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dictionary containing mean Precision, Recall, and F1-score over the batch.
    """
    if predictions.dim() == 5 and predictions.shape[1] == 1:
        predictions = predictions.squeeze(1)
    if targets.dim() == 5 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} must match target shape {targets.shape}"
        )

    pred_binary = (torch.sigmoid(predictions) > threshold).byte()
    targets_binary = targets.byte()

    pred_flat = pred_binary.view(pred_binary.shape[0], -1)
    targets_flat = targets_binary.view(targets_binary.shape[0], -1)

    tp = (pred_flat & targets_flat).sum(dim=1).float()
    fp = (pred_flat & ~targets_flat).sum(dim=1).float()
    fn = (~pred_flat & targets_flat).sum(dim=1).float()

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
    Computes all relevant voxel metrics (IoU, Precision, Recall, F1).

    Args:
        predictions: Predicted voxel logits [Batch, 1, D, H, W] or [Batch, D, H, W].
        targets: Ground truth voxel occupancy (binary) [Batch, 1, D, H, W] or [Batch, D, H, W].
        threshold: Threshold to binarize predictions.
        smooth: Smoothing factor.

    Returns:
        Dictionary of all voxel metrics.
    """
    metrics = {}
    metrics["voxel_iou"] = compute_voxel_iou(predictions, targets, threshold, smooth)
    metrics.update(
        compute_voxel_precision_recall_f1(predictions, targets, threshold, smooth)
    )
    return metrics


def compute_viewpoint_specific_metrics(
    azimuth_pred: torch.Tensor,
    elevation_pred: torch.Tensor,
    azimuth_target: torch.Tensor,
    elevation_target: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute viewpoint-specific metrics for 3D object pose estimation

    Args:
        azimuth_pred: Predicted azimuth angles in degrees
        elevation_pred: Predicted elevation angles in degrees
        azimuth_target: Target azimuth angles in degrees
        elevation_target: Target elevation angles in degrees

    Returns:
        Dictionary of viewpoint-specific metrics
    """
    if isinstance(azimuth_pred, torch.Tensor):
        azimuth_pred = azimuth_pred.cpu().numpy()
    if isinstance(elevation_pred, torch.Tensor):
        elevation_pred = elevation_pred.cpu().numpy()
    if isinstance(azimuth_target, torch.Tensor):
        azimuth_target = azimuth_target.cpu().numpy()
    if isinstance(elevation_target, torch.Tensor):
        elevation_target = elevation_target.cpu().numpy()

    metrics = {}

    # Angular errors
    azimuth_error = np.abs(azimuth_pred - azimuth_target)
    elevation_error = np.abs(elevation_pred - elevation_target)

    # Handle azimuth wraparound (0-360 degrees)
    azimuth_error = np.minimum(azimuth_error, 360 - azimuth_error)

    # Individual angle metrics
    metrics.update(
        {
            "azimuth_mae": np.mean(azimuth_error),
            "azimuth_medae": np.median(azimuth_error),
            "elevation_mae": np.mean(elevation_error),
            "elevation_medae": np.median(elevation_error),
        }
    )

    # Combined angular distance
    angular_distance = np.sqrt(azimuth_error**2 + elevation_error**2)
    metrics.update(
        {
            "angular_distance_mean": np.mean(angular_distance),
            "angular_distance_median": np.median(angular_distance),
        }
    )

    # Accuracy at different thresholds
    for threshold in [15, 30, 45]:
        azimuth_acc = np.mean(azimuth_error <= threshold)
        elevation_acc = np.mean(elevation_error <= threshold)
        combined_acc = np.mean(
            (azimuth_error <= threshold) & (elevation_error <= threshold)
        )

        metrics.update(
            {
                f"azimuth_acc_{threshold}deg": azimuth_acc,
                f"elevation_acc_{threshold}deg": elevation_acc,
                f"combined_acc_{threshold}deg": combined_acc,
            }
        )

    return metrics


def analyze_error_patterns(
    predictions: np.ndarray,
    targets: np.ndarray,
    categories: Optional[np.ndarray] = None,
    category_names: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Analyze error patterns across different object categories

    Args:
        predictions: Predicted values
        targets: Target values
        categories: Category labels for each sample
        category_names: Names of categories

    Returns:
        Dictionary with error analysis
    """
    analysis = {}

    # Overall error statistics
    errors = np.abs(predictions - targets)
    analysis["overall"] = {
        "mean_error": np.mean(errors),
        "std_error": np.std(errors),
        "min_error": np.min(errors),
        "max_error": np.max(errors),
        "percentiles": {
            "25th": np.percentile(errors, 25),
            "50th": np.percentile(errors, 50),
            "75th": np.percentile(errors, 75),
            "90th": np.percentile(errors, 90),
            "95th": np.percentile(errors, 95),
        },
    }

    # Per-category analysis
    if categories is not None:
        unique_categories = np.unique(categories)
        analysis["per_category"] = {}

        for cat_id in unique_categories:
            mask = categories == cat_id
            cat_errors = errors[mask]

            cat_name = (
                category_names[cat_id]
                if category_names and cat_id < len(category_names)
                else f"category_{cat_id}"
            )

            analysis["per_category"][cat_name] = {
                "mean_error": np.mean(cat_errors),
                "std_error": np.std(cat_errors),
                "num_samples": np.sum(mask),
                "median_error": np.median(cat_errors),
            }

    return analysis


class MetricsTracker:
    """Track metrics across training epochs"""

    def __init__(self):
        self.history = {"train": [], "val": [], "test": []}
        self.best_metrics = {}
        self.best_epoch = 0

    def update(self, split: str, metrics: Dict[str, float], epoch: int):
        """Update metrics for a given split"""
        metrics_with_epoch = {"epoch": epoch, **metrics}
        self.history[split].append(metrics_with_epoch)

        # Track best validation metrics
        if split == "val":
            # Use validation loss or primary metric for early stopping
            primary_metric = metrics.get(
                "loss", metrics.get("mae", metrics.get("accuracy", 0))
            )

            if not self.best_metrics or self._is_better(metrics, self.best_metrics):
                self.best_metrics = metrics.copy()
                self.best_epoch = epoch

    def _is_better(self, current: Dict[str, float], best: Dict[str, float]) -> bool:
        """Check if current metrics are better than best"""
        # loss-like metrics (lower is better)
        if "loss" in current:
            return current["loss"] < best.get("loss", float("inf"))

        # mae (lower is better)
        if "mae" in current:
            return current["mae"] < best.get("mae", float("inf"))

        # accuracy (higher is better)
        if "accuracy" in current:
            return current["accuracy"] > best.get("accuracy", 0)

        # l bozo: use loss
        return current.get("loss", float("inf")) < best.get("loss", float("inf"))

    def get_best_metrics(self) -> Tuple[Dict[str, float], int]:
        """Get best validation metrics and epoch"""
        return self.best_metrics, self.best_epoch

    def get_history(self, split: str) -> List[Dict[str, float]]:
        """Get training history for a split"""
        return self.history.get(split, [])

    def should_stop_early(self, patience: int, min_epochs: int = 10) -> bool:
        """Check if training should stop early"""
        if len(self.history["val"]) < min_epochs:
            return False

        current_epoch = self.history["val"][-1]["epoch"]
        return (current_epoch - self.best_epoch) >= patience
