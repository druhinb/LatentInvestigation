"""
Viewpoint-specific metrics for 3D object pose estimation and error analysis.
"""
import numpy as np
from typing import Dict, List, Optional
import torch


def compute_viewpoint_specific_metrics(
    azimuth_pred: torch.Tensor,
    elevation_pred: torch.Tensor,
    azimuth_target: torch.Tensor,
    elevation_target: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute viewpoint-specific metrics for 3D object pose estimation.

    Args:
        azimuth_pred: Predicted azimuth angles in degrees [batch]
        elevation_pred: Predicted elevation angles in degrees [batch]
        azimuth_target: Target azimuth angles in degrees [batch]
        elevation_target: Target elevation angles in degrees [batch]

    Returns:
        Dictionary of metrics including MAE, MedAE, angular distance, and accuracy thresholds.
    """
    # Convert to numpy
    az_p = azimuth_pred.detach().cpu().numpy() if isinstance(azimuth_pred, torch.Tensor) else np.array(azimuth_pred)
    el_p = elevation_pred.detach().cpu().numpy() if isinstance(elevation_pred, torch.Tensor) else np.array(elevation_pred)
    az_t = azimuth_target.detach().cpu().numpy() if isinstance(azimuth_target, torch.Tensor) else np.array(azimuth_target)
    el_t = elevation_target.detach().cpu().numpy() if isinstance(elevation_target, torch.Tensor) else np.array(elevation_target)

    metrics: Dict[str, float] = {}

    # Angular errors
    az_err = np.abs(az_p - az_t)
    el_err = np.abs(el_p - el_t)
    # Handle wrap-around for azimuth
    az_err = np.minimum(az_err, 360.0 - az_err)

    metrics['azimuth_mae'] = np.mean(az_err)
    metrics['azimuth_medae'] = np.median(az_err)
    metrics['elevation_mae'] = np.mean(el_err)
    metrics['elevation_medae'] = np.median(el_err)

    # Angular distance
    ang_dist = np.sqrt(az_err**2 + el_err**2)
    metrics['angular_distance_mean'] = np.mean(ang_dist)
    metrics['angular_distance_median'] = np.median(ang_dist)

    # Accuracy thresholds
    for thr in [15, 30, 45]:
        metrics[f'azimuth_acc_{thr}deg'] = np.mean(az_err <= thr)
        metrics[f'elevation_acc_{thr}deg'] = np.mean(el_err <= thr)
        metrics[f'combined_acc_{thr}deg'] = np.mean((az_err <= thr) & (el_err <= thr))

    return metrics


def analyze_error_patterns(
    predictions: np.ndarray,
    targets: np.ndarray,
    categories: Optional[np.ndarray] = None,
    category_names: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Analyze error patterns across different object categories.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        categories: Category labels per sample
        category_names: Optional list of category names

    Returns:
        Dictionary with overall and per-category error analysis.
    """
    analysis: Dict[str, any] = {}
    errors = np.abs(predictions - targets)
    # Overall stats
    analysis['overall'] = {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'min_error': np.min(errors),
        'max_error': np.max(errors),
        'percentiles': {
            '25th': np.percentile(errors, 25),
            '50th': np.percentile(errors, 50),
            '75th': np.percentile(errors, 75),
            '90th': np.percentile(errors, 90),
            '95th': np.percentile(errors, 95),
        }
    }
    # Per-category
    if categories is not None:
        analysis['per_category'] = {}
        unique = np.unique(categories)
        for cat in unique:
            mask = (categories == cat)
            cat_errors = errors[mask]
            name = category_names[cat] if category_names and cat < len(category_names) else f'category_{cat}'
            analysis['per_category'][name] = {
                'mean_error': np.mean(cat_errors),
                'std_error': np.std(cat_errors),
                'num_samples': int(mask.sum()),
                'median_error': np.median(cat_errors),
            }
    return analysis
