"""
Regression metrics computations for probing experiments.
"""
<<<<<<< HEAD

=======
>>>>>>> 9b54dff6f6376eb3d35334ce60964369eda4a2c7
import numpy as np
import torch
from sklearn.metrics import r2_score


def compute_regression_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, return_per_dim: bool = False
) -> dict:
    """
    Compute regression metrics: MAE, Median AE, RMSE, R2, MAPE, and per-dimension metrics.
    """
<<<<<<< HEAD
    preds = (
        predictions.detach().cpu().numpy()
        if isinstance(predictions, torch.Tensor)
        else predictions
    )
    targs = (
        targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets
    )

    metrics = {}
    mae = np.mean(np.abs(preds - targs))
    metrics["mae"] = mae
    medae = np.median(np.abs(preds - targs))
    metrics["medae"] = medae
    rmse = np.sqrt(np.mean((preds - targs) ** 2))
    metrics["rmse"] = rmse
    metrics["r2"] = r2_score(targs, preds, multioutput="uniform_average")

    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((targs - preds) / targs)) * 100
        if np.isfinite(mape):
            metrics["mape"] = mape
=======
    preds = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    targs = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    metrics = {}
    mae = np.mean(np.abs(preds - targs))
    metrics['mae'] = mae
    medae = np.median(np.abs(preds - targs))
    metrics['medae'] = medae
    rmse = np.sqrt(np.mean((preds - targs) ** 2))
    metrics['rmse'] = rmse
    metrics['r2'] = r2_score(targs, preds, multioutput='uniform_average')

    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((targs - preds) / targs)) * 100
        if np.isfinite(mape):
            metrics['mape'] = mape
>>>>>>> 9b54dff6f6376eb3d35334ce60964369eda4a2c7

    if return_per_dim and preds.ndim == 2 and preds.shape[1] > 1:
        for dim in range(preds.shape[1]):
            pred_d = preds[:, dim]
            targ_d = targs[:, dim]
<<<<<<< HEAD
            metrics[f"mae_dim_{dim}"] = np.mean(np.abs(pred_d - targ_d))
            metrics[f"r2_dim_{dim}"] = r2_score(targ_d, pred_d)
            metrics[f"rmse_dim_{dim}"] = np.sqrt(np.mean((pred_d - targ_d) ** 2))
=======
            metrics[f'mae_dim_{dim}'] = np.mean(np.abs(pred_d - targ_d))
            metrics[f'r2_dim_{dim}'] = r2_score(targ_d, pred_d)
            metrics[f'rmse_dim_{dim}'] = np.sqrt(np.mean((pred_d - targ_d) ** 2))
>>>>>>> 9b54dff6f6376eb3d35334ce60964369eda4a2c7
    return metrics
