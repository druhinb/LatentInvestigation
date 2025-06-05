"""
Classification metrics computations for probing experiments.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import List, Optional, Dict


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics (accuracy, F1 scores, top-k accuracy).
    """
    preds_np = (
        predictions.detach().cpu().numpy()
        if isinstance(predictions, torch.Tensor)
        else predictions
    )
    targs_np = (
        targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets
    )

    # Determine predicted classes
    if preds_np.ndim == 2:
        pred_classes = np.argmax(preds_np, axis=1)
    else:
        pred_classes = preds_np

    metrics = {}
    metrics["accuracy"] = accuracy_score(targs_np, pred_classes)
    f1_macro = f1_score(targs_np, pred_classes, average="macro", zero_division=0)
    f1_micro = f1_score(targs_np, pred_classes, average="micro", zero_division=0)
    f1_weighted = f1_score(targs_np, pred_classes, average="weighted", zero_division=0)
    metrics.update(
        {"f1_macro": f1_macro, "f1_micro": f1_micro, "f1_weighted": f1_weighted}
    )

    # Per-class F1
    if num_classes is None:
        num_classes = max(np.max(targs_np), np.max(pred_classes)) + 1
    f1_per = f1_score(
        targs_np,
        pred_classes,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    for i, f1 in enumerate(f1_per):
        name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
        metrics[f"f1_{name}"] = f1

    # Top-k accuracy
    if preds_np.ndim == 2 and preds_np.shape[1] > 1:
        for k in [3, 5]:
            if k < preds_np.shape[1]:
                topk = np.argsort(preds_np, axis=1)[:, -k:]
                topk_acc = np.mean(
                    [targs_np[i] in topk[i] for i in range(len(targs_np))]
                )
                metrics[f"top_{k}_accuracy"] = topk_acc
    return metrics


"""
Classification metrics computations for probing experiments.
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import List, Optional, Dict


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics (accuracy, F1 scores, top-k accuracy).
    """
    preds_np = (
        predictions.detach().cpu().numpy()
        if isinstance(predictions, torch.Tensor)
        else predictions
    )
    targs_np = (
        targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets
    )

    # Determine predicted classes
    if preds_np.ndim == 2:
        pred_classes = np.argmax(preds_np, axis=1)
    else:
        pred_classes = preds_np

    metrics = {}
    metrics["accuracy"] = accuracy_score(targs_np, pred_classes)
    f1_macro = f1_score(targs_np, pred_classes, average="macro", zero_division=0)
    f1_micro = f1_score(targs_np, pred_classes, average="micro", zero_division=0)
    f1_weighted = f1_score(targs_np, pred_classes, average="weighted", zero_division=0)
    metrics.update(
        {"f1_macro": f1_macro, "f1_micro": f1_micro, "f1_weighted": f1_weighted}
    )

    # Per-class F1
    if num_classes is None:
        num_classes = max(np.max(targs_np), np.max(pred_classes)) + 1
    f1_per = f1_score(
        targs_np,
        pred_classes,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    for i, f1 in enumerate(f1_per):
        name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
        metrics[f"f1_{name}"] = f1

    # Top-k accuracy
    if preds_np.ndim == 2 and preds_np.shape[1] > 1:
        for k in [3, 5]:
            if k < preds_np.shape[1]:
                topk = np.argsort(preds_np, axis=1)[:, -k:]
                topk_acc = np.mean(
                    [targs_np[i] in topk[i] for i in range(len(targs_np))]
                )
                metrics[f"top_{k}_accuracy"] = topk_acc
    return metrics
