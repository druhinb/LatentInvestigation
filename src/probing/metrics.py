"""
Evaluation metrics for probing experiments

This module provides evaluation metrics for both regression
and classification tasks in the context of probing the SSL vision models
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


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
