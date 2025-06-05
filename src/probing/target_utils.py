"""
Utils for preparing targets for different probing tasks.
"""

import torch
import numpy as np
from typing import Dict, Union


def prepare_targets_for_task(
    metadata: Dict, task_type: str, output_format: str = "tensor"
) -> Union[torch.Tensor, np.ndarray]:
    """Prepare targets based on the probing task"""
    if task_type == "viewpoint_regression":
        az = torch.tensor(metadata["azimuths"], dtype=torch.float32)
        el = torch.tensor(metadata["elevations"], dtype=torch.float32)
        targets = torch.stack([az, el], dim=1)
    elif task_type == "shape_classification":
        cats = metadata.get("categories", [])
        uniques = sorted(set(cats))
        idx_map = {c: i for i, c in enumerate(uniques)}
        targets = torch.tensor([idx_map[c] for c in cats], dtype=torch.long)
    elif task_type == "view_classification":
        az = np.array(metadata.get("azimuths", []))
        el = np.array(metadata.get("elevations", []))
        az_bins = np.linspace(0, 360, 9)
        el_bins = np.linspace(-30, 30, 4)
        a_idx = np.digitize(az, az_bins) - 1
        e_idx = np.digitize(el, el_bins) - 1
        cls_idx = a_idx * len(el_bins - 1) + e_idx
        targets = torch.tensor(cls_idx, dtype=torch.long)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    return targets.numpy() if output_format == "numpy" else targets
