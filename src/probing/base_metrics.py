"""
Utility functions for metric computation, such as tensor conversion and flattening.
"""
import torch
import numpy as np


def to_numpy(x):
    """
    Convert torch.Tensor or numpy array to numpy array on CPU.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)
