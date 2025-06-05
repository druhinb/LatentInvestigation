from abc import ABC, abstractmethod
import torch.nn as nn


class BaseProbe(nn.Module, ABC):
    """Abstract base class for all probing modules, providing common weight initialization."""
    def __init__(self):
        super().__init__()

    def _init_weights(self):
        """Initialize all nn.Linear layers with Xavier uniform and zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @abstractmethod
    def get_loss_function(self):
        """Return the appropriate loss function for the probe."""
        pass
