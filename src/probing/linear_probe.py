import torch
import torch.nn as nn
from .base_probe import BaseProbe


class LinearProbe(BaseProbe):
    """Simple linear probe for regression or classification tasks"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: str = "regression",
        dropout_rate: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type.lower()

        if self.task_type not in ["regression", "classification"]:
            raise ValueError(
                f"task_type must be 'regression' or 'classification', got {task_type}"
            )

        layers = []
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(input_dim, output_dim, bias=bias))

        self.probe = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe(x)

    def get_loss_function(self):
        if self.task_type == "regression":
            return nn.MSELoss()
        elif self.task_type == "classification":
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()
