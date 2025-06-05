import torch
import torch.nn as nn
from .base_probe import BaseProbe
from typing import List, Optional


class MLPProbe(BaseProbe):
    """Multi-layer perceptron probe for more complex feature relationships"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        task_type: str = "regression",
        activation: str = "relu",
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.task_type = task_type.lower()
        self.activation = activation.lower()
        self.batch_norm = batch_norm
        self.bias = bias

        if self.task_type not in ["regression", "classification"]:
            raise ValueError(
                f"task_type must be 'regression' or 'classification', got {task_type}"
            )

        if self.activation not in ["relu", "gelu", "tanh"]:
            raise ValueError(
                f"activation must be 'relu', 'gelu', or 'tanh', got {activation}"
            )

        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError(f"invalid dropout rate: {dropout_rate}")

        self._build_mlp(dropout_rate, batch_norm, bias)
        self._init_weights()

    def _get_activation(self) -> nn.Module:
        if self.activation == "relu":
            return nn.ReLU()
        if self.activation == "gelu":
            return nn.GELU()
        if self.activation == "tanh":
            return nn.Tanh()

    def _build_mlp(self, dropout_rate: float, batch_norm: bool, bias: bool):
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        layers = []
        for i in range(len(dims) - 1):
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i < len(dims) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(self._get_activation())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def get_loss_function(self):
        if self.task_type == "regression":
            return nn.MSELoss()
        return nn.CrossEntropyLoss()
