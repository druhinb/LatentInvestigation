import torch
import torch.nn as nn
from .base_probe import BaseProbe


class AttentionProbe(BaseProbe):
    """Probe with attention mechanism over patch tokens"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: str = "regression",
        attention_dim: int = 64,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type.lower()
        self.attention_dim = attention_dim

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1),
        )

        # Final classifier/regressor
        layers = []
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(input_dim, output_dim))

        self.classifier = nn.Sequential(*layers)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_patches, input_dim]
        attn_weights = self.attention(x)  # [batch_size, num_patches, 1]
        attn_applied = torch.sum(x * attn_weights, dim=1)  # [batch_size, input_dim]
        return self.classifier(attn_applied)

    def get_loss_function(self):
        if self.task_type == "regression":
            return nn.MSELoss()
        return nn.CrossEntropyLoss()
