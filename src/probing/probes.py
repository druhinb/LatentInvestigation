"""
Probe implementations for linear and MLP-based probes as well as attention probe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Simple linear probe for regression or classification tasks"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: str = "regression",
        dropout_rate: float = 0.0,
        bias: bool = True,
    ):
        """
        Initialize linear probe

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output (2 for viewpoint regression, N for N-class classification)
            task_type: Either "regression" or "classification"
            dropout_rate: Dropout rate (applied before final layer)
            bias: Whether to use bias in linear layer
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type.lower()

        if self.task_type not in ["regression", "classification"]:
            raise ValueError(
                f"task_type must be 'regression' or 'classification', got {task_type}"
            )

        # Build probe
        layers = []

        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(input_dim, output_dim, bias=bias))

        # if we're classifying, the loss function handles the softmax
        if self.task_type == "classification" and output_dim > 1:
            pass

        self.probe = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize probe weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for all linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Output predictions [batch_size, output_dim]
        """
        return self.probe(x)

    def get_loss_function(self):
        """Get appropriate loss function for the task"""
        if self.task_type == "regression":
            return nn.MSELoss()
        elif self.task_type == "classification":
            return nn.CrossEntropyLoss()


class MLPProbe(nn.Module):
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
        """
        Initialize MLP probe

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output
            hidden_dims: List of hidden layer dimensions (e.g., [256, 128])
            task_type: Either "regression" or "classification"
            activation: Activation function ("relu", "gelu", "tanh")
            dropout_rate: Dropout rate (applied after each hidden layer)
            batch_norm: Whether to use batch normalization
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        pass

    def _get_activation_function(self):
        """Get activation function from string"""
        pass

    def _build_mlp(self, dropout_rate: float, batch_norm: bool, bias: bool):
        """Build the MLP architecture"""
        pass

    def _init_weights(self):
        """Initialize MLP weights"""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Output predictions [batch_size, output_dim]
        """
        pass

    def get_loss_function(self):
        """Get appropriate loss function for the task"""
        pass


class AttentionProbe(nn.Module):
    """Probe with attention mechanism over patch tokens"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: str = "regression",
        attention_dim: int = 64,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize attention-based probe

        Args:
            input_dim: Dimension of each patch token
            output_dim: Dimension of output
            task_type: Either "regression" or "classification"
            attention_dim: Dimension of attention mechanism
            dropout_rate: Dropout rate
        """
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

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention over patch tokens

        Args:
            x: Patch token features [batch_size, num_patches, input_dim]

        Returns:
            Output predictions [batch_size, output_dim]
        """
        # Compute attention weights
        attention_weights = self.attention(x)  # [batch_size, num_patches, 1]

        # Apply attention to aggregate features
        attended_features = torch.sum(
            x * attention_weights, dim=1
        )  # [batch_size, input_dim]

        # Final prediction
        output = self.classifier(attended_features)

        return output

    def get_loss_function(self):
        """Get appropriate loss function for the task"""
        if self.task_type == "regression":
            return nn.MSELoss()
        elif self.task_type == "classification":
            return nn.CrossEntropyLoss()


def create_probe(config: dict) -> nn.Module:
    """
    Factory function to create probe from configuration

    Args:
        config: Configuration dictionary with probe parameters

    Returns:
        Configured probe instance
    """
    probe_type = config.get("type", "linear").lower()

    if probe_type == "linear":
        return LinearProbe(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            task_type=config.get("task_type", "regression"),
            dropout_rate=config.get("dropout_rate", 0.0),
            bias=config.get("bias", True),
        )

    elif probe_type == "mlp":
        return MLPProbe(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            hidden_dims=config.get("hidden_dims", [256]),
            task_type=config.get("task_type", "regression"),
            activation=config.get("activation", "relu"),
            dropout_rate=config.get("dropout_rate", 0.0),
            batch_norm=config.get("batch_norm", False),
            bias=config.get("bias", True),
        )

    elif probe_type == "attention":
        return AttentionProbe(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            task_type=config.get("task_type", "regression"),
            attention_dim=config.get("attention_dim", 64),
            dropout_rate=config.get("dropout_rate", 0.0),
        )

    else:
        raise ValueError(f"Unknown probe type: {probe_type}")


class ProbeTrainer:
    """Helper class for training probes"""

    def __init__(
        self,
        probe: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize probe trainer

        Args:
            probe: Probe model to train
            device: Device to train on
        """
        self.probe = probe
        self.device = device
        self.probe.to(device)

        # Set up loss function
        self.criterion = probe.get_loss_function()

    def train_epoch(self, dataloader, optimizer, scheduler=None) -> float:
        """Train probe for one epoch"""
        self.probe.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.probe(features)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        return total_loss / num_batches

    def evaluate(self, dataloader) -> dict:
        """Evaluate probe on dataset"""
        self.probe.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)

                outputs = self.probe(features)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = {"loss": total_loss / len(dataloader)}

        # Task-specific metrics
        if self.probe.task_type == "regression":
            # Mean Absolute Error
            mae = torch.mean(torch.abs(predictions - targets)).item()
            # R-squared
            ss_res = torch.sum((targets - predictions) ** 2)
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            metrics.update({"mae": mae, "r2": r2.item()})

        elif self.probe.task_type == "classification":
            # Accuracy
            predicted_classes = torch.argmax(predictions, dim=1)
            accuracy = torch.mean((predicted_classes == targets).float()).item()

            metrics.update({"accuracy": accuracy})

        return metrics
