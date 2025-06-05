"""
Probe implementations for linear and MLP-based probes as well as attention probe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple
import logging
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader

from .base_probe import BaseProbe
from src.probing.metrics import (
    MetricsTracker,
)

logger = logging.getLogger(__name__)


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
            raise NotImplementedError("This method is not yet implemented.")

        self.probe = nn.Sequential(*layers)

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
        # Return appropriate loss based on task_type
        if self.task_type == "regression":
            return nn.MSELoss()
        elif self.task_type == "classification":
            # Assuming multi-class classification; for binary, BCEWithLogitsLoss might be better
            # if output_dim is 1 and it's binary.
            return nn.CrossEntropyLoss()
        else:
            # Fallback or raise error
            logger.warning(
                f"No specific loss function for task_type {self.task_type}, defaulting to MSELoss."
            )
            return nn.MSELoss()


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

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.task_type = task_type.lower()
        self.activation = activation.lower()
        self.batch_hnorm = batch_norm
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

    def _get_activation_function(self):
        """Get activation function from string"""
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "gelu":
            return nn.GELU()
        elif self.activation == "tanh":
            return nn.Tanh()

    def _build_mlp(self, dropout_rate: float, batch_norm: bool, bias: bool):
        """Build the MLP architecture"""
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        layers = []

        for i in range(len(layer_dims) - 1):
            cin = layer_dims[i]
            cout = layer_dims[i + 1]

            # Add dropout if requested
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            layers.append(nn.Linear(cin, cout, bias=bias))

            # Only add BN + activation for hidden layers (is this better for probes?)
            if i < len(layer_dims) - 2:
                # Add batchnorm if requested
                if batch_norm:
                    layers.append(
                        nn.BatchNorm1d(cout)
                    )  # should we implement a batchnorm config section that allows us to change eps, momentum, etc.

                layers.append(self._get_activation_function())

        self.mlp_probe = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Output predictions [batch_size, output_dim]
        """
        return self.mlp_probe(x)

    def get_loss_function(self):
        """Get appropriate loss function for the task"""
        if self.task_type == "regression":
            return nn.MSELoss()
        elif self.task_type == "classification":
            return nn.CrossEntropyLoss()


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


class VoxelProbe(BaseProbe):
    """
    A probe that takes 2D features and outputs a 3D voxel occupancy grid.
    Architecture: Linear projection -> Reshape -> ConvTranspose3D stack -> Final Conv3D.
    Outputs logits for voxel occupancy, intended for use with BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_dim: int,
        voxel_resolution: int = 32,
        initial_channels: int = 256,
        upsampling_channels: List[int] = [128, 64, 32],  # Defines num_upsampling_layers
        kernel_size_transpose: int = 4,
        stride_transpose: int = 2,
        padding_transpose: int = 1,
        final_conv_kernel_size: int = 3,
        activation_fn_str: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.voxel_resolution = voxel_resolution
        self.initial_channels = initial_channels
        self.task_type = "voxel_reconstruction"  # For ProbeTrainer, if it's extended

        if activation_fn_str.lower() == "relu":
            self.activation_fn = nn.ReLU(inplace=True)
        elif activation_fn_str.lower() == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn_str}")

        self.num_upsampling_layers = len(upsampling_channels)

        # Derive initial_reshaped_dim (the spatial dimension after the first projection and reshape)
        # e.g., if voxel_resolution=32, num_upsampling_layers=3, stride_transpose=2,
        # then initial_reshaped_dim = 32 / (2^3) = 4.
        self.initial_reshaped_dim = self.voxel_resolution // (
            stride_transpose**self.num_upsampling_layers
        )

        if (
            self.voxel_resolution % (stride_transpose**self.num_upsampling_layers) != 0
            or self.initial_reshaped_dim <= 0
        ):
            raise ValueError(
                f"Voxel resolution {self.voxel_resolution} is not cleanly achievable from a positive initial dimension "
                f"with {self.num_upsampling_layers} upsampling layers of stride {stride_transpose}. "
                f"Calculated initial_reshaped_dim: {self.initial_reshaped_dim}. "
                "Please adjust voxel_resolution, the number of upsampling_channels (which defines num_upsampling_layers), or stride_transpose."
            )

        # 1. Linear projection to a size that can be reshaped into a small 3D volume
        projector_output_features = self.initial_channels * (
            self.initial_reshaped_dim**3
        )
        self.projector = nn.Linear(input_dim, projector_output_features)

        # 2. Upsampling layers (ConvTranspose3d stack)
        upsampling_layers_list = []
        current_channels = self.initial_channels
        for i in range(self.num_upsampling_layers):
            out_c = upsampling_channels[i]
            upsampling_layers_list.append(
                nn.ConvTranspose3d(
                    in_channels=current_channels,
                    out_channels=out_c,
                    kernel_size=kernel_size_transpose,
                    stride=stride_transpose,
                    padding=padding_transpose,
                    bias=False,  # Common practice when using BatchNorm
                )
            )
            upsampling_layers_list.append(nn.BatchNorm3d(out_c))
            upsampling_layers_list.append(self.activation_fn)
            current_channels = out_c
        self.upsampler = nn.Sequential(*upsampling_layers_list)

        # 3. Final Conv3d layer to produce 1 channel (occupancy logits)
        # The output resolution should match self.voxel_resolution
        self.final_conv = nn.Conv3d(
            in_channels=current_channels,
            out_channels=1,  # Single channel for occupancy logits
            kernel_size=final_conv_kernel_size,
            padding=final_conv_kernel_size
            // 2,  # 'same' padding to maintain resolution
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is expected to be [Batch, input_dim]
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input shape [batch_size, input_dim={self.input_dim}], but got {x.shape}"
            )

        batch_size = x.shape[0]

        # 1. Project and reshape
        x = self.projector(x)
        # Reshape to [Batch, initial_channels, initial_reshaped_dim, initial_reshaped_dim, initial_reshaped_dim]
        x = x.view(
            batch_size,
            self.initial_channels,
            self.initial_reshaped_dim,
            self.initial_reshaped_dim,
            self.initial_reshaped_dim,
        )

        # 2. Upsample through ConvTranspose3D stack
        x = self.upsampler(x)

        # 3. Final convolution to get single-channel logits
        # Output shape: [Batch, 1, voxel_resolution, voxel_resolution, voxel_resolution]
        x = self.final_conv(x)

        return x  # Outputting logits

    def get_loss_function(self):
        return nn.BCEWithLogitsLoss()


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

    elif probe_type == "voxel":
        return VoxelProbe(
            input_dim=config["input_dim"],
            voxel_resolution=config.get("voxel_resolution", 32),
            initial_channels=config.get("initial_channels", 256),
            upsampling_channels=config.get("upsampling_channels", [128, 64, 32]),
            kernel_size_transpose=config.get("kernel_size_transpose", 4),
            stride_transpose=config.get("stride_transpose", 2),
            padding_transpose=config.get("padding_transpose", 1),
            final_conv_kernel_size=config.get("final_conv_kernel_size", 3),
            activation_fn_str=config.get("activation_fn", "relu"),
        )

    else:
        raise ValueError(f"Unknown probe type: {probe_type}")


class ProbeTrainer:
    """Helper class for training probes"""

    def __init__(
        self,
        probe: nn.Module,
        device: str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
        MetricsTracker: Optional[
            MetricsTracker
        ] = None,  # Retained for external use by experiment
    ):
        """
        Initialize probe trainer

        Args:
            probe: Probe model to train
            device: Device to train on
            MetricsTracker: Optional tracker instance, managed by calling experiment.
        """
        self.probe = probe
        self.device = device
        self.probe.to(device)

        self.metrics_tracker = MetricsTracker

        # Set up loss function
        self.criterion = probe.get_loss_function()

    def train(
        self,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        patience: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        probe_type: Optional[str] = None,
        layer: Optional[int] = None,
        wandb_enabled: bool = False,
        save_best_probe: bool = True,
        probe_save_path: Optional[str] = None,
    ):
        """Train the probe model"""
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        best_probe_state_dict = self.probe.state_dict()

        for epoch in range(epochs):
            self.probe.train()
            train_loss_epoch = 0
            num_batches_train = 0

            progress_bar_train = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Train Lyr:{layer} Prb:{probe_type}]",
                leave=False,
            )
            for batch in progress_bar_train:
                optimizer.zero_grad()

                if self.probe.task_type == "voxel_reconstruction":
                    features = batch["processed_views"].to(self.device)
                    targets = batch["target_voxels"].to(self.device)
                    features = features.view(features.size(0), -1)
                elif self.probe.task_type == "viewpoint_regression":
                    features = batch["features"].to(self.device)
                    targets = batch["camera_params"].to(self.device)
                else:
                    features = batch["features"].to(self.device)
                    targets = batch["labels"].to(self.device)

                if self.probe.task_type in [
                    "regression",
                    "viewpoint_regression",
                    "voxel_reconstruction",
                ]:
                    targets = targets.float()
                elif self.probe.task_type == "classification":
                    targets = targets.long()

                outputs = self.probe(features)
                loss = self.criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item()
                num_batches_train += 1
                progress_bar_train.set_postfix(
                    loss=train_loss_epoch / num_batches_train
                )

            avg_train_loss = train_loss_epoch / num_batches_train

            self.probe.eval()
            val_loss_epoch = 0
            num_batches_val = 0
            progress_bar_val = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Val Lyr:{layer} Prb:{probe_type}]",
                leave=False,
            )
            with torch.no_grad():
                for batch in progress_bar_val:
                    if self.probe.task_type == "voxel_reconstruction":
                        features = batch["processed_views"].to(self.device)
                        targets = batch["target_voxels"].to(self.device)
                        features = features.view(features.size(0), -1)  # Flatten
                    elif self.probe.task_type == "viewpoint_regression":
                        features = batch["features"].to(self.device)
                        targets = batch["camera_params"].to(self.device)
                    else:
                        features = batch["features"].to(self.device)
                        targets = batch["labels"].to(self.device)

                    if self.probe.task_type in [
                        "regression",
                        "viewpoint_regression",
                        "voxel_reconstruction",
                    ]:
                        targets = targets.float()
                    elif self.probe.task_type == "classification":
                        targets = targets.long()

                    outputs = self.probe(features)
                    loss = self.criterion(outputs, targets)
                    val_loss_epoch += loss.item()
                    num_batches_val += 1
                    progress_bar_val.set_postfix(loss=val_loss_epoch / num_batches_val)

            avg_val_loss = val_loss_epoch / num_batches_val
            logger.info(
                f"Epoch {epoch+1}/{epochs} Lyr:{layer} Prb:{probe_type} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            if wandb_enabled:
                log_data = {
                    "probe_training/epoch": epoch + 1,
                    f"probe_training/train_loss_l{layer}_pt{probe_type}": avg_train_loss,
                    f"probe_training/val_loss_l{layer}_pt{probe_type}": avg_val_loss,
                    f"probe_training/lr_l{layer}_pt{probe_type}": optimizer.param_groups[
                        0
                    ][
                        "lr"
                    ],
                }
                wandb.log(log_data)

            if self.metrics_tracker is not None:
                self.metrics_tracker.update("train", {"loss": avg_train_loss}, epoch)
                self.metrics_tracker.update("val", {"loss": avg_val_loss}, epoch)

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
                best_probe_state_dict = self.probe.state_dict()
                if save_best_probe and probe_save_path is not None:
                    torch.save(self.probe.state_dict(), probe_save_path)
                    logger.info(
                        f"Saved best probe for layer {layer}, type {probe_type} to {probe_save_path} (Val Loss: {self.best_val_loss:.4f})"
                    )
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch+1} for layer {layer}, probe {probe_type}."
                    )
                    break

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

        return best_probe_state_dict, self.best_val_loss

    def evaluate(
        self,
        test_loader: DataLoader,
        wandb_enabled: bool = False,
        metrics_prefix: str = "Test",
        probe_type: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> dict:
        """Evaluate the probe model on the test set"""
        self.probe.eval()
        test_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []

        progress_bar_eval = tqdm(
            test_loader, desc=f"Evaluating Lyr:{layer} Prb:{probe_type}", leave=False
        )
        with torch.no_grad():
            for batch in progress_bar_eval:
                if self.probe.task_type == "voxel_reconstruction":
                    features = batch["processed_views"].to(self.device)
                    targets = batch["target_voxels"].to(self.device)
                    features = features.view(features.size(0), -1)
                elif self.probe.task_type == "viewpoint_regression":
                    features = batch["features"].to(self.device)
                    targets = batch["camera_params"].to(self.device)
                else:
                    features = batch["features"].to(self.device)
                    targets = batch["labels"].to(self.device)

                if self.probe.task_type in [
                    "regression",
                    "viewpoint_regression",
                    "voxel_reconstruction",
                ]:
                    targets = targets.float()
                elif self.probe.task_type == "classification":
                    targets = targets.long()

                outputs = self.probe(features)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                num_batches += 1

                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
                progress_bar_eval.set_postfix(loss=test_loss / num_batches)

        avg_loss = test_loss / num_batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = {"loss": avg_loss}
        if self.probe.task_type == "regression":
            # Mean Absolute Error
            mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
            # Root Mean Square Error
            rmse = torch.sqrt(torch.mean((all_predictions - all_targets) ** 2)).item()

            if all_targets.dim() == 1:  #
                ss_res = torch.sum((all_targets - all_predictions) ** 2)
                ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
                if ss_tot == 0:
                    r2 = (
                        torch.tensor(float("-inf")) if ss_res > 0 else torch.tensor(1.0)
                    )
                else:
                    r2 = 1 - (ss_res / ss_tot)
            elif all_targets.size(1) > 0:
                target_mean = torch.mean(all_targets, dim=0, keepdim=True)
                ss_res_per_dim = torch.sum((all_targets - all_predictions) ** 2, dim=0)
                ss_tot_per_dim = torch.sum((all_targets - target_mean) ** 2, dim=0)

                valid_dims_mask = ss_tot_per_dim > 1e-10  # so tiny

                r2_per_dim = torch.zeros_like(ss_tot_per_dim)
                r2_per_dim[valid_dims_mask] = 1 - (
                    ss_res_per_dim[valid_dims_mask] / ss_tot_per_dim[valid_dims_mask]
                )
                r2_per_dim[~valid_dims_mask & (ss_res_per_dim <= 1e-10)] = 1.0
                r2_per_dim[~valid_dims_mask & (ss_res_per_dim > 1e-10)] = float("-inf")

                r2 = (
                    torch.mean(r2_per_dim[valid_dims_mask])
                    if torch.any(valid_dims_mask)
                    else torch.tensor(float("nan"))
                )
            else:
                r2 = torch.tensor(float("nan"))

            metrics.update({"mae": mae, "rmse": rmse, "r2": r2.item()})

        elif self.probe.task_type == "classification":
            predicted_classes = torch.argmax(all_predictions, dim=1)
            accuracy = torch.mean((predicted_classes == all_targets).float()).item()
            metrics.update({"accuracy": accuracy})

        elif self.probe.task_type == "viewpoint_regression":
            from src.probing.metrics import (
                compute_viewpoint_specific_metrics,
            )  # Local import

            vp_metrics = compute_viewpoint_specific_metrics(
                azimuth_pred=all_predictions[:, 0],
                elevation_pred=all_predictions[:, 1],
                azimuth_target=all_targets[:, 0],
                elevation_target=all_targets[:, 1],
            )
            metrics.update(vp_metrics)
            mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
            rmse = torch.sqrt(torch.mean((all_predictions - all_targets) ** 2)).item()
            metrics.update({"mae": mae, "rmse": rmse})

        elif self.probe.task_type == "voxel_reconstruction":
            from src.probing.voxel_metrics import compute_voxel_metrics

            voxel_metrics = compute_voxel_metrics(all_predictions, all_targets)
            metrics.update(voxel_metrics)

        logger.info(
            f"{metrics_prefix} Metrics (Lyr:{layer} Prb:{probe_type}): {metrics}"
        )
        if wandb_enabled:
            wandb_log_data = {}
            for k, v in metrics.items():

                log_key = f"probe_evaluation/{metrics_prefix.lower()}_{k}"
                if probe_type:
                    log_key += f"_pt{probe_type}"
                if layer is not None:
                    log_key += f"_l{layer}"
                wandb_log_data[log_key] = v
            wandb.log(wandb_log_data)

        return metrics
