"""
Feature extraction module for ViT-based models

This module provides functionality to extract features from various layers of
Vision Transformer models including DINOv2, I-JEPA, MoCo v3, and supervised ViTs.
Designed for probing experiments where the backbone model must remain frozen.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from transformers import (
    ViTModel,
    ViTImageProcessor,
    AutoModel,
    AutoImageProcessor,
    AutoProcessor,
)
import timm
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """Extract features from various layers of ViT-based models for probing"""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize feature extractor

        Args:
            model_name: Name/identifier of the model (e.g., 'dinov2', 'ijepa', 'mocov3', 'supervised_vit')
            checkpoint_path: Path to fine-tuned model checkpoint (if applicable)
            device: Device to load model on
            cache_dir: Directory to cache downloaded models
        """
        super().__init__()

        self.model_name = model_name.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.cache_dir = cache_dir

        # Model and processor will be set in _load_model
        self.model = None
        self.processor = None
        self.hooks = []
        self.feature_cache = {}

        self._load_model()
        self._setup_hooks()

    def _load_model(self):
        """Load the appropriate model based on model_name"""

        if self.model_name == "dinov2":
            # DINOv2 model
            model_id = "facebook/dinov2-base"
            self.model = AutoModel.from_pretrained(
                model_id, cache_dir=self.cache_dir, output_hidden_states=True
            )
            self.processor = AutoImageProcessor.from_pretrained(
                model_id, cache_dir=self.cache_dir
            )

        elif self.model_name == "ijepa":
            # I-JEPA model - using timm bc hf doesn't have it!!!
            model_id = "facebook/vit_huge_patch14_224_ijepa"
            self.model = timm.create_model(
                model_id, pretrained=True, num_classes=0
            )
            from torchvision import transforms

            self.processor = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    # transforms.ToTensor(), -- requires PIL
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        elif self.model_name == "mocov3":
            # MoCo v3 model
            try:
                self.model = timm.create_model(
                    "vit_base_patch16_224.mocov3_in1k", pretrained=True, num_classes=0
                )
                from torchvision import transforms

                self.processor = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            except Exception as e:
                logger.warning(f"Could not load MoCo v3 model from timm: {e}")

        elif self.model_name == "supervised_vit":
            # Supervised ViT baseline
            model_id = "google/vit-base-patch16-224"
            self.model = ViTModel.from_pretrained(
                model_id, cache_dir=self.cache_dir, output_hidden_states=True
            )
            self.processor = ViTImageProcessor.from_pretrained(
                model_id, cache_dir=self.cache_dir
            )

        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        # load fine-tuned checkpoint if it's there
        if self.checkpoint_path:
            self._load_checkpoint()

        # Move model to device, set eval mode
        self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(f"Loaded {self.model_name} model on {self.device}")

    def _load_checkpoint(self):
        """Load fine-tuned checkpoint"""
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # for some reason theres two checkpoint keys, so we need to check both
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict, strict=False
        )

        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(
                f"Unexpected keys when loading checkpoint: {unexpected_keys}"
            )

        logger.info(f"Loaded checkpoint from {self.checkpoint_path}")

    def _setup_hooks(self):
        """Setup forward hooks to capture intermediate layer outputs"""
        self.hooks = []
        self.feature_cache = {}

        def create_hook(layer_name):
            def hook(module, input, output):
                self.feature_cache[layer_name] = output

            return hook

        # Register hooks based on model architecture
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            # ViT models
            for i, layer in enumerate(self.model.encoder.layer):
                hook = layer.register_forward_hook(create_hook(f"layer_{i}"))
                self.hooks.append(hook)
        elif hasattr(self.model, "blocks"):
            for i, block in enumerate(self.model.blocks):
                hook = block.register_forward_hook(create_hook(f"layer_{i}"))
                self.hooks.append(hook)
        else:
            logger.warning("Could not setup, probably an invalid architecture.")

    def extract_features(
        self,
        images: torch.Tensor,
        layers: Optional[List[Union[int, str]]] = None,
        feature_type: str = "cls_token",
        return_all_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from the specified layers

        Args:
            images: Input images tensor [batch_size, channels, height, width]
            layers: List of layer indices/names to extract from. If None, uses final layer
            feature_type: Type of features to extract:
                - "cls_token": CLS token from transformer
                - "patch_mean": Mean of all patch tokens
                - "patch_tokens": All patch tokens
                - "all": All available feature types
            return_all_layers: If True, return features from all layers

        Returns:
            Dictionary mapping layer names to feature tensors
        """

        if images.device != self.device:
            images = images.to(self.device)

        # Clear previous cache
        self.feature_cache = {}

        with torch.no_grad():
            # Preprocess images
            if hasattr(self.processor, "preprocess"):

                try:
                    # Convert tensor to numpy if needed for transformers processor
                    if isinstance(images, torch.Tensor):
                        # Ensure images are in [0, 1] range
                        if images.max() > 1.0 or images.min() < 0.0:
                            images = torch.clamp(images, 0, 1)
                        # Convert to numpy and handle batch dimension
                        if images.dim() == 4:  # [B, C, H, W]
                            images_list = []
                            for i in range(images.shape[0]):
                                img = (
                                    images[i].cpu().numpy().transpose(1, 2, 0)
                                )  # CHW -> HWC
                                img = (img * 255).astype(np.uint8)
                                images_list.append(img)
                            processed = self.processor(images_list, return_tensors="pt")
                        else:  # Single image
                            img = images.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                            img = (img * 255).astype(np.uint8)

                            processed = self.processor(img, return_tensors="pt")
                        pixel_values = processed.pixel_values.to(self.device)
                    else:
                        processed = self.processor(images, return_tensors="pt")
                        pixel_values = processed.pixel_values.to(self.device)
                except Exception as e:
                    logger.warning(
                        f"Failed to process: {e}, falling back to direct input"
                    )
                    pixel_values = images
            elif callable(self.processor):
                # torchvision transform
                if images.dim() == 4:
                    # Process each image in batch
                    pixel_values = torch.stack([self.processor(img) for img in images])
                else:
                    pixel_values = self.processor(images).unsqueeze(0)
                pixel_values = pixel_values.to(self.device)
            else:
                pixel_values = images

            # Forward pass
            if hasattr(self.model, "forward_features"):
                # timm models
                outputs = self.model.forward_features(pixel_values)
            else:
                # Transformers models
                outputs = self.model(pixel_values, output_hidden_states=True)

        extracted_features = {}

        # Determine which layers to extract from
        if return_all_layers:
            if hasattr(outputs, "hidden_states"):
                available_layers = list(range(len(outputs.hidden_states)))
            else:
                available_layers = list(range(len(self.feature_cache)))
        elif layers is None:
            available_layers = [-1]  # Final layer only
        else:
            available_layers = layers

        for layer_idx in available_layers:
            layer_name = f"layer_{layer_idx}" if layer_idx >= 0 else "final"

            # Get layer output
            if hasattr(outputs, "hidden_states") and layer_idx < len(
                outputs.hidden_states
            ):
                # Transformers models
                layer_output = outputs.hidden_states[layer_idx]
            elif f"layer_{layer_idx}" in self.feature_cache:
                # timm models with hooks
                layer_output = self.feature_cache[f"layer_{layer_idx}"]
            elif layer_idx == -1:
                # Final layer output
                if hasattr(outputs, "last_hidden_state"):
                    layer_output = outputs.last_hidden_state
                elif isinstance(outputs, torch.Tensor):
                    layer_output = outputs
                else:
                    continue
            else:
                continue

            # Extract specific feature types
            layer_features = {}

            if feature_type in ["cls_token", "all"]:
                # CLS token (first token)
                if layer_output.dim() == 3:  # [batch, seq_len, hidden_dim]
                    cls_features = layer_output[:, 0]  # First token
                    layer_features["cls_token"] = cls_features

            if feature_type in ["patch_mean", "all"]:
                # Mean of patch tokens (excluding CLS token)
                if layer_output.dim() == 3:
                    patch_features = layer_output[:, 1:]  # Exclude CLS token
                    patch_mean = patch_features.mean(dim=1)
                    layer_features["patch_mean"] = patch_mean

            if feature_type in ["patch_tokens", "all"]:
                # All patch tokens
                if layer_output.dim() == 3:
                    patch_features = layer_output[:, 1:]  # Exclude CLS token
                    layer_features["patch_tokens"] = patch_features

            # Store features for this layer
            if feature_type == "all":
                for feat_name, feat_tensor in layer_features.items():
                    extracted_features[f"{layer_name}_{feat_name}"] = feat_tensor
            else:
                if feature_type in layer_features:
                    extracted_features[layer_name] = layer_features[feature_type]

        return extracted_features

    def get_feature_dim(self, feature_type: str = "cls_token") -> int:
        """Get the dimensionality of extracted features"""
        # Create a dummy input to determine the feature dimensions
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        features = self.extract_features(
            dummy_input, layers=[-1], feature_type=feature_type
        )

        if features:
            feature_tensor = list(features.values())[0]
            if feature_tensor.dim() == 2:  # [batch, feature_dim]
                return feature_tensor.shape[1]
            elif feature_tensor.dim() == 3:  # [batch, seq_len, feature_dim]
                return feature_tensor.shape[2]

        # Fallback
        return 768

    def __del__(self):
        """Clean up hooks when object is destroyed"""
        for hook in self.hooks:
            hook.remove()


def load_feature_extractor(config: Dict) -> FeatureExtractor:
    """
    Factory function to create feature extractor from config

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        Configured FeatureExtractor instance
    """
    return FeatureExtractor(
        model_name=config.get("model_name"),
        checkpoint_path=config.get("checkpoint_path"),
        device=config.get(
            "device",
            (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            ),
        ),
        cache_dir=config.get("cache_dir"),
    )
