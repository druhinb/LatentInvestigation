"""Concise feature extraction module for ViT-based models."""

import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional
import logging
from .model_loader import load_model_and_preprocessor
from .feature_collector import FeatureCollector
from .base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class ReconstructionFeatureExtractor(BaseFeatureExtractor):
    """Extracts features from ViT-based models for reconstruction tasks."""

    def __init__(
        self,
        model_name: str,
        ckpt_path: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            checkpoint_path=ckpt_path,
            device=device,
            cache_dir=cache_dir,
        )
        # Load backbone model and its preprocessor
        self.model, self.preprocessor = load_model_and_preprocessor(
            self.model_name, self.ckpt_path, self.device, self.cache_dir
        )
        # expose processor and transform for legacy methods
        self.processor = getattr(self.preprocessor, 'processor', None)
        self.transform = getattr(self.preprocessor, 'transform', None)
        self.hooks, self.feature_cache = [], {}

        if not self.model_name.startswith("timm_"):
            self._setup_hooks()

        self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze model
        logger.info(f"Loaded and froze {self.model_name} on {self.device}.")

    def _setup_hooks(self):
        def hook_fn_closure(name):
            def hook(module, input_tensor, output_tensor):
                self.feature_cache[name] = (
                    output_tensor[0]
                    if isinstance(output_tensor, tuple)
                    else output_tensor
                )

            return hook

        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            for i, layer_module in enumerate(self.model.encoder.layer):
                self.hooks.append(
                    layer_module.register_forward_hook(hook_fn_closure(f"layer_{i}"))
                )

        elif hasattr(self.model, "blocks"):
            for i, block_module in enumerate(self.model.blocks):
                self.hooks.append(
                    block_module.register_forward_hook(hook_fn_closure(f"layer_{i}"))
                )
        else:
            logger.warning(
                f"Could not auto-setup hooks for {self.model_name}. Intermediate feature extraction might be limited."
            )

    # inherits extract_features from BaseFeatureExtractor
    # inherits get_feature_dim
    # inherits __del__ from BaseFeatureExtractor


def load_image_feature_extractor(config: Dict) -> ReconstructionFeatureExtractor:
    """Factory function to create ImageFeatureExtractor from a configuration dictionary."""
    return ReconstructionFeatureExtractor(
        model_name=config.get("model_name", "supervised_vit"),
        ckpt_path=config.get("checkpoint_path"),
        device=config.get("device", None),
        cache_dir=config.get("cache_dir"),
    )
