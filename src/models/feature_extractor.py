"""
Feature extraction module for ViT-based models

This module provides functionality to extract features from various layers of
Vision Transformer models including DINOv2, I-JEPA, MoCo v3, and supervised ViTs.
Designed for probing experiments where the backbone model must remain frozen.
"""

import torch
import logging
from typing import Dict, Optional

from .base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class FeatureExtractor(BaseFeatureExtractor):
    """Extract features from ViT-based models for probing"""

    def __init__(
        self, model_name: str, checkpoint_path: Optional[str] = None, device: str = None, cache_dir: Optional[str] = None
    ):
        super().__init__(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            cache_dir=cache_dir,
        )

    # inherits extract_features from BaseFeatureExtractor
    # inherits get_feature_dim from BaseFeatureExtractor
    # inherits __del__ from BaseFeatureExtractor


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
        device=config.get("device"),
        cache_dir=config.get("cache_dir"),
    )
