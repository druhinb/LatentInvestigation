import torch
import torch.nn as nn
import logging
from typing import Dict, List, Union, Optional, Tuple
from .model_loader import load_model_and_preprocessor
from .feature_collector import FeatureCollector

logger = logging.getLogger(__name__)


class BaseFeatureExtractor(nn.Module):
    """
    Unified feature extractor for Vision Transformers.
    Provides loading, preprocessing, hook-based caching, and feature collection.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.device = device
        self.cache_dir = cache_dir
        # Load model and preprocessor
        self.model, self.preprocessor = load_model_and_preprocessor(
            self.model_name, checkpoint_path, self.device, self.cache_dir
        )
        # expose legacy interfaces
        self.processor = getattr(self.preprocessor, "processor", None)
        self.transform = getattr(self.preprocessor, "transform", None)
        self.hooks: List = []
        self.feature_cache: Dict[str, torch.Tensor] = {}
        # register hooks for HF-style models
        if not self.model_name.startswith("timm_"):
            self._setup_hooks()

    def _setup_hooks(self):
        """Setup forward hooks to capture intermediate layer outputs"""

        def create_hook(layer_name):
            def hook(module, input, output):
                self.feature_cache[layer_name] = output

            return hook

        # register on HF encoder.layer or timm blocks
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            for i, layer in enumerate(self.model.encoder.layer):
                h = layer.register_forward_hook(create_hook(f"layer_{i}"))
                self.hooks.append(h)
        elif hasattr(self.model, "blocks"):
            for i, block in enumerate(self.model.blocks):
                h = block.register_forward_hook(create_hook(f"layer_{i}"))
                self.hooks.append(h)
        else:
            logger.warning(
                f"No transformer blocks found for hooks in {self.model_name}"
            )

    def extract_features(
        self,
        images: torch.Tensor,
        layers: Optional[List[Union[int, str]]] = None,
        feature_type: str = "cls_token",
        chunk_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts features at specified transformer layers and returns a dict of tensors.

        Args:
            images: Input images tensor
            layers: Layers to extract from
            feature_type: Type of features to extract
            chunk_size: Process images in chunks to reduce memory usage
        """
        if chunk_size is not None and images.size(0) > chunk_size:
            return self._extract_features_chunked(
                images, layers, feature_type, chunk_size
            )

        # reset cache
        self.feature_cache = {}
        # preprocess
        inputs, is_dict = self.preprocessor.preprocess(images)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(**inputs) if is_dict else self.model(inputs)

        # collect features
        collector = FeatureCollector(
            outputs=outputs,
            cache=self.feature_cache,
            model=self.model,
            model_name=self.model_name,
        )
        results = collector.collect(layers, feature_type)

        self.feature_cache.clear()

        return results

    def _extract_features_chunked(
        self,
        images: torch.Tensor,
        layers: Optional[List[Union[int, str]]] = None,
        feature_type: str = "cls_token",
        chunk_size: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features in chunks to reduce memory usage for large batches.

        Args:
            images: Input images tensor [B, C, H, W]
            layers: Layers to extract from
            feature_type: Type of features to extract
            chunk_size: Number of images to process at once
        """
        batch_size = images.size(0)
        all_features = {}

        for i in range(0, batch_size, chunk_size):
            chunk_end = min(i + chunk_size, batch_size)
            chunk = images[i:chunk_end]

            chunk_features = self.extract_features(
                chunk, layers, feature_type, chunk_size=None
            )

            for key, features in chunk_features.items():
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(features.cpu())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_features = {}
        for key, feature_list in all_features.items():
            final_features[key] = torch.cat(feature_list, dim=0)

        return final_features

    def get_feature_dim(
        self,
        layers: Optional[List[Union[int, str]]] = None,
        feature_type: str = "cls_token",
    ) -> int:
        """
        Determine feature dimensionality via a dummy forward pass.
        """
        # infer height/width from preprocessor
        h, w = 224, 224
        size_dict = getattr(self.preprocessor, "size", None)
        if isinstance(size_dict, dict) and "height" in size_dict:
            h = size_dict["height"]
            w = size_dict["width"]
        # fallback for HF processor size
        elif hasattr(self.processor, "size") and isinstance(self.processor.size, dict):
            h = self.processor.size.get("height", h)
            w = self.processor.size.get("width", w)
        # dummy input
        dummy = torch.randn(1, 3, h, w).to(self.device)
        feats = self.extract_features(
            dummy, layers=layers or [-1], feature_type=feature_type
        )
        if not feats and hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        if not feats:
            logger.warning("Could not infer feature dim; defaulting to 768")
            return 768
        first = next(iter(feats.values()))
        return first.shape[-1]

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_stats = {}

        if torch.cuda.is_available():
            memory_stats["cuda_allocated"] = (
                torch.cuda.memory_allocated() / 1024**3
            )  # GB
            memory_stats["cuda_reserved"] = torch.cuda.memory_reserved() / 1024**3  # GB
        elif torch.backends.mps.is_available():
            # MPS doesn't have detailed memory stats, but we can track basic info
            memory_stats["mps_available"] = True

        return memory_stats

    def clear_memory(self):
        """Clear GPU memory and caches."""
        self.feature_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            # MPS equivalent (less comprehensive than CUDA)
            pass

    def __del__(self):
        for h in self.hooks:
            h.remove()
