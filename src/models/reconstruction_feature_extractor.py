"""Concise feature extraction module for ViT-based models."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union, Optional
from transformers import AutoModel, AutoImageProcessor
import timm
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReconstructionFeatureExtractor(nn.Module):
    """Extracts features from ViT-based models."""

    def __init__(
        self,
        model_name: str,
        ckpt_path: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name, self.ckpt_path, self.device, self.cache_dir = (
            model_name.lower(),
            ckpt_path,
            device,
            cache_dir,
        )
        self.model, self.processor, self.transform = (
            None,
            None,
            None,
        )  # processor for HF, transform for timm
        self.hooks, self.feature_cache = [], {}

        self._load_model_and_processor()

        if not self.model_name.startswith("timm_"):
            self._setup_hooks()

        self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze model
        logger.info(f"Loaded and froze {self.model_name} on {self.device}.")

    def _load_model_and_processor(self):
        logger.info(f"Loading model and processor for: {self.model_name}")
        hf_model_id = None
        timm_model_name = None

        if self.model_name == "dinov2":
            hf_model_id = "facebook/dinov2-base"
        elif self.model_name == "ijepa":
            hf_model_id = "facebook/vit-huge-patch14-224-ijepa"
        elif self.model_name == "mocov3":
            timm_model_name = "vit_base_patch16_224.mocov3_in1k"
        elif self.model_name == "supervised_vit":
            hf_model_id = "google/vit-base-patch16-224"

        # General DINOv2 shorthands
        elif self.model_name == "dinov2_vits14":
            hf_model_id = "facebook/dinov2-small"
        elif self.model_name == "dinov2_vitb14":
            hf_model_id = "facebook/dinov2-base"
        elif self.model_name == "dinov2_vitl14":
            hf_model_id = "facebook/dinov2-large"
        elif self.model_name == "dinov2_vitg14":
            hf_model_id = "facebook/dinov2-giant"

        # General timm prefix
        elif self.model_name.startswith("timm_"):
            timm_model_name = self.model_name.split("timm_", 1)[1]

        elif self.model_name.startswith("vit_"):
            hf_model_id = f"google/{self.model_name.replace('_','-')}"
        else:
            hf_model_id = self.model_name

        if timm_model_name:
            logger.info(f"Attempting to load timm model: {timm_model_name}")
            self.model = timm.create_model(
                timm_model_name, pretrained=True, num_classes=0, features_only=False
            )
            data_cfg = timm.data.resolve_model_data_config(self.model)
            self.transform = timm.data.create_transform(**data_cfg, is_training=False)
            logger.info(
                f"Loaded timm model: {timm_model_name} with its default transform."
            )
        elif hf_model_id:
            logger.info(f"Attempting to load HuggingFace model: {hf_model_id}")
            self.model = AutoModel.from_pretrained(
                hf_model_id, cache_dir=self.cache_dir, output_hidden_states=True
            )
            self.processor = AutoImageProcessor.from_pretrained(
                hf_model_id, cache_dir=self.cache_dir
            )
            logger.info(f"Loaded HuggingFace model: {hf_model_id}")
        else:
            raise ValueError(
                f"Could not determine how to load model: {self.model_name}. Provide a valid name/prefix."
            )

        if self.ckpt_path:
            self._load_ckpt()

    def _load_ckpt(self):
        if not Path(self.ckpt_path).exists():
            logger.error(f"Checkpoint path not found: {self.ckpt_path}")
            return
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        sd = ckpt.get(
            "model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt))
        )
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning(f"Missing keys in ckpt: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in ckpt: {unexpected}")
        logger.info(f"Loaded checkpoint: {self.ckpt_path}")

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

    def extract_features(
        self,
        images: torch.Tensor,
        layers: Optional[List[Union[int, str]]] = None,
        feature_type: str = "cls_token",
    ) -> Dict[str, torch.Tensor]:
        """Extracts features from specified layers and types of the model."""
        images_on_device = images.to(self.device)
        self.feature_cache = {}

        with torch.no_grad():

            if self.processor:
                inputs = self.processor(images_on_device, return_tensors="pt").to(
                    self.device
                )
                model_outputs = self.model(**inputs)
            elif self.transform:
                processed_images = torch.stack(
                    [self.transform(img) for img in images_on_device]
                ).to(self.device)
                model_outputs = self.model(processed_images)
            else:
                model_outputs = self.model(images_on_device)

        extracted_features = {}
        num_transformer_layers = 0
        if hasattr(self.model.config, "num_hidden_layers"):
            num_transformer_layers = self.model.config.num_hidden_layers
        elif hasattr(self.model, "blocks"):  # Timm
            num_transformer_layers = len(self.model.blocks)

        if not layers:
            target_indices_eff = [num_transformer_layers]
        else:
            target_indices_eff = []
            for l_spec in layers:
                if isinstance(l_spec, str) and l_spec == "final":
                    target_indices_eff.append(num_transformer_layers)
                elif isinstance(l_spec, int):
                    actual_idx = (
                        l_spec + 1
                        if l_spec >= 0
                        else num_transformer_layers + 1 + l_spec
                    )
                    target_indices_eff.append(actual_idx)

        for l_idx_hf_style in target_indices_eff:
            output_tensor = None
            hook_layer_name = f"layer_{l_idx_hf_style - 1}"

            if hasattr(model_outputs, "hidden_states") and 0 <= l_idx_hf_style < len(
                model_outputs.hidden_states
            ):
                output_tensor = model_outputs.hidden_states[l_idx_hf_style]
            elif hook_layer_name in self.feature_cache:
                output_tensor = self.feature_cache[hook_layer_name]

            elif l_idx_hf_style == num_transformer_layers:
                if hasattr(model_outputs, "last_hidden_state"):
                    output_tensor = model_outputs.last_hidden_state
                elif isinstance(
                    model_outputs, torch.Tensor
                ) and self.model_name.startswith("timm_"):

                    output_tensor = model_outputs

            if output_tensor is None:
                continue

            feature_key_name = (
                f"layer_{l_idx_hf_style -1}_out"
                if l_idx_hf_style > 0
                else "embedding_out"
            )

            if output_tensor.dim() == 3:  # Expected shape: [Batch, SeqLen, HiddenDim]
                if feature_type == "cls_token":
                    extracted_features[feature_key_name] = output_tensor[:, 0]
                elif feature_type == "patch_mean":
                    extracted_features[feature_key_name] = output_tensor[:, 1:].mean(
                        dim=1
                    )
                elif feature_type == "patch_tokens":
                    extracted_features[feature_key_name] = output_tensor[:, 1:]
                else:
                    extracted_features[feature_key_name] = output_tensor[:, 0]
            else:
                extracted_features[feature_key_name] = output_tensor

        if not extracted_features:
            final_output_candidate = getattr(
                model_outputs, "last_hidden_state", None
            ) or (
                model_outputs
                if isinstance(model_outputs, torch.Tensor)
                else getattr(model_outputs, "pooler_output", None)
            )
            if final_output_candidate is not None:
                extracted_features["final_fallback"] = (
                    final_output_candidate[:, 0]
                    if final_output_candidate.dim() == 3
                    else final_output_candidate
                )
                logger.info("Used a fallback mechanism for final feature extraction.")
        return extracted_features

    def get_feature_dim(
        self,
        layers: Optional[List[Union[int, str]]] = None,
        feature_type: str = "cls_token",
    ) -> int:
        """Determines feature dimension by a dummy forward pass."""
        h = w = 224
        if (
            self.processor
            and hasattr(self.processor, "size")
            and isinstance(self.processor.size, dict)
            and "height" in self.processor.size
        ):
            h = self.processor.size["height"]
            w = self.processor.size["width"]
        elif self.transform:
            pass

        dummy_input = torch.randn(1, 3, h, w)

        features_dict = self.extract_features(
            dummy_input, layers=layers or [-1], feature_type=feature_type
        )

        if not features_dict:
            if hasattr(self.model.config, "hidden_size"):
                return self.model.config.hidden_size
            logger.warning(
                "Could not determine feature dimension from dummy pass or config. Returning default 768."
            )
            return 768

        first_feature_tensor = list(features_dict.values())[0]
        return first_feature_tensor.shape[-1]

    def __del__(self):

        for hook in self.hooks:
            hook.remove()


def load_image_feature_extractor(config: Dict) -> ReconstructionFeatureExtractor:
    """Factory function to create ImageFeatureExtractor from a configuration dictionary."""
    return ReconstructionFeatureExtractor(
        model_name=config.get("model_name", "supervised_vit"),
        ckpt_path=config.get("checkpoint_path"),
        device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        cache_dir=config.get("cache_dir"),
    )
