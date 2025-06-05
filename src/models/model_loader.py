import logging
from pathlib import Path
from typing import Optional, Tuple

import timm
import torch
from transformers import AutoModel, AutoImageProcessor

from .preprocessors import (
    BasePreprocessor,
    HFPreprocessor,
    TimmPreprocessor,
    IdentityPreprocessor,
)

logger = logging.getLogger(__name__)

# Standard model mapping for loader
MODEL_MAPPINGS = {
    "dinov2": ("facebook/dinov2-base", "hf"),
    "ijepa": ("facebook/vit-huge-patch14-224-ijepa", "hf"),
    "mocov3": ("vit_base_patch16_224.mocov3_in1k", "timm"),
    "supervised_vit": ("google/vit-base-patch16-224", "hf"),
    "dinov2_vits14": ("facebook/dinov2-small", "hf"),
    "dinov2_vitb14": ("facebook/dinov2-base", "hf"),
    "dinov2_vitl14": ("facebook/dinov2-large", "hf"),
    "dinov2_vitg14": ("facebook/dinov2-giant", "hf"),
}


def load_model_and_preprocessor(
    model_name: str,
    ckpt_path: Optional[str],
    device: str,
    cache_dir: Optional[str],
) -> Tuple[torch.nn.Module, BasePreprocessor]:
    """
    Load a vision transformer model (HuggingFace or timm) along with the appropriate preprocessor.

    Returns:
        model: the loaded model in eval mode, untrained for feature extraction
        preprocessor: an instance of BasePreprocessor
    """
    hf_model_id: Optional[str] = None
    timm_model_name: Optional[str] = None

    # determine whether to load HF or timm model
    if model_name in MODEL_MAPPINGS:
        mid, mtype = MODEL_MAPPINGS[model_name]
        if mtype == "hf":
            hf_model_id = mid
        else:
            timm_model_name = mid
    elif model_name.startswith("timm_"):
        timm_model_name = model_name.split("timm_", 1)[1]
    elif model_name.startswith("vit_"):
        hf_model_id = f"google/{model_name.replace('_', '-') }"
    else:
        hf_model_id = model_name

    hf_processor = None
    timm_transform = None
    size: Optional[Tuple[int, ...]] = None

    if timm_model_name:
        logger.info(f"Loading timm model '{timm_model_name}'")
        model = timm.create_model(
            timm_model_name, pretrained=True, num_classes=0, features_only=False
        )
        data_cfg = timm.data.resolve_model_data_config(model)
        timm_transform = timm.data.create_transform(**data_cfg, is_training=False)
        # capture expected input size for dimension inference
        size = data_cfg.get("input_size")
    elif hf_model_id:
        logger.info(f"Loading HuggingFace model '{hf_model_id}'")
        model = AutoModel.from_pretrained(
            hf_model_id, cache_dir=cache_dir, output_hidden_states=True
        )
        hf_processor = AutoImageProcessor.from_pretrained(
            hf_model_id, cache_dir=cache_dir
        )
    else:
        raise ValueError(f"Unknown model name {model_name}")

    # load checkpoint if provided
    if ckpt_path:
        if not Path(ckpt_path).exists():
            logger.error(f"Checkpoint not found at {ckpt_path}")
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            sd = ckpt.get(
                "model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt))
            )
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(f"Missing keys: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys: {unexpected}")
            logger.info(f"Loaded checkpoint from {ckpt_path}")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # select appropriate preprocessor instance
    if hf_processor is not None:
        preprocessor = HFPreprocessor(hf_processor, device)
    elif timm_transform is not None:
        preprocessor = TimmPreprocessor(timm_transform, device, size=size)
    else:
        preprocessor = IdentityPreprocessor(device)

    return model, preprocessor
