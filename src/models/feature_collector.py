import logging
import torch
from typing import Any, Dict, List, Union, Optional

logger = logging.getLogger(__name__)


class FeatureCollector:
    """
    Helper to extract features from model outputs and hook cache.
    """

    def __init__(
        self,
        outputs: Any,
        cache: Dict[str, torch.Tensor],
        model: Any,
        model_name: str,
    ):
        self.outputs = outputs
        self.cache = cache
        self.model = model
        self.model_name = model_name

    def _num_layers(self) -> int:
        if hasattr(self.model.config, "num_hidden_layers"):
            return self.model.config.num_hidden_layers
        if hasattr(self.model, "blocks"):
            return len(self.model.blocks)
        return 0

    def _resolve_indices(self, layers: Optional[List[Union[int, str]]]) -> List[int]:
        nl = self._num_layers()
        if not layers:
            return [nl]
        indices: List[int] = []
        for spec in layers:
            if isinstance(spec, str) and spec == "final":
                indices.append(nl)
            elif isinstance(spec, int):
                idx = spec + 1 if spec >= 0 else nl + 1 + spec
                indices.append(idx)
            else:
                logger.warning(f"Unknown layer spec '{spec}', skipping.")
        return indices

    def collect(
        self,
        layers: Optional[List[Union[int, str]]],
        feature_type: str = "cls_token",
    ) -> Dict[str, torch.Tensor]:
        results: Dict[str, torch.Tensor] = {}
        nl = self._num_layers()
        for idx in self._resolve_indices(layers):
            tensor: Optional[torch.Tensor] = None
            layer_key = idx - 1
            hook_name = f"layer_{layer_key}"
            # try hidden_states
            hs = getattr(self.outputs, "hidden_states", None)
            if hs is not None and 0 <= idx < len(hs):
                tensor = hs[idx]
            elif hook_name in self.cache:
                tensor = self.cache[hook_name]
            elif idx == nl:
                if hasattr(self.outputs, "last_hidden_state"):
                    tensor = self.outputs.last_hidden_state
                elif isinstance(
                    self.outputs, torch.Tensor
                ) and self.model_name.startswith("timm_"):
                    tensor = self.outputs
            if tensor is None:
                continue
            key_name = "embedding_out" if layer_key < 0 else f"layer_{layer_key}_out"
            # slice features
            if tensor.dim() == 3:
                if feature_type == "cls_token":
                    results[key_name] = tensor[:, 0]
                elif feature_type == "patch_mean":
                    results[key_name] = tensor[:, 1:].mean(dim=1)
                elif feature_type == "patch_tokens":
                    results[key_name] = tensor[:, 1:]
                else:
                    results[key_name] = tensor[:, 0]
            else:
                results[key_name] = tensor
        # fallback if empty
        if not results:
            candidate = getattr(self.outputs, "last_hidden_state", None)
            if candidate is None:
                if isinstance(self.outputs, torch.Tensor):
                    candidate = self.outputs
                else:
                    candidate = getattr(self.outputs, "pooler_output", None)
            if candidate is not None:
                key = "final_fallback"
                results[key] = candidate[:, 0] if candidate.dim() == 3 else candidate
                logger.info("Used a fallback mechanism for final feature extraction.")
        return results
