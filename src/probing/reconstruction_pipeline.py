import torch
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import logging

from .base_pipeline import BasePipeline
from ..models.reconstruction_feature_extractor import ReconstructionFeatureExtractor

logger = logging.getLogger(__name__)


class ReconstructionDataset(Dataset):
    """Dataset wrapper for reconstruction pipeline outputs"""

    def __init__(
        self, view_data: torch.Tensor, voxels: torch.Tensor, ids: list, cats: list
    ):
        assert len(view_data) == len(voxels) == len(ids) == len(cats)
        self.view_data = view_data
        self.voxels = voxels
        self.ids = ids
        self.cats = cats

    def __len__(self) -> int:
        return len(self.view_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "processed_views": self.view_data[idx],
            "target_voxels": self.voxels[idx],
            "model_id": self.ids[idx],
            "category": self.cats[idx],
        }


class ReconstructionPipeline(BasePipeline):
    """Pipeline for extracting and caching multi-view reconstruction data"""

    def __init__(
        self,
        image_pipeline: ReconstructionFeatureExtractor,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir)
        self.image_pipeline = image_pipeline
        self.device = device

    def _process_batch(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # batch keys: images [B, NV, C, H, W], camera_params [B, NV, 5], voxel_gt [B, ...]
        images = batch["images"]
        cam_params = batch["camera_params"]
        voxels = batch["voxel_gt"]
        B, NV, C, H, W = images.shape
        feats_dict = self.image_pipeline.extract_features_memopt(
            images,
            layers=getattr(self, "layers", None),
            feature_type=getattr(self, "feature_type", "cls_token"),
            max_batch_size=32,  # Limit batch size for memory
        )
        feats_flat = (
            list(feats_dict.values())[0] if feats_dict else torch.empty(B * NV, 0)
        )
        feats_views = feats_flat.view(B, NV, -1)
        # process camera params
        cam_feats = cam_params.to(self.device).cpu()
        # combine
        combined = torch.cat((feats_views, cam_feats), dim=-1)
        combined_flat = combined.view(B, -1)

        # metadata
        meta = {
            "model_ids": list(batch["model_id"]),
            "categories": list(batch["category"]),
        }
        return combined_flat, voxels.to(self.device), meta

    def create_dataset(
        self,
        dataloader: DataLoader,
        layers: Optional[list] = None,
        feature_type: str = "cls_token",
        cache_key: Optional[str] = None,
        force_recompute: bool = False,
    ) -> ReconstructionDataset:
        # set processing parameters
        self.layers = layers
        self.feature_type = feature_type
        feats, voxels, meta = self.extract(
            dataloader, cache_key=cache_key, force_recompute=force_recompute
        )
        return ReconstructionDataset(
            feats, voxels, meta.get("model_ids", []), meta.get("categories", [])
        )
