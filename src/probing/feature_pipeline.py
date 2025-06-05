import torch
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Tuple

from .base_pipeline import BasePipeline
from .target_utils import prepare_targets_for_task
from ..models.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class ProbingDataset(Dataset):
    """Dataset for probing experiments with pre-extracted features"""

    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[Dict] = None,
    ):
        assert len(features) == len(targets), "Features and targets must have same length"
        self.features = features
        self.targets = targets
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {"features": self.features[idx], "targets": self.targets[idx]}
        for key, values in self.metadata.items():
            if isinstance(values, (list, tuple, torch.Tensor)) and len(values) > idx:
                sample[key] = values[idx]
        return sample


class ProbingPipeline(BasePipeline):
    """Pipeline for extracting features and preparing probing datasets"""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir)
        self.feature_extractor = feature_extractor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

    def _process_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        images = batch["image"].to(self.device)
        feats_dict = self.feature_extractor.extract_features(
            images=images,
            layers=getattr(self, 'layers', None),
            feature_type=getattr(self, 'feature_type', 'cls_token'),
        )
        feats = (list(feats_dict.values())[0] if len(feats_dict) == 1 else torch.cat(list(feats_dict.values()), dim=-1))
        meta = {
            'categories': list(batch['category']),
            'model_ids': list(batch['model_id']),
            'view_indices': batch['view_idx'].tolist(),
            'azimuths': batch['viewpoint'][:, 0].tolist(),
            'elevations': batch['viewpoint'][:, 1].tolist(),
        }
        targets = prepare_targets_for_task(meta, getattr(self, 'task_type', 'viewpoint_regression'))
        return feats, targets, meta

    def create_datasets(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        layers: Optional[List[int]] = None,
        feature_type: str = 'cls_token',
        task_type: str = 'viewpoint_regression',
        experiment_name: Optional[str] = None,
    ) -> Tuple[ProbingDataset, ProbingDataset, ProbingDataset]:
        """Create train/val/test ProbingDataset using BasePipeline.extract"""
        self.layers = layers
        self.feature_type = feature_type
        self.task_type = task_type
        datasets = []
        for split, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            cache_key = f"{experiment_name}_{split}" if experiment_name else None
            feats, targets, meta = self.extract(loader, cache_key=cache_key)
            dataset = ProbingDataset(features=feats, targets=targets, metadata=meta)
            logger.info(f"{split.upper()} dataset: {len(dataset)} samples")
            datasets.append(dataset)
        return tuple(datasets)
