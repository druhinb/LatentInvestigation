"""Data preprocessing pipeline for the probing experiments"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)


class ProbingDataset(Dataset):
    """Dataset for probing experiments with pre-extracted features"""

    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[Dict] = None,
    ):
        assert len(features) == len(
            targets
        ), "Features and targets must have same length"
        self.features = features
        self.targets = targets
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {"features": self.features[idx], "targets": self.targets[idx]}

        # Add metadata if available
        for key, values in self.metadata.items():
            if (
                isinstance(values, (list, tuple, np.ndarray, torch.Tensor))
                and len(values) > idx
            ):
                sample[key] = values[idx]

        return sample


class FeatureExtractorPipeline:
    """Pipeline for extracting features from models for probing"""

    def __init__(
        self,
        feature_extractor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
    ):
        self.feature_extractor = feature_extractor
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_features_from_dataloader(
        self,
        dataloader: DataLoader,
        layers: Optional[List[int]] = None,
        feature_type: str = "cls_token",
        task_type: str = "viewpoint_regression",
        cache_key: Optional[str] = None,
        force_recompute: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Extract features from a dataloader"""
        # Check cache first
        if self.cache_dir and cache_key and not force_recompute:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                logger.info(f"Loading cached features from {cache_file}")
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                return (
                    cached_data["features"],
                    cached_data["targets"],
                    cached_data["metadata"],
                )

        all_features = []
        metadata = {
            "categories": [],
            "model_ids": [],
            "view_indices": [],
            "azimuths": [],
            "elevations": [],
        }

        logger.info(f"Extracting features from {len(dataloader)} batches...")

        self.feature_extractor.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                images = batch["image"].to(self.device)

                # Extract features
                features_dict = self.feature_extractor.extract_features(
                    images=images, layers=layers, feature_type=feature_type
                )

                # we need to handle just one layer or possibly multiple
                if len(features_dict) == 1:
                    features = list(features_dict.values())[0]
                else:
                    features = torch.cat(list(features_dict.values()), dim=-1)

                all_features.append(features.cpu())

                # Collect metadata
                metadata["categories"].extend(batch["category"])
                metadata["model_ids"].extend(batch["model_id"])
                metadata["view_indices"].extend(batch["view_idx"].tolist())
                metadata["azimuths"].extend(batch["viewpoint"][:, 0].tolist())
                metadata["elevations"].extend(batch["viewpoint"][:, 1].tolist())

        # Concatenate features and prepare the prediction targets based on them
        features = torch.cat(all_features, dim=0)
        targets = prepare_targets_for_task(metadata, task_type)

        logger.info(f"Extracted features shape: {features.shape}")
        logger.info(f"Targets shape: {targets.shape}")

        # Cache results
        if self.cache_dir and cache_key:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            cache_data = {
                "features": features,
                "targets": targets,
                "metadata": metadata,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cached features to {cache_file}")

        return features, targets, metadata

    def create_probing_datasets(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        layers: Optional[List[int]] = None,
        feature_type: str = "cls_token",
        task_type: str = "viewpoint_regression",
        experiment_name: Optional[str] = None,
    ) -> Tuple[ProbingDataset, ProbingDataset, ProbingDataset]:
        """Create probing datasets from train/val/test dataloaders"""
        datasets = []

        for split, dataloader in [
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ]:
            cache_key = f"{experiment_name}_{split}" if experiment_name else None

            features, targets, metadata = self.extract_features_from_dataloader(
                dataloader=dataloader,
                layers=layers,
                feature_type=feature_type,
                task_type=task_type,
                cache_key=cache_key,
            )

            dataset = ProbingDataset(
                features=features, targets=targets, metadata=metadata
            )
            datasets.append(dataset)
            logger.info(f"{split.upper()} dataset: {len(dataset)} samples")

        return tuple(datasets)


def prepare_targets_for_task(
    metadata: Dict, task_type: str, output_format: str = "tensor"
) -> Union[torch.Tensor, np.ndarray]:
    """Prepare targets based on the probing task"""
    if task_type == "viewpoint_regression":
        azimuths = torch.tensor(metadata["azimuths"], dtype=torch.float32)
        elevations = torch.tensor(metadata["elevations"], dtype=torch.float32)
        targets = torch.stack([azimuths, elevations], dim=1)

    elif task_type == "shape_classification":
        categories = metadata["categories"]
        unique_categories = sorted(list(set(categories)))
        cat_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        targets = torch.tensor(
            [cat_to_idx[cat] for cat in categories], dtype=torch.long
        )

    elif task_type == "view_classification":
        azimuths = np.array(metadata["azimuths"])
        elevations = np.array(metadata["elevations"])

        # we need to discretely bin azimuths and elevations
        azimuth_bins = np.linspace(0, 360, 9)
        elevation_bins = np.linspace(-30, 30, 4)

        azimuth_indices = np.digitize(azimuths, azimuth_bins) - 1
        elevation_indices = np.digitize(elevations, elevation_bins) - 1

        # Combine into single class index
        targets = azimuth_indices * 3 + elevation_indices
        targets = torch.tensor(targets, dtype=torch.long)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return targets.numpy() if output_format == "numpy" else targets


def create_probing_dataloaders(
    train_dataset: ProbingDataset,
    val_dataset: ProbingDataset,
    test_dataset: ProbingDataset,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for probing datasets"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
