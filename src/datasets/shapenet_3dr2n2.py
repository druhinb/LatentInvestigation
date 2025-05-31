"""
ShapeNet 3D-R2N2 Dataset Implementation
Dataset from: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io as sio
from omegaconf import DictConfig
from hydra.utils import instantiate


class ShapeNet3DR2N2(Dataset):
    """3D-R2N2 ShapeNet rendered dataset"""

    CATEGORIES = {
        "02691156": "airplane",
        "02828884": "bench",
        "02933112": "cabinet",
        "02958343": "car",
        "03001627": "chair",
        "03211117": "display",
        "03636649": "lamp",
        "03691459": "speaker",
        "04090263": "rifle",
        "04256520": "sofa",
        "04379243": "table",
        "04401088": "phone",
        "04530566": "watercraft",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        categories: Optional[List[str]] = None,
        transform=None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_seed: int = 42,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Load or create split file
        split_file = self.root / f"{split}.lst"
        if not split_file.exists():
            print(f"Split file {split_file} not found. Creating splits...")
            self._create_split_files(train_ratio, val_ratio, split_seed)

        with open(split_file, "r") as f:
            self.model_list = [line.strip() for line in f]

        # Filter categories
        if categories:
            cat_ids = [k for k, v in self.CATEGORIES.items() if v in categories]
            self.model_list = [m for m in self.model_list if m.split("/")[0] in cat_ids]

        self.samples = self._prepare_samples()

    def _create_split_files(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42
    ):
        """Create train/val/test split files if they don't exist

        Args:
            train_ratio: Proportion for training set (default: 0.7)
            val_ratio: Proportion for validation set (default: 0.15)
            seed: Random seed for reproducible splits (default: 42)
        """
        import random

        # Find all available models
        all_models = []
        for cat_id in self.CATEGORIES.keys():
            cat_dir = self.root / cat_id
            if cat_dir.exists():
                for obj_dir in cat_dir.iterdir():
                    if obj_dir.is_dir():
                        model_id = f"{cat_id}/{obj_dir.name}"
                        # Check if rendering directory exists
                        if (obj_dir / "rendering").exists():
                            all_models.append(model_id)

        if not all_models:
            raise ValueError(f"No models found in {self.root}")

        print(
            f"Found {len(all_models)} models across {len(self.CATEGORIES)} categories"
        )

        # Shuffle for random splits
        random.seed(seed)
        random.shuffle(all_models)

        # Calculate split sizes
        n_total = len(all_models)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_models = all_models[:n_train]
        val_models = all_models[n_train : n_train + n_val]
        test_models = all_models[n_train + n_val :]

        # Write split files
        splits = {"train": train_models, "val": val_models, "test": test_models}

        for split_name, models in splits.items():
            split_file = self.root / f"{split_name}.lst"
            with open(split_file, "w") as f:
                for model in models:
                    f.write(f"{model}\n")
            print(f"Created {split_file} with {len(models)} models")

        print(f"\nSplit summary:")
        print(f"  Train: {len(train_models)} models ({len(train_models)/n_total:.1%})")
        print(f"  Val: {len(val_models)} models ({len(val_models)/n_total:.1%})")
        print(f"  Test: {len(test_models)} models ({len(test_models)/n_total:.1%})")

        return splits

    def _prepare_samples(self):
        """Prepare all samples with viewpoint info"""
        samples = []

        for model_id in self.model_list:
            cat_id, obj_id = model_id.split("/")
            rendering_dir = self.root / cat_id / obj_id / "rendering"
            metadata_file = rendering_dir / "rendering_metadata.txt"

            # Read camera parameters
            if metadata_file.exists():
                camera_params = self._read_camera_params(metadata_file)
            else:
                # Use default parameters if metadata missing
                camera_params = self._get_default_params()

            # Each object has 24 rendered views
            for i in range(24):
                img_path = rendering_dir / f"{i:02d}.png"
                if img_path.exists():
                    samples.append(
                        {
                            "img_path": str(img_path),
                            "category": self.CATEGORIES.get(cat_id, cat_id),
                            "model_id": model_id,
                            "view_idx": i,
                            "azimuth": camera_params[i]["azimuth"],
                            "elevation": camera_params[i]["elevation"],
                        }
                    )

        return samples

    def _read_camera_params(self, metadata_file):
        """Read camera parameters from metadata"""
        params = []
        with open(metadata_file, "r") as f:
            for i, line in enumerate(f):
                if i >= 24:  # Only 24 views
                    break
                parts = line.strip().split()
                # Format: azimuth elevation tilt distance
                params.append(
                    {"azimuth": float(parts[0]), "elevation": float(parts[1])}
                )
        return params

    def _get_default_params(self):
        """Default camera parameters (24 views around object)"""
        params = []
        for i in range(24):
            azimuth = i * 15.0  # 0, 15, 30, ..., 345
            elevation = 30.0 if i % 2 == 0 else -30.0  # Alternate elevations
            params.append({"azimuth": azimuth, "elevation": elevation})
        return params

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Normalize viewpoint to [-1, 1]
        azimuth_norm = (sample["azimuth"] / 180.0) - 1.0
        elevation_norm = sample["elevation"] / 90.0
        viewpoint = torch.tensor([azimuth_norm, elevation_norm], dtype=torch.float32)

        return {
            "image": img,
            "viewpoint": viewpoint,
            "category": sample["category"],
            "model_id": sample["model_id"],
            "view_idx": sample["view_idx"],
        }


def prepare_3dr2n2_dataset(cfg: DictConfig) -> str:
    """Download and extract 3D-R2N2 dataset"""
    import tarfile
    import urllib.request

    # Get preparation config
    prep_cfg = cfg.get("preparation", {})
    tar_path_config = prep_cfg.get("tar_path", None)
    url = prep_cfg.get("url", "http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz")
    auto_download = prep_cfg.get("auto_download", True)

    # Determine data directory and target
    data_dir = Path(cfg.root).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = Path(cfg.root)

    # If dataset already exists, return early
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Dataset already exists at: {extract_dir}")
        return str(extract_dir)

    # Determine tar file path
    if tar_path_config:
        tar_path = Path(tar_path_config)
        if not tar_path.exists():
            raise FileNotFoundError(f"Specified tar file not found: {tar_path}")
        print(f"Using existing tar file: {tar_path}")
    else:
        tar_path = data_dir / "ShapeNetRendering.tgz"

        if not tar_path.exists():
            if not auto_download:
                raise ValueError(
                    f"Tar file not found at {tar_path} and auto_download is disabled. "
                    f"Either provide dataset.preparation.tar_path or enable auto_download."
                )

            print(f"Downloading ShapeNet 3D-R2N2 dataset from {url}...")
            try:
                urllib.request.urlretrieve(url, tar_path)
                print(f"Downloaded to {tar_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset: {e}")

    # Validate tar file
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            # Quick validation - check if it's a valid tar.gz
            pass
    except Exception as e:
        raise ValueError(f"Invalid tar file {tar_path}: {e}")

    # Extract dataset
    print("Extracting dataset...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        print(f"Extracted to {extract_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract dataset: {e}")

    # Verify extraction
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        raise RuntimeError(
            f"Dataset extraction failed - directory empty: {extract_dir}"
        )

    return str(extract_dir)


def create_3dr2n2_dataloaders(
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders"""

    # Instantiate transform from config
    transform = instantiate(cfg.transform) if cfg.get("transform") else None

    # Get dataloader config
    dataloader_cfg = cfg.get("dataloader", {})
    batch_size = dataloader_cfg.get("batch_size", 32)
    num_workers = dataloader_cfg.get("num_workers", 4)
    pin_memory = dataloader_cfg.get("pin_memory", True)
    drop_last = dataloader_cfg.get("drop_last", True)

    # Get split configuration
    splits_cfg = cfg.get("splits", {})

    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = ShapeNet3DR2N2(
            root=cfg.root,
            split=split,
            categories=cfg.get("categories", None),
            transform=transform,
            # Pass split configuration
            train_ratio=splits_cfg.get("train_ratio", 0.7),
            val_ratio=splits_cfg.get("val_ratio", 0.15),
            split_seed=splits_cfg.get("seed", 42),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(drop_last and split == "train"),
        )

    return loaders["train"], loaders["val"], loaders["test"]


def get_dataset_from_config(cfg: DictConfig) -> ShapeNet3DR2N2:
    """Create a single dataset instance from Hydra config"""
    transform = instantiate(cfg.transform) if cfg.get("transform") else None

    # Get split configuration
    splits_cfg = cfg.get("splits", {})

    return ShapeNet3DR2N2(
        root=cfg.root,
        split=cfg.get("split", "train"),
        categories=cfg.get("categories", None),
        transform=transform,
        train_ratio=splits_cfg.get("train_ratio", 0.7),
        val_ratio=splits_cfg.get("val_ratio", 0.15),
        split_seed=splits_cfg.get("seed", 42),
    )
