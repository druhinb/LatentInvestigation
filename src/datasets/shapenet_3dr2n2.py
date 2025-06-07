"""
ShapeNet 3D-R2N2 Dataset Implementation
Dataset from: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
"""

from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import DataLoader
from PIL import Image
from omegaconf import DictConfig
from hydra.utils import instantiate
import torchvision.transforms as transforms
from .base_dataset import BaseSplitDataset


class ShapeNet3DR2N2(BaseSplitDataset):
    """3D-R2N2 ShapeNet rendered dataset with train/val/test split"""

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
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_seed: int = 42,
        subset_percentage: Optional[float] = None,
        transform=None,
    ):
        super().__init__(
            root=root,
            split=split,
            categories=categories,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_seed=split_seed,
            subset_percentage=subset_percentage,
        )
        self.transform = transform

    def _gather_all_models(self) -> List[str]:
        """List all models with rendering directory available"""
        models = []
        for cat_id in self.CATEGORIES.keys():
            cat_dir = self.root / cat_id
            if not cat_dir.exists():
                continue
            for obj_dir in cat_dir.iterdir():
                if obj_dir.is_dir() and (obj_dir / "rendering").exists():
                    models.append(f"{cat_id}/{obj_dir.name}")
        return models

    def _prepare_samples(self) -> List[Dict[str, Any]]:
        """Build sample list for each view of each model in the split"""
        samples = []
        for model_id in self.model_list:
            cat_id, obj_id = model_id.split("/")
            rendering_dir = self.root / cat_id / obj_id / "rendering"
            metadata_file = rendering_dir / "rendering_metadata.txt"
            camera_params = (
                self._read_camera_params(metadata_file)
                if metadata_file.exists()
                else self._get_default_params()
            )
            for i in range(len(camera_params)):
                img_path = rendering_dir / f"{i:02d}.png"
                if img_path.exists():
                    az, el = camera_params[i]["azimuth"], camera_params[i]["elevation"]
                    samples.append(
                        {
                            "img_path": str(img_path),
                            "category": self.CATEGORIES.get(cat_id, cat_id),
                            "model_id": model_id,
                            "view_idx": i,
                            "azimuth": az,
                            "elevation": el,
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

    def __getitem__(self, idx):  # no change
        sample = super().__getitem__(idx)

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
    dataset_config: DictConfig,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    subset_percentage: Optional[float] = None,  # New parameter
    post_transform=None,  # transform should input PIL and output Tensor
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create 3D-R2N2 ShapeNet dataloaders"""

    transform = (
        instantiate(dataset_config.transform)
        if dataset_config.get("transform")
        else None
    )

    # Compose preprocessing and main transforms if both are provided
    if post_transform and transform:
        transform = transforms.Compose(
            [
                transform,  # PIL → Tensor
                transforms.ToPILImage(),  # Need to compose back
                post_transform,  # PIL → Tensor
            ]
        )
    elif post_transform:
        transform = post_transform

    datasets = {}
    for split in ["train", "val", "test"]:
        dataset = ShapeNet3DR2N2(
            root=dataset_config.root,
            split=split,
            categories=dataset_config.get("categories"),
            transform=transform,
            train_ratio=dataset_config.get("train_ratio", 0.7),
            val_ratio=dataset_config.get("val_ratio", 0.15),
            split_seed=dataset_config.get("split_seed", 42),
            subset_percentage=subset_percentage,  # Pass the new parameter
        )
        datasets[split] = dataset

    # Get dataloader specific configs, falling back to function args if not present
    dataloader_cfg = dataset_config.get("dataloader", {})
    effective_batch_size = dataloader_cfg.get("batch_size", batch_size)
    effective_num_workers = dataloader_cfg.get("num_workers", num_workers)
    effective_pin_memory = dataloader_cfg.get("pin_memory", pin_memory)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


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
