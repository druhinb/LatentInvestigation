import os
import json
from tqdm import tqdm
import numpy as np
import tarfile
import torch
import urllib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from omegaconf import DictConfig
from hydra.utils import instantiate
from src.utils.binvox_utils import read_as_3d_array


class ShapeNet3DR2N2Reconstruction(Dataset):
    """
    3D-R2N2 ShapeNet dataset for 3D reconstruction from multiple views.
    We load 24 views, their camera parameters, and a ground truth voxel model.
    """

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
    NUM_VIEWS = 24

    def __init__(
        self,
        root: str,
        voxel_root: Optional[str] = None,
        split: str = "train",
        categories: Optional[List[str]] = None,
        transform=None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_seed: int = 42,
        subset_percentage: Optional[float] = None,
        normalize_cameras: bool = True,
    ):
        self.root = Path(root)

        # if voxel_root isn't provided, assume the stuff is in the same folder
        self.voxel_root = Path(voxel_root) if voxel_root else self.root
        self.split = split
        self.transform = transform
        self.normalize_cameras = normalize_cameras

        # create dataset splits
        split_file = self.root / f"{split}.lst"
        if not split_file.exists():
            print(
                f"Split file {split_file} not found. Creating splits based on models in {self.root}..."
            )
            self._create_split_files(train_ratio, val_ratio, split_seed)

        with open(split_file, "r") as f:
            # each line contains cat_id/obj_id
            self.model_id_list = [line.strip() for line in f]

        # only keep the categories that were reqeested
        if categories:
            cat_ids_to_keep = [k for k, v in self.CATEGORIES.items() if v in categories]
            self.model_id_list = [
                m for m in self.model_id_list if m.split("/")[0] in cat_ids_to_keep
            ]

        self.samples = self._prepare_samples()

        # return a randomized subset of the data if requested
        if subset_percentage is not None:
            if not (0 < subset_percentage <= 1.0):
                raise ValueError("subset_percentage must be between 0.0 and 1.0")
            import random

            random.seed(split_seed)
            num_samples = int(len(self.samples) * subset_percentage)
            self.samples = random.sample(self.samples, num_samples)
            print(
                f"Using {subset_percentage*100:.2f}% of {split} data: {len(self.samples)} samples."
            )

    def _create_split_files(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42
    ):
        """Create train/val/test split files if they don't exist"""
        import random

        all_models = []
        for cat_id in self.CATEGORIES.keys():
            cat_dir = self.root / cat_id
            if cat_dir.exists():
                for obj_dir in cat_dir.iterdir():
                    if obj_dir.is_dir():
                        all_models.append(f"{cat_id}/{obj_dir.name}")

        if not all_models:
            raise ValueError(
                f"No models with renderings and voxels found in {self.root} (and {self.voxel_root})"
            )

        print(
            f"Found {len(all_models)} models across {len(self.CATEGORIES)} categories."
        )

        random.seed(seed)
        random.shuffle(all_models)

        n_total = len(all_models)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_models = all_models[:n_train]
        val_models = all_models[n_train : n_train + n_val]
        test_models = all_models[n_train + n_val :]

        splits = {"train": train_models, "val": val_models, "test": test_models}
        for split_name, models in splits.items():
            split_file = self.root / f"{split_name}.lst"
            with open(split_file, "w") as f:
                for model in models:
                    f.write(f"{model}\n")
            print(f"Created {split_file} with {len(models)} models")

    def _prepare_samples(self):
        """
        Prepare samples. We also verify that the voxel data exists for each model_id.
        """
        prepared_samples = []
        print("Preparing samples...")

        for model_id_str in tqdm(self.model_id_list):
            cat_id, obj_id = model_id_str.split("/")
            voxel_path = self.voxel_root / cat_id / obj_id / "model.binvox"
            rendering_dir = self.root / cat_id / obj_id / "rendering"

            if voxel_path.exists() and rendering_dir.exists():
                if not (rendering_dir / "00.png").exists():
                    continue
                prepared_samples.append(
                    {
                        "model_id_str": model_id_str,
                        "category": self.CATEGORIES.get(cat_id, cat_id),
                        "voxel_path": str(voxel_path),
                        "rendering_dir": str(rendering_dir),
                    }
                )
        print(f"Prepared {len(prepared_samples)} valid samples for split {self.split}.")
        return prepared_samples

    def _read_camera_params_from_file(
        self, metadata_file: Path
    ) -> List[Dict[str, float]]:
        """Aggregate the rendering metadata into a list of dicts for each view"""

        params_all_views = []
        with open(metadata_file, "r") as f:
            for i, line in enumerate(f):
                if i >= self.NUM_VIEWS:
                    break
                parts = line.strip().split()
                # data should be in: azimuth elevation in-plane-rotation distance field-of-view
                params_all_views.append(
                    {
                        "azimuth": float(parts[0]),  # degrees
                        "elevation": float(parts[1]),  # degrees
                        "in_plane_rotation": float(parts[2]),  # degrees
                        "distance": float(parts[3]),  # blender units ig???
                        "fov": float(parts[4]),  # degrees
                    }
                )

        return params_all_views

    def _normalize_camera_params(
        self, camera_params_list: List[Dict[str, float]]
    ) -> torch.Tensor:
        """Normalize camera parameters and convert to a tensor."""
        processed_params = []
        for params in camera_params_list:
            az_rad = np.deg2rad(params["azimuth"])
            el_rad = np.deg2rad(params["elevation"])
            ipr_rad = np.deg2rad(params["in_plane_rotation"])

            # 2.5 mean distance with 0.5 variance according to r2n2 paper
            norm_dist = (params["distance"] - 2.5) / 0.5  # Assuming mean ~2.5, std ~0.5

            # same story, 45 mean 15 std
            norm_fov = (params["fov"] - 45.0) / 15.0  # Assuming mean ~45, std ~15

            view_meta = [
                np.sin(az_rad),
                np.cos(az_rad),
                np.sin(el_rad),
                np.cos(el_rad),
                np.sin(ipr_rad),
                np.cos(ipr_rad),
                norm_dist,
                norm_fov,
            ]
            processed_params.append(view_meta)
        return torch.tensor(
            processed_params, dtype=torch.float32
        )  # shape: [NUM_VIEWS, K_params]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]
        model_id_str = sample_info["model_id_str"]  # cat_id/obj_id
        cat_id, obj_id = model_id_str.split("/")

        # 1. load the voxel occupancy map first (32x32x32)
        voxel_path = Path(sample_info["voxel_path"])

        voxel_data = read_as_3d_array(voxel_path.open("rb")).data.astype(np.float32)
        voxel_tensor = torch.from_numpy(voxel_data).unsqueeze(
            0
        )  # add  a channel dim to make it [1, D, H, W]

        # 2. load all NUM_VIEWS images
        images = []
        rendering_dir = Path(sample_info["rendering_dir"])
        for i in range(self.NUM_VIEWS):
            img_path = rendering_dir / f"{i:02d}.png"
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # stack images into a single tensor: [NUM_VIEWS, C, H, W]
        images_tensor = torch.stack(images)

        # 3. load rendering metadata
        metadata_file = rendering_dir / "rendering_metadata.txt"
        camera_params_list = self._read_camera_params_from_file(metadata_file)

        # 4. process metadata
        if self.normalize_cameras:
            camera_params_tensor = self._normalize_camera_params(camera_params_list)
        else:
            params = [
                [
                    p["azimuth"],
                    p["elevation"],
                    p["in_plane_rotation"],
                    p["distance"],
                    p["fov"],
                ]
                for p in camera_params_list
            ]
            camera_params_tensor = torch.tensor(params, dtype=torch.float32)

        return {
            "model_id": model_id_str,
            "category": sample_info["category"],
            "images": images_tensor,  # [NUM_VIEWS, C, H, W]
            "camera_params": camera_params_tensor,  # [NUM_VIEWS, K_params]
            "voxel_gt": voxel_tensor,  # [1, D, H, W] or [D, H, W]
        }


def prepare_3dr2n2_reconstruction_dataset(cfg: DictConfig) -> Tuple[str, str]:
    """
    Download and extract 3D-R2N2 ShapeNetRendering and ShapeNetVox32 datasets.
    """
    import urllib.request
    import tarfile
    from pathlib import Path

    def download_and_extract(url, tar_path, extract_dir):
        """Helper function to download and extract tar files."""
        if not tar_path.exists():
            print(f"Downloading from {url} to {tar_path}...")
            try:
                urllib.request.urlretrieve(url, tar_path)
                print(f"Downloaded to {tar_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download from {url}: {e}")

        print(f"Extracting {tar_path}...")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            print(f"Extracted to {extract_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract {tar_path}: {e}")

    prep_cfg = cfg.get("preparation", {})

    # ShapeNetRendering dataset configuration
    rendering_url = prep_cfg.get(
        "rendering_url", "http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz"
    )
    rendering_tar_path = Path(
        prep_cfg.get("rendering_tar_path", "ShapeNetRendering.tgz")
    )
    rendering_extract_dir = Path(cfg.render_root)

    if not rendering_extract_dir.exists() or not any(rendering_extract_dir.iterdir()):
        download_and_extract(rendering_url, rendering_tar_path, rendering_extract_dir)

    # ShapeNetVox32 dataset configuration
    voxel_url = prep_cfg.get(
        "voxel_url", "http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz"
    )
    voxel_tar_path = Path(prep_cfg.get("voxel_tar_path", "ShapeNetVox32.tgz"))
    voxel_extract_dir = Path(cfg.get("voxel_root", cfg.render_root))

    if not voxel_extract_dir.exists() or not any(voxel_extract_dir.iterdir()):
        download_and_extract(voxel_url, voxel_tar_path, voxel_extract_dir)

    return tuple(str(rendering_extract_dir), str(voxel_extract_dir))


def create_3dr2n2_reconstruction_dataloaders(
    dataset_config: DictConfig,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    subset_percentage: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create 3D-R2N2 ShapeNet Dataloaders for reconstruction task."""

    transform = (
        instantiate(dataset_config.transform)
        if dataset_config.get("transform")
        else None
    )

    dataloaders = {}
    for split in ["train", "val", "test"]:
        current_subset_percentage = subset_percentage
        if (
            dataset_config.get("splits")
            and dataset_config.splits.get("subset_percentage") is not None
        ):
            current_subset_percentage = dataset_config.splits.subset_percentage

        datasets = ShapeNet3DR2N2Reconstruction(
            root=dataset_config.render_root,  # Path to ShapeNetRendering
            voxel_root=dataset_config.get(
                "voxel_root", dataset_config.render_root
            ),  # Path to ShapeNetVox32
            split=split,
            categories=dataset_config.get("categories"),
            transform=transform,
            train_ratio=dataset_config.get("train_ratio", 0.7),
            val_ratio=dataset_config.get("val_ratio", 0.15),
            split_seed=dataset_config.get("split_seed", 42),
            subset_percentage=current_subset_percentage,
            normalize_cameras=dataset_config.get("normalize_cameras", True),
        )

        dataloader_cfg = dataset_config.get("dataloader", {})
        batch_size = dataloader_cfg.get(
            f"{split}_batch_size", dataloader_cfg.get("batch_size", batch_size)
        )
        num_workers = dataloader_cfg.get("num_workers", num_workers)
        pin_memory = dataloader_cfg.get("pin_memory", pin_memory)

        is_train = split == "train"

        dataloaders[split] = DataLoader(
            datasets,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,  # drop_last usually True for training, False for eval
        )

        print(
            f"Created {split} DataLoader with {len(datasets)} samples, batch size {batch_size}."
        )
        if len(datasets) == 0:
            print(f"WARNING: {split} DataLoader is empty!!! Something went wrong.")

    return dataloaders["train"], dataloaders["val"], dataloaders["test"]


def get_reconstruction_dataset_from_config(
    cfg: DictConfig,
) -> ShapeNet3DR2N2Reconstruction:
    """Create a single ShapeNet3DR2N2Reconstruction dataset instance from Hydra config."""

    transform = instantiate(cfg.transform) if cfg.get("transform") else None

    train_ratio = cfg.get("train_ratio", 0.7)
    val_ratio = cfg.get("val_ratio", 0.15)
    split_seed = cfg.get("split_seed", 42)

    subset_percentage = cfg.get("subset_percentage", None)

    return ShapeNet3DR2N2Reconstruction(
        root=cfg.render_root,
        voxel_root=cfg.get("voxel_root", cfg.render_root),
        split=cfg.get("split", "train"),
        categories=cfg.get("categories", None),
        transform=transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_seed=split_seed,
        subset_percentage=subset_percentage,
        normalize_cameras=cfg.get("normalize_cameras", True),
    )
