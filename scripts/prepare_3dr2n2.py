#!/usr/bin/env python
"""Prepare 3D-R2N2 ShapeNet dataset using Hydra configuration"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from src.datasets.shapenet_3dr2n2 import (
    prepare_3dr2n2_dataset,
    create_3dr2n2_dataloaders,
    get_dataset_from_config,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Prepares, creates, and tests 3D-R2N2 ShapeNet dataset using Hydra configuration

    Usage examples:
    # Use default config
    python prepare_3dr2n2.py

    # Use preexisting tar fileline
    python prepare_3dr2n2.py dataset.preparation.tar_path=/path/to/ShapeNetRendering.tgz

    # Use different dataset config
    python prepare_3dr2n2_hydra.py dataset=shapenet_3dr2n2_diff

    # Disable auto-download
    python prepare_3dr2n2_hydra.py dataset.preparation.auto_download=false
    """
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Get dataset config
    dataset_cfg = cfg.datasets

    # Prepare dataset if needed
    if not Path(dataset_cfg.root).exists():
        print(f"Dataset not found at {dataset_cfg.root}")
        print("Preparing dataset...")

        root = prepare_3dr2n2_dataset(dataset_cfg)
        print(f"Dataset prepared at: {root}")
    else:
        print(f"Using existing dataset at: {dataset_cfg.root}")

    # Test loading with Hydra config
    print("\nTesting dataloader creation...")
    train_loader, val_loader, test_loader = create_3dr2n2_dataloaders(dataset_cfg)

    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Viewpoint shape: {batch['viewpoint'].shape}")
    print(f"Categories: {batch['category']}")

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")

    # Test single dataset creation
    print("\nTesting the creation of a single dataset...")
    single_dataset_cfg = OmegaConf.create({**dataset_cfg, "split": "train"})
    train_dataset = get_dataset_from_config(single_dataset_cfg)
    print(f"Train dataset size: {len(train_dataset)} samples")

    # Test a single sample
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Category: {sample['category']}")
    print(f"Model ID: {sample['model_id']}")


if __name__ == "__main__":
    main()
