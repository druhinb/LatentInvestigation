#!/usr/bin/env python
"""Prepare 3D-R2N2 ShapeNet dataset using Hydra configuration"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from src.datasets.shapenet_voxel_meshes import (
    prepare_3dr2n2_reconstruction_dataset,
    create_3dr2n2_reconstruction_dataloaders,
    get_reconstruction_dataset_from_config,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Prepares, creates, and tests the ShapeNet-for-voxel-reconstruction using Hydra config

    Usage examples:
    # Use default config
    python prepare_3dr2n2_reconstruction.py

    # Use different dataset config
    python prepare_3dr2n2_reconstruction.py dataset=shapenet_voxel_meshes_diff
    """
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Get dataset config
    dataset_cfg = cfg.datasets

    # Prepare dataset
    if (
        not Path(dataset_cfg.render_root).exists()
        or not Path(dataset_cfg.voxel_root).exists()
    ):
        print(
            f"Dataset not found at {dataset_cfg.render_root}, {dataset_cfg.voxel_root}"
        )
        print("Preparing dataset...")

        prepare_3dr2n2_reconstruction_dataset(dataset_cfg)
        print(f"Dataset prepared!")
    else:
        print(
            f"Using existing dataset at: {dataset_cfg.render_root}, {dataset_cfg.voxel_root}"
        )

    print("\nTesting dataloader creation...")
    train_loader, val_loader, test_loader = create_3dr2n2_reconstruction_dataloaders(
        dataset_cfg
    )

    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Image shape: {batch['images'].shape}")

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")

    # Test single dataset creation
    print("\nTesting the creation of a single dataset...")
    single_dataset_cfg = OmegaConf.create({**dataset_cfg, "split": "train"})
    train_dataset = get_reconstruction_dataset_from_config(single_dataset_cfg)
    print(f"Train dataset size: {len(train_dataset)} samples")

    # Test a single sample
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Images shape: {sample['images'].shape}")
    print(f"Category: {sample['category']}")
    print(f"Model ID: {sample['model_id']}")


if __name__ == "__main__":
    main()
