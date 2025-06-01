# ShapeNet 3D-R2N2 Dataset

## Quick Start

```bash
# Prepare dataset with default settings
python scripts/prepare_3dr2n2.py

# Use existing tar file
python scripts/prepare_3dr2n2.py datasets.preparation.tar_path=/path/to/ShapeNetRendering.tgz

# Custom split ratios (automatic when split files are missing)
python scripts/prepare_3dr2n2.py datasets.splits.train_ratio=0.8 datasets.splits.val_ratio=0.1
```

## Configuration

Basic config (`configs/datasets/shapenet_3dr2n2.yaml`):
```yaml
_target_: src.datasets.shapenet_3dr2n2.ShapeNet3DR2N2
root: ./data/ShapeNetRendering
categories: null  # null for all, or [airplane, car, chair]

preparation:
  tar_path: null    # Path to .tgz file, null to auto-download
  auto_download: true

splits:
  train_ratio: 0.7  # Automatic split creation when .lst files missing
  val_ratio: 0.15
  seed: 42

dataloader:
  batch_size: 32
  num_workers: 4
```

## Python Usage

```python
from hydra.utils import instantiate
from src.datasets.shapenet_3dr2n2 import create_3dr2n2_dataloaders

# Load config and create dataloaders
cfg = OmegaConf.load("configs/datasets/shapenet_3dr2n2.yaml")
train_loader, val_loader, test_loader = create_3dr2n2_dataloaders(cfg)

# Direct instantiation
dataset = instantiate(cfg, split="train")
```

