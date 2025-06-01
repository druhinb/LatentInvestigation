# ShapeNet 3D-R2N2 Dataset Generation

This guide provides step-by-step instructions for generating and preparing the ShapeNet 3D-R2N2 dataset for use with our probing experiments.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration Options](#configuration-options)
- [Manual Setup](#manual-setup)
- [Troubleshooting](#troubleshooting)

## Overview

The ShapeNet 3D-R2N2 dataset provides multi-view renderings of 3D objects from the ShapeNet dataset. Our implementation:
- ✅ **Auto-downloads** the dataset if not present
- ✅ **Creates train/val/test splits** automatically when missing
- ✅ **Supports existing tar files** for offline preparation
- ✅ **Hydra configuration** for flexible setup
- ✅ **Configurable categories** (chairs, cars, airplanes, etc.)

## Prerequisites

Ensure you have completed the main installation:
```bash
conda activate LatentInvestigation
# OR
source venv/bin/activate
```

## Quick Start

### Option 1: Automatic Download (Recommended)
```bash
# Prepare dataset with default settings
python scripts/prepare_3dr2n2.py

# This will:
# 1. Download ShapeNetRendering.tgz (~25GB)
# 2. Extract to ./data/ShapeNetRendering/
# 3. Create train/val/test split files (.lst)
# 4. Process all categories
```

### Option 2: Use Existing Tar File
If you already have the ShapeNet tar file:
```bash
python scripts/prepare_3dr2n2.py \
  datasets.preparation.tar_path=/path/to/ShapeNetRendering.tgz
```

### Option 3: Specific Categories Only
```bash
python scripts/prepare_3dr2n2.py \
  datasets.categories=[chair,car,airplane]
```

## Configuration Options

The dataset behavior is controlled via `configs/datasets/shapenet_3dr2n2.yaml`:

```yaml
# Basic settings
_target_: src.datasets.shapenet_3dr2n2.ShapeNet3DR2N2
root: ./data/ShapeNetRendering
categories: null  # null=all categories, or [chair,car,airplane]

# Data preparation
preparation:
  tar_path: null        # Path to existing .tgz file
  auto_download: true   # Download if not found

# Split configuration (automatic when .lst files missing)
splits:
  train_ratio: 0.7     # 70% training data
  val_ratio: 0.15      # 15% validation data  
  seed: 42             # Reproducible splits

# DataLoader settings  
dataloader:
  batch_size: 32
  num_workers: 4
  shuffle: true
```

### Common Configuration Overrides

```bash
# Custom split ratios
python scripts/prepare_3dr2n2.py \
  datasets.splits.train_ratio=0.8 \
  datasets.splits.val_ratio=0.1

# Specific data location
python scripts/prepare_3dr2n2.py \
  datasets.root=/custom/path/to/data

# Different batch size
python scripts/prepare_3dr2n2.py \
  datasets.dataloader.batch_size=64
```

## Manual Setup

### 1. Download Dataset Manually
```bash
# Create data directory
mkdir -p data/

# Download (alternative to auto-download)
wget -P data/ https://cvgl.stanford.edu/data2/ShapeNetRendering.tgz

# Extract
cd data/
tar -xzf ShapeNetRendering.tgz
cd ..
```

### 2. Prepare Dataset
```bash
python scripts/prepare_3dr2n2.py \
  datasets.preparation.auto_download=false
```

### 3. Verify Setup
```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Load config
cfg = OmegaConf.load("configs/datasets/shapenet_3dr2n2.yaml")

# Test dataset loading
dataset = instantiate(cfg, split="train")
print(f"Training samples: {len(dataset)}")

# Test dataloader creation
from src.datasets.shapenet_3dr2n2 import create_3dr2n2_dataloaders
train_loader, val_loader, test_loader = create_3dr2n2_dataloaders(cfg)
print(f"Train batches: {len(train_loader)}")
```

## Using the Dataset in Code

### Basic Usage
```python
from src.datasets.shapenet_3dr2n2 import create_3dr2n2_dataloaders
from omegaconf import OmegaConf

# Load configuration
cfg = OmegaConf.load("configs/datasets/shapenet_3dr2n2.yaml")

# Create dataloaders
train_loader, val_loader, test_loader = create_3dr2n2_dataloaders(cfg)

# Use in training loop
for batch_idx, (images, labels) in enumerate(train_loader):
    # images: [batch_size, num_views, 3, H, W]
    # labels: dict with shape info and metadata
    pass
```

### Direct Dataset Access
```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/datasets/shapenet_3dr2n2.yaml")

# Create individual datasets
train_dataset = instantiate(cfg, split="train")
val_dataset = instantiate(cfg, split="val") 
test_dataset = instantiate(cfg, split="test")

# Access single sample
sample = train_dataset[0]
images = sample['images']  # Shape: [num_views, 3, H, W]
category = sample['category']
model_id = sample['model_id']
```

## Dataset Structure

After preparation, your data directory will look like:
```
data/ShapeNetRendering/
├── 02691156/           # Category: airplane
│   ├── 1a04e3eab45ca15dd86060f189eb133/
│   │   └── rendering/
│   │       ├── 00.png  # View 0
│   │       ├── 01.png  # View 1
│   │       └── ...     # Views 2-23
│   └── ...
├── 02958343/           # Category: car
├── 03001627/           # Category: chair
├── ...
├── train.lst           # Training split (auto-created)
├── val.lst             # Validation split (auto-created)
└── test.lst            # Test split (auto-created)
```

## Troubleshooting

### Common Issues

**1. Download fails or is slow**
```bash
# Check internet connection and retry
python scripts/prepare_3dr2n2.py datasets.preparation.auto_download=true

# Or download manually and use local file
python scripts/prepare_3dr2n2.py datasets.preparation.tar_path=/path/to/file.tgz
```

**2. Out of disk space**
- The dataset requires ~25GB for download + ~25GB extracted = ~50GB total
- Consider using a different data directory:
```bash
python scripts/prepare_3dr2n2.py datasets.root=/path/to/large/disk/data
```

**3. Split files not found**
The dataset automatically creates splits when `.lst` files are missing. If you need to recreate them:
```bash
# Delete existing splits and regenerate
rm data/ShapeNetRendering/*.lst
python scripts/prepare_3dr2n2.py
```

**4. CUDA/Memory issues during loading**
```bash
# Reduce batch size and workers
python scripts/prepare_3dr2n2.py \
  datasets.dataloader.batch_size=16 \
  datasets.dataloader.num_workers=2
```

### Verification Commands

```bash
# Check dataset size
du -sh data/ShapeNetRendering/

# Count samples in splits
wc -l data/ShapeNetRendering/*.lst

# Test loading speed
python -c "
from src.datasets.shapenet_3dr2n2 import create_3dr2n2_dataloaders
from omegaconf import OmegaConf
import time

cfg = OmegaConf.load('configs/datasets/shapenet_3dr2n2.yaml')
train_loader, _, _ = create_3dr2n2_dataloaders(cfg)

start = time.time()
batch = next(iter(train_loader))
print(f'First batch loaded in {time.time()-start:.2f}s')
print(f'Batch images shape: {batch[\"images\"].shape}')
"
```

## Next Steps

After dataset preparation:
1. **Run probing experiments**: See main README for experiment commands
2. **Explore with notebooks**: Check `notebooks/` for analysis examples  
3. **Customize transforms**: Modify the config to add data augmentations
4. **Multi-GPU training**: Configure dataloaders for distributed training

---

For more advanced usage and configuration details, see [`docs/shapenet_hydra_usage.md`](docs/shapenet_hydra_usage.md).
