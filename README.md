# Empirical Probing of Intrinsic Properties in Self-Supervised Vision Models

This project provides aims to provide a comprehensive framework for probing and analyzing intrinsic visual properties learned by self-supervised vision models (DINO, JEPA, MoCo v3) compared to supervised baselines.

##  Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)

## Overview

We investigate how well different self-supervised learning (SSL) methods capture the fundamentally complex visual property of 3D Object Pose, using it as a segue to understanding the inner representations and features of such SSL models.
##  Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ compatible GPU
- (Optional) Docker with NVIDIA Container Toolkit

### Option 1: Conda Environment (Recommended)
```bash
# Clone the repository
git clone https://github.com/druhinb/LatentInvestigation.git
cd LatentInvestigation

# Create conda environment
conda env create -f environment.yml
conda activate LatentInvestigation 

# Install project in dev mode
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Option 2: Docker
```bash
# Build Docker image
docker build -t LatentInvestigation .

# Run with Docker Compose
docker-compose up -d LatentInvestigation

# Access container
docker-compose exec LatentInvestigation bash
```

### Option 3: pip (Basic)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

##  Quick Start

### 1. Download Pre-trained Models

PLACEHOLDER

### 2. Prepare Datasets

We use the ShapeNet 3D-R2N2 dataset for our probing experiments:

```bash
# Quick setup - downloads and prepares dataset automatically
python scripts/prepare_3dr2n2.py

# Or with custom settings
python scripts/prepare_3dr2n2.py \
  datasets.categories=[chair,car,airplane] \
  datasets.splits.train_ratio=0.8
```



### 3. Run Probing Experiments

PLACEHOLDER

### 4. Analyze Results

PLACEHOLDER

##  Project Structure

```
LatentInvestigation/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── model/                 # Model-specific configs
│   └── dataset/               # Dataset-specific configs
├── src/
│   ├── models/                # Model loading and feature extraction
│   ├── datasets/              # Dataset implementations
│   ├── probing/               # Linear probing framework
│   ├── analysis/              # Visualization and statistics
│   └── utils/                 # Helper functions
├── scripts/                   # Executable scripts
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Unit tests
├── data/                      # Dataset storage (not tracked)
├── checkpoints/               # Model weights (not tracked)
├── results/                   # Experiment outputs
└── outputs/                   # Hydra run outputs
```

##  Usage

### Configuration

The project uses Hydra for all of our configuration management. Key parameters:

```yaml
# configs/config.yaml
model:
  name: dino_vitb16
  checkpoint: facebook/dino-vitb16
  layers_to_probe: [11, 9, 7, 5]

dataset:
  name: shapenet
  categories: [chair, car, airplane]
  num_views: 24
  
probing:
  properties:
    viewpoint:
      type: regression
      output_dim: 2 
    shape:
      type: classification
      num_classes: 55
  
  training:
    batch_size: 64
    learning_rate: 0.001
    epochs: 100
    early_stopping_patience: 10
```

### Custom Experiments

PLACEHOLDER

##  Experiments

PLACEHOLDER

##  Results

Our experiments reveal:
- **DINO/DINOv2**: Strong viewpoint and shape encoding, especially in middle layers
- **I-JEPA**: Superior abstract property understanding
- **MoCo v3**: Good instance-level features, weaker on intrinsic properties
- **Supervised**: Strong semantic features but less geometric understanding

Detailed results and visualizations available in `notebooks/results_analysis.ipynb`

##  Testing

PLACEHOLDER 

##  Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes with a [good commit message](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13) (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- We use Black for formatting: `black src/`
- Type hints are encouraged
- Document new functions and classes

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{bhowal2025probing,
  title={Empirical Probing of Intrinsic Properties in Self-Supervised Vision Models},
  author={Bhowal, Druhin and Sengupta, Rajat and Shi, Noah},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

##  Acknowledgments

PLACEHOLDER

##  Contact

- Druhin Bhowal - dbhowal@cs.washington.edu
- Rajat Sengupta - rsen0811@cs.washington.edu  
- Noah Shi - noahshi@cs.washington.edu

Project Link: [https://github.com/druhinb/LatentInvestigation](https://github.com/druhinb/LatentInvestigation)

---
<p align="center">Made with ❤️ at the University of Washington</p>