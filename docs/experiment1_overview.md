# Probing Experiment Overview

Quick rundown of our SSL vision model probing setup. 

## What We're Doing

Testing how well different SSL models (DINOv2, I-JEPA, MoCo v3) + supervised ViT capture 3D viewpoint info. The basic idea is that we extract features from different layers, train linear/MLP probes to predict azimuth/elevation angles.

## Models We Test

- **DINOv2**: Self-distillation approach, usually good at spatial stuff
- **I-JEPA**: Meta's image-based joint embedding, focuses on semantic understanding  
- **MoCo v3**: Momentum contrast, contrastive learning
- **Supervised ViT**: Regular supervised training baseline

## Dataset

ShapeNet 3D-R2N2 - renders of 3D objects from different viewpoints. We use:
- 13 object categories (chairs, cars, planes, etc.)
- 24 viewpoints per object 
- Azimuth: 0-330° (15° steps), Elevation: 30° fixed mostly

## Code Structure

```
src/
├── models/feature_extractor.py     # Loads pretrained models, extracts features
├── datasets/shapenet_3dr2n2.py     # Dataset loading, preprocessing
├── probing/
│   ├── probes.py                   # Linear + MLP probe definitions
│   ├── data_preprocessing.py       # Feature extraction pipeline
│   └── metrics.py                  # Regression metrics (MAE, RMSE, R²)
└── analysis/layer_analysis.py      # Layer-wise analysis, plotting
```

## Key Scripts

- `scripts/prepare_3dr2n2.py` - Downloads/preps dataset
- `scripts/run_probing_experiment.py` - Main experiment runner
- `scripts/dry_run_pipeline.py` - Quick test without full training

## How It Works

1. **Extract the Features**: Pull features from multiple layers (2,4,6,8,10,11) using `FeatureExtractorPipeline`
2. **Probe Training**: Train linear/MLP probes on extracted features to predict viewpoint
3. **Analyze Layer-Wise Performance**: Compare performance across layers using `LayerWiseAnalyzer`
4. **Results**: JSON dumps + plots showing which layers work best

## Config System

Again, we use Hydra for everything. Main configs:

```yaml
# Model config (configs/models/dinov2.yaml)
model_name: "dinov2_vitb14"
layers: [2, 4, 6, 8, 10, 11]  # Which layers to probe

# Experiment config  
probing:
  probe_types: ["linear", "mlp"]
  task_type: "viewpoint_regression"
  output_dim: 2  # azimuth + elevation
```

## Running Experiments

```bash
python scripts/run_probing_experiment.py \
  models=dinov2 \
  experiment.name=my_test
```

## Analysis Output

Results saved to `results/{experiment_name}/`:
- `results.json` - Raw metrics, training history
- `analysis_report.md` - Summary stats
- Various plots showing layer performance

## Metrics We Track

- **MAE**: Mean absolute error on angle prediction
- **RMSE**: Root mean squared error  
- **R²**: Coefficient of determination
- **Azimuth/Elevation specific**: Separate metrics for each angle

## Layer Analysis

The `LayerWiseAnalyzer` automatically:
- Finds optimal layers per probe type
- Plots performance vs depth
- Generates heatmaps across layers/probes
- Computes training efficiency metrics

## Known Issues

- Some models might need different layer ranges
- Feature caching can eat disk space quickly
- WandB logging sometimes flaky

## TODO

- Add error handling for model loading failures  
- Implement classification probes for shape categories
- Speed up feature extraction with better caching
