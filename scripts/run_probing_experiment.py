#!/usr/bin/env python
"""Main script for probing experiments on SSL models"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
import wandb
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Import our modules
from src.models.feature_extractor import FeatureExtractor, load_feature_extractor
from src.datasets.shapenet_3dr2n2 import create_3dr2n2_dataloaders
from src.probing.probes import create_probe, ProbeTrainer
from src.probing.data_preprocessing import (
    FeatureExtractorPipeline,
    create_probing_dataloaders,
    ProbingDataset,
)
from src.probing.metrics import (
    compute_regression_metrics,
    compute_viewpoint_specific_metrics,
    MetricsTracker,
)
from src.analysis.layer_analysis import LayerWiseAnalyzer

logger = logging.getLogger(__name__)


class ProbingExperiment:
    """Orchestrates probing experiments"""

    def __init__(self, config: DictConfig):
        self.config = config
        # Determine device: prioritize models.device, then top-level device, then auto-detect
        device_to_use = config.models.get("device", config.get("device"))
        if device_to_use:
            self.device = device_to_use
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        logger.info(f"Using device: {self.device}")

        # Initialize wandb
        if config.get("wandb", {}).get("enabled", False):
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.get("entity"),
                name=config.experiment.name,
                config=OmegaConf.to_container(config, resolve=True),
            )

        # Setup paths
        self.results_dir = Path(config.get("results_dir", "./results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(config.get("cache_dir", "./cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analyzer
        self.analyzer = LayerWiseAnalyzer(self.results_dir / config.experiment.name)

    def run_experiment(self) -> Dict:
        """Run the complete probing experiment"""
        logger.info("Starting probing experiment...")

        # Load dataset and feature extractor
        feature_extractor = self._load_feature_extractor()
        train_loader, val_loader, test_loader = self._load_dataset()

        # Get experiment configuration
        extraction_config = self.config.models.get("feature_extraction", {})
        layers = extraction_config.get("layers", [11])
        feature_type = extraction_config.get("feature_type", "cls_token")
        task_type = self.config.probing.get("task_type", "viewpoint_regression")

        # Run layer-wise probing
        results = {}
        for layer in tqdm(layers):
            logger.info(f"Processing layer {layer}...")

            # Extract features for this layer
            train_dataset, val_dataset, test_dataset = self._extract_features_for_layer(
                feature_extractor,
                train_loader,
                val_loader,
                test_loader,
                layer,
                feature_type,
                task_type,
            )

            # Create probing dataloaders
            probe_train_loader, probe_val_loader, probe_test_loader = (
                create_probing_dataloaders(
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    batch_size=self.config.probing.get("training", {}).get(
                        "batch_size", 64
                    ),
                    num_workers=self.config.get("num_workers", 4),
                )
            )

            # Run probing experiments for each probe type
            layer_results = {}
            for probe_type in self.config.probing.probe_types:
                logger.info(f"Running {probe_type} probe on layer {layer}...")
                probe_results = self._run_probe_experiment(
                    probe_type,
                    probe_train_loader,
                    probe_val_loader,
                    probe_test_loader,
                    train_dataset.features.shape[1],
                    layer,
                )
                layer_results[probe_type] = probe_results

            results[f"layer_{layer}"] = layer_results

        # Save results
        logger.info("Analyzing results...")
        self._save_results(results)

        # Analyze results
        logger.info("Creating analysis and visualizations...")
        self.analyzer.analyze_experiment_results(results)

        logger.info("Experiment completed!")
        return results

    def _load_dataset(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load the dataset"""
        subset_percentage = self.config.datasets.get("subset_percentage", None)
        return create_3dr2n2_dataloaders(
            self.config.datasets, subset_percentage=subset_percentage
        )

    def _load_feature_extractor(self) -> FeatureExtractor:
        """Load and setup feature extractor"""
        model_config = self.config.models
        model_config.device = self.device
        model_config.cache_dir = str(self.cache_dir / "models")

        feature_extractor = load_feature_extractor(OmegaConf.to_container(model_config))
        logger.info(f"Loaded {model_config.model_name} feature extractor")
        return feature_extractor

    def _extract_features_for_layer(
        self,
        feature_extractor: FeatureExtractor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        layer: int,
        feature_type: str,
        task_type: str,
    ) -> Tuple[ProbingDataset, ProbingDataset, ProbingDataset]:
        """Extract features for a specific layer"""
        pipeline = FeatureExtractorPipeline(
            feature_extractor=feature_extractor,
            device=self.device,
            batch_size=self.config.get("extraction_batch_size", 32),
            cache_dir=str(self.cache_dir / "features"),
        )

        experiment_name = f"{self.config.models.model_name}_{self.config.experiment.name}_layer_{layer}"

        return pipeline.create_probing_datasets(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            layers=[layer],
            feature_type=feature_type,
            task_type=task_type,
            experiment_name=experiment_name,
        )

    def _run_probe_experiment(
        self,
        probe_type: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        feature_dim: int,
        layer: int,
    ) -> Dict:
        """Run a single probe experiment"""

        logger.info(
            f"Running {probe_type} probe on layer {layer} (feature_dim: {feature_dim})"
        )

        # Get probe configuration
        probe_config = self.config.probing.get(probe_type, {})
        # Make a mutable copy for modification
        probe_config = OmegaConf.to_container(probe_config, resolve=True)

        # Create probe
        probe_config["input_dim"] = feature_dim
        probe_config["output_dim"] = self.config.probing.get("output_dim", 2)

        main_task_type = self.config.probing.get("task_type", "regression")
        if main_task_type == "viewpoint_regression":
            probe_config["task_type"] = "regression"
        elif main_task_type == "view_classification":
            probe_config["task_type"] = "classification"
        else:
            probe_config["task_type"] = main_task_type

        probe = create_probe(probe_config)

        # Setup trainer
        trainer = ProbeTrainer(probe, device=self.device)

        # Setup optimizer and scheduler
        training_config = probe_config.get("training", {})
        optimizer = self._create_optimizer(probe, training_config.get("optimizer", {}))
        scheduler = self._create_scheduler(
            optimizer, training_config.get("scheduler", {})
        )

        # Training parameters
        epochs = training_config.get("epochs", 30)
        early_stopping_patience = training_config.get("early_stopping_patience", 15)

        # Metrics tracker
        metrics_tracker = MetricsTracker()

        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss = trainer.train_epoch(train_loader, optimizer, scheduler)

            # Validate
            val_metrics = trainer.evaluate(val_loader)
            val_loss = val_metrics["loss"]

            # Update metrics
            metrics_tracker.update("train", {"loss": train_loss}, epoch)
            metrics_tracker.update("val", val_metrics, epoch)

            # Log to wandb
            if wandb.run:
                wandb.log(
                    {
                        f"{probe_type}/train_loss": train_loss,
                        f"{probe_type}/val_loss": val_loss,
                        f"{probe_type}/val_mae": val_metrics.get("mae", 0),
                        f"{probe_type}/val_r2": val_metrics.get("r2", 0),
                        "epoch": epoch,
                    }
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = probe.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

        # Load best model and evaluate on test set
        probe.load_state_dict(best_model_state)
        test_metrics = trainer.evaluate(test_loader)

        # Compute detailed viewpoint metrics on test set
        detailed_metrics = self._compute_detailed_metrics(probe, test_loader)

        results = {
            "train_history": metrics_tracker.get_history("train"),
            "val_history": metrics_tracker.get_history("val"),
            "test_metrics": test_metrics,
            "detailed_metrics": detailed_metrics,
            "best_epoch": metrics_tracker.best_epoch,
            "total_epochs": epoch + 1,
        }

        return results

    def _create_optimizer(
        self, model: nn.Module, optimizer_config: Dict
    ) -> torch.optim.Optimizer:
        """Create optimizer from config using Hydra instantiate"""
        from hydra.utils import instantiate

        # Create a copy of config and add model parameters
        optimizer_config = optimizer_config.copy()
        optimizer_config["params"] = model.parameters()

        return instantiate(optimizer_config)

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer, scheduler_config: Dict
    ):
        """Create learning rate scheduler from config using Hydra instantiate"""
        if not scheduler_config:
            return None

        from hydra.utils import instantiate

        # Create a copy of config and add optimizer
        scheduler_config = scheduler_config.copy()
        scheduler_config["optimizer"] = optimizer

        return instantiate(scheduler_config)

    def _compute_detailed_metrics(
        self, probe: nn.Module, test_loader: DataLoader
    ) -> Dict:
        """Compute alles metrics"""
        probe.eval()

        all_predictions = []
        all_targets = []
        all_categories = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                targets = batch["targets"]

                outputs = probe(features)

                all_predictions.append(outputs.cpu())
                all_targets.append(targets)

                # Get categories if available
                if "categories" in batch:
                    all_categories.extend(batch["categories"])

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # Basic regression metrics
        metrics = compute_regression_metrics(predictions, targets, return_per_dim=True)

        # Viewpoint-specific metrics
        if predictions.shape[1] == 2:
            viewpoint_metrics = compute_viewpoint_specific_metrics(
                azimuth_pred=predictions[:, 0],
                elevation_pred=predictions[:, 1],
                azimuth_target=targets[:, 0],
                elevation_target=targets[:, 1],
            )
            metrics.update(viewpoint_metrics)

        return metrics

    def _save_results(self, results: Dict):
        """Save results to disk"""
        import json

        # Create experiment directory
        exp_dir = self.results_dir / self.config.experiment.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        results_file = exp_dir / "results.json"

        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)

        combined_results = {
            "config": OmegaConf.to_container(self.config, resolve=True),
            "results": serializable_results,
        }

        with open(results_file, "w") as f:
            json.dump(combined_results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, "tolist") else float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj


@hydra.main(
    version_base=None, config_path="../configs", config_name="experiment_config"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for probing experiments"""

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.get("log_level", "INFO")),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run experiment
    experiment = ProbingExperiment(cfg)
    results = experiment.run_experiment()

    # Print summary
    logger.info("=== EXPERIMENT SUMMARY ===")
    for layer_key, layer_results in results.items():
        logger.info(f"{layer_key.upper()}:")
        for probe_type, probe_results in layer_results.items():
            test_metrics = probe_results["test_metrics"]
            logger.info(f"  {probe_type.upper()}:")
            logger.info(f"    MAE: {test_metrics.get('mae', 0):.4f}")
            logger.info(f"    RMSE: {test_metrics.get('rmse', 0):.4f}")
            logger.info(f"    R²: {test_metrics.get('r2', 0):.4f}")


if __name__ == "__main__":
    main()
