"""Layer-wise analysis utilities for probing experiments"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class LayerAnalysisResult:
    """Results from layer-wise probe analysis"""

    layer: int
    probe_type: str
    mae: float
    rmse: float
    r2: float
    best_epoch: int
    total_epochs: int


class LayerWiseAnalyzer:
    """Analyzer for all the probing results"""

    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = Path(results_dir or "./analysis_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def extract_layer_results(
        self, experiment_results: Dict
    ) -> List[LayerAnalysisResult]:
        """Extract layer-wise results from experiment data"""
        results = []

        for layer_key, layer_data in experiment_results.items():
            if not layer_key.startswith("layer_"):
                continue

            layer_num = int(layer_key.split("_")[1])

            for probe_type, probe_data in layer_data.items():
                metrics = probe_data.get("test_metrics", {})
                results.append(
                    LayerAnalysisResult(
                        layer=layer_num,
                        probe_type=probe_type,
                        mae=metrics.get("mae", 0.0),
                        rmse=metrics.get("rmse", 0.0),
                        r2=metrics.get("r2", 0.0),
                        best_epoch=probe_data.get("best_epoch", 0),
                        total_epochs=probe_data.get("total_epochs", 0),
                    )
                )

        return results

    def find_optimal_layers(
        self, layer_results: List[LayerAnalysisResult]
    ) -> Dict[str, Dict]:
        """Find best performing layers for each probe type"""
        optimal = {}
        probe_types = set(r.probe_type for r in layer_results)

        for probe_type in probe_types:
            probe_results = [r for r in layer_results if r.probe_type == probe_type]
            optimal[probe_type] = {
                "best_mae": min(probe_results, key=lambda x: x.mae),
                "best_rmse": min(probe_results, key=lambda x: x.rmse),
                "best_r2": max(probe_results, key=lambda x: x.r2),
            }

        return optimal

    def compute_layer_statistics(
        self, layer_results: List[LayerAnalysisResult]
    ) -> Dict:
        """Compute statistics across layers"""
        if not layer_results:
            return {}

        # Group by layer
        layers_data = {}
        for result in layer_results:
            if result.layer not in layers_data:
                layers_data[result.layer] = []
            layers_data[result.layer].append(result)

        # Compute stats per layer
        layer_stats = {}
        for layer, results in layers_data.items():
            maes = [r.mae for r in results]
            rmses = [r.rmse for r in results]
            r2s = [r.r2 for r in results]

            layer_stats[layer] = {
                "num_probes": len(results),
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
                "rmse_mean": np.mean(rmses),
                "rmse_std": np.std(rmses),
                "r2_mean": np.mean(r2s),
                "r2_std": np.std(r2s),
                "probe_types": [r.probe_type for r in results],
            }

        return layer_stats

    def analyze_layer_trends(self, layer_results: List[LayerAnalysisResult]) -> Dict:
        """Analyze trends across early, middle, and late layers"""
        if not layer_results:
            return {}

        layers = sorted(set(r.layer for r in layer_results))
        n = len(layers)

        # Split into thirds
        early = layers[: n // 3] if n > 3 else layers[:1]
        middle = layers[n // 3 : 2 * n // 3] if n > 3 else layers[1:2] if n > 1 else []
        late = layers[2 * n // 3 :] if n > 3 else layers[-1:] if n > 2 else layers[-1:]

        def get_group_stats(group_layers, probe_type):
            group_results = [
                r
                for r in layer_results
                if r.layer in group_layers and r.probe_type == probe_type
            ]
            if not group_results:
                return {"mae": float("inf"), "rmse": float("inf"), "r2": 0.0}

            return {
                "mae": np.mean([r.mae for r in group_results]),
                "rmse": np.mean([r.rmse for r in group_results]),
                "r2": np.mean([r.r2 for r in group_results]),
            }

        probe_types = set(r.probe_type for r in layer_results)
        trends = {}

        for probe_type in probe_types:
            trends[probe_type] = {
                "early_layers": {
                    "layers": early,
                    "stats": get_group_stats(early, probe_type),
                },
                "middle_layers": {
                    "layers": middle,
                    "stats": get_group_stats(middle, probe_type),
                },
                "late_layers": {
                    "layers": late,
                    "stats": get_group_stats(late, probe_type),
                },
            }

        return trends

    def create_plots(
        self,
        layer_results: List[LayerAnalysisResult],
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """Create visualizations for layer analysis"""
        if not layer_results:
            return []

        output_dir = output_dir or self.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("default")
        sns.set_palette("Set2")

        plots = []
        plots.append(self._plot_performance_vs_depth(layer_results, output_dir))
        plots.append(self._plot_performance_heatmap(layer_results, output_dir))
        plots.append(self._plot_training_efficiency(layer_results, output_dir))
        plots.append(self._plot_layer_trends(layer_results, output_dir))

        return [p for p in plots if p is not None]

    def _plot_performance_vs_depth(
        self, layer_results: List[LayerAnalysisResult], output_dir: Path
    ) -> Optional[Path]:
        """Plot performance metrics vs layer depth"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            probe_types = sorted(set(r.probe_type for r in layer_results))
            colors = sns.color_palette("Set2", len(probe_types))

            metrics = [
                ("mae", "MAE"),
                ("rmse", "RMSE"),
                ("r2", "R²"),
                ("total_epochs", "Training Epochs"),
            ]

            for i, (metric, label) in enumerate(metrics):
                ax = axes[i]

                for j, probe_type in enumerate(probe_types):
                    probe_results = [
                        r for r in layer_results if r.probe_type == probe_type
                    ]
                    probe_results.sort(key=lambda x: x.layer)

                    layers = [r.layer for r in probe_results]
                    values = [getattr(r, metric) for r in probe_results]

                    ax.plot(
                        layers,
                        values,
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        label=probe_type,
                        color=colors[j],
                    )

                ax.set_xlabel("Layer")
                ax.set_ylabel(label)
                ax.set_title(f"{label} vs Layer")
                ax.legend()
                ax.grid(True, alpha=0.3)

                if metric in ["mae", "rmse"]:
                    ax.set_ylim(bottom=0)
                elif metric == "r2":
                    ax.set_ylim(0, 1)

            plt.tight_layout()
            plot_file = output_dir / "performance_vs_depth.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            return plot_file

        except Exception as e:
            logger.error(f"Error creating performance plot: {e}")
            return None

    def _plot_performance_heatmap(
        self, layer_results: List[LayerAnalysisResult], output_dir: Path
    ) -> Optional[Path]:
        """Create heatmap of performance across layers and probe types"""
        try:
            probe_types = sorted(set(r.probe_type for r in layer_results))
            layers = sorted(set(r.layer for r in layer_results))

            # Create MAE matrix
            data_matrix = np.full((len(probe_types), len(layers)), np.nan)

            for i, probe_type in enumerate(probe_types):
                for j, layer in enumerate(layers):
                    result = next(
                        (
                            r
                            for r in layer_results
                            if r.probe_type == probe_type and r.layer == layer
                        ),
                        None,
                    )
                    if result:
                        data_matrix[i, j] = result.mae

            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(data_matrix, cmap="viridis_r", aspect="auto")

            # Labels
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([f"L{l}" for l in layers])
            ax.set_yticks(range(len(probe_types)))
            ax.set_yticklabels(probe_types)

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("MAE", rotation=270, labelpad=20)

            # Add values
            for i in range(len(probe_types)):
                for j in range(len(layers)):
                    if not np.isnan(data_matrix[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{data_matrix[i, j]:.3f}",
                            ha="center",
                            va="center",
                            color="white",
                            fontweight="bold",
                        )

            ax.set_title("Layer-wise Performance Heatmap (MAE)")
            plt.tight_layout()

            plot_file = output_dir / "performance_heatmap.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            return plot_file

        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return None

    def _plot_training_efficiency(
        self, layer_results: List[LayerAnalysisResult], output_dir: Path
    ) -> Optional[Path]:
        """Plot training efficiency metrics"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            probe_types = sorted(set(r.probe_type for r in layer_results))
            colors = sns.color_palette("Set2", len(probe_types))

            # Epochs vs MAE scatter
            for i, probe_type in enumerate(probe_types):
                probe_results = [r for r in layer_results if r.probe_type == probe_type]
                epochs = [r.total_epochs for r in probe_results]
                maes = [r.mae for r in probe_results]
                ax1.scatter(
                    epochs, maes, label=probe_type, alpha=0.7, color=colors[i], s=60
                )

            ax1.set_xlabel("Training Epochs")
            ax1.set_ylabel("MAE")
            ax1.set_title("Epochs vs Performance")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Best epochs by layer
            for i, probe_type in enumerate(probe_types):
                probe_results = sorted(
                    [r for r in layer_results if r.probe_type == probe_type],
                    key=lambda x: x.layer,
                )
                layers = [r.layer for r in probe_results]
                best_epochs = [r.best_epoch for r in probe_results]
                ax2.plot(
                    layers,
                    best_epochs,
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    label=probe_type,
                    color=colors[i],
                )

            ax2.set_xlabel("Layer")
            ax2.set_ylabel("Best Epoch")
            ax2.set_title("Convergence by Layer")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = output_dir / "training_efficiency.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            return plot_file

        except Exception as e:
            logger.error(f"Error creating efficiency plot: {e}")
            return None

    def _plot_layer_trends(
        self, layer_results: List[LayerAnalysisResult], output_dir: Path
    ) -> Optional[Path]:
        """Plot trends across early/middle/late layers"""
        try:
            trend_analysis = self.analyze_layer_trends(layer_results)
            if not trend_analysis:
                return None

            probe_types = list(trend_analysis.keys())
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            metrics = ["mae", "rmse", "r2"]
            labels = ["MAE", "RMSE", "R² Score"]

            for i, (metric, label) in enumerate(zip(metrics, labels)):
                ax = axes[i]
                x_pos = np.arange(len(probe_types))
                width = 0.25

                early_vals = [
                    trend_analysis[pt]["early_layers"]["stats"][metric]
                    for pt in probe_types
                ]
                middle_vals = [
                    trend_analysis[pt]["middle_layers"]["stats"][metric]
                    for pt in probe_types
                ]
                late_vals = [
                    trend_analysis[pt]["late_layers"]["stats"][metric]
                    for pt in probe_types
                ]

                ax.bar(x_pos - width, early_vals, width, label="Early", alpha=0.8)
                ax.bar(x_pos, middle_vals, width, label="Middle", alpha=0.8)
                ax.bar(x_pos + width, late_vals, width, label="Late", alpha=0.8)

                ax.set_xlabel("Probe Type")
                ax.set_ylabel(label)
                ax.set_title(f"{label} by Layer Group")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(probe_types)
                ax.legend()
                ax.grid(True, alpha=0.3, axis="y")

                if metric in ["mae", "rmse"]:
                    ax.set_ylim(bottom=0)
                elif metric == "r2":
                    ax.set_ylim(0, 1)

            plt.tight_layout()
            plot_file = output_dir / "layer_trends.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            return plot_file

        except Exception as e:
            logger.error(f"Error creating trends plot: {e}")
            return None

    def save_analysis_report(
        self,
        layer_results: List[LayerAnalysisResult],
        output_file: Optional[Path] = None,
    ) -> Path:
        """Save analysis report to JSON"""
        output_file = output_file or (self.results_dir / "layer_analysis_report.json")

        optimal_layers = self.find_optimal_layers(layer_results)
        layer_stats = self.compute_layer_statistics(layer_results)
        trend_analysis = self.analyze_layer_trends(layer_results)

        report = {
            "summary": {
                "total_layers": len(set(r.layer for r in layer_results)),
                "total_probe_types": len(set(r.probe_type for r in layer_results)),
                "total_experiments": len(layer_results),
            },
            "optimal_layers": {
                probe_type: {
                    metric: {
                        "layer": result.layer,
                        "mae": result.mae,
                        "rmse": result.rmse,
                        "r2": result.r2,
                    }
                    for metric, result in metrics.items()
                }
                for probe_type, metrics in optimal_layers.items()
            },
            "layer_statistics": layer_stats,
            "trend_analysis": trend_analysis,
            "raw_results": [
                {
                    "layer": r.layer,
                    "probe_type": r.probe_type,
                    "mae": r.mae,
                    "rmse": r.rmse,
                    "r2": r.r2,
                    "best_epoch": r.best_epoch,
                    "total_epochs": r.total_epochs,
                }
                for r in layer_results
            ],
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Analysis report saved to {output_file}")
        return output_file


def analyze_experiment_results(
    results_file: Path, output_dir: Optional[Path] = None
) -> Dict:
    """Analyze experiment results from file"""
    with open(results_file, "r") as f:
        experiment_data = json.load(f)

    results = experiment_data.get("results", {})
    analyzer = LayerWiseAnalyzer(output_dir)
    layer_results = analyzer.extract_layer_results(results)

    if not layer_results:
        logger.warning("No layer results found in experiment data")
        return {}

    analysis = {
        "optimal_layers": analyzer.find_optimal_layers(layer_results),
        "layer_statistics": analyzer.compute_layer_statistics(layer_results),
        "trend_analysis": analyzer.analyze_layer_trends(layer_results),
    }

    # Create plots and save report
    plot_files = analyzer.create_plots(layer_results, output_dir)
    analysis["plot_files"] = [str(p) for p in plot_files]

    report_file = analyzer.save_analysis_report(
        layer_results, output_dir / "layer_analysis_report.json"
    )
    analysis["report_file"] = str(report_file)

    return analysis
