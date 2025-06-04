"""Concise preprocessing pipeline for 3D reconstruction from multi-view data."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)


class ReconstructionInputDataset(Dataset):
    """Holds processed inputs and targets for 3D reconstruction."""

    def __init__(
        self,
        view_data: torch.Tensor,
        voxels: torch.Tensor,
        ids: List[str],
        cats: List[str],
    ):
        assert (
            len(view_data) == len(voxels) == len(ids) == len(cats)
        ), "All inputs to ReconstructionInputDataset must have the same length."
        self.view_data, self.voxels, self.ids, self.cats = view_data, voxels, ids, cats

    def __len__(self) -> int:
        return len(self.view_data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        return {
            "processed_views": self.view_data[idx],
            "target_voxels": self.voxels[idx],
            "model_id": self.ids[idx],
            "category": self.cats[idx],
        }


class CameraParameterProcessor:
    """Processes raw camera parameters (azimuth, elevation, in-plane rotation, distance, FoV)."""

    def __init__(self):
        pass

    def process(self, raw_cam_params_batch: torch.Tensor) -> torch.Tensor:
        """
        Normalizes and transforms raw camera parameters.
        Input: raw_cam_params_batch [B, NUM_VIEWS, 5]
               (az, el, ipr, dist, fov in degrees/original units)
        Output: processed_params [B, NUM_VIEWS, D_cam_feat] (D_cam_feat is 8 if sin_cos, else 5)
        """
        B, NV, _ = raw_cam_params_batch.shape
        num_cam_features = 8
        processed_views = torch.zeros(
            B,
            NV,
            num_cam_features,
            dtype=torch.float32,
            device=raw_cam_params_batch.device,
        )

        for b_idx in range(B):
            for v_idx in range(NV):
                az, el, ipr, dist, fov = raw_cam_params_batch[b_idx, v_idx].tolist()

                az_rad, el_rad, ipr_rad = (
                    np.deg2rad(az),
                    np.deg2rad(el),
                    np.deg2rad(ipr),
                )

                norm_dist, norm_fov = (dist - 2.5) / 0.5, (fov - 45.0) / 15.0

                processed_views[b_idx, v_idx] = torch.tensor(
                    [
                        np.sin(az_rad),
                        np.cos(az_rad),
                        np.sin(el_rad),
                        np.cos(el_rad),
                        np.sin(ipr_rad),
                        np.cos(ipr_rad),
                        norm_dist,
                        norm_fov,
                    ],
                    device=raw_cam_params_batch.device,
                )
        return processed_views


class ReconstructionPipeline:
    """Prepares data for 3D reconstruction models by processing views and camera parameters."""

    def __init__(
        self,
        image_feature_extractor,
        camera_processor: Optional[CameraParameterProcessor] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        self.image_feature_extractor = image_feature_extractor
        self.camera_processor = camera_processor or CameraParameterProcessor()
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def prepare_reconstruction_data(
        self,
        source_dataloader: DataLoader,
        image_feat_layers: Optional[List[int]] = None,
        image_feat_type: str = "cls_token",
        cache_key: Optional[str] = None,
        force_recompute: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        """
        Processes source dataloader to create data for reconstruction.
        Assumes source_dataloader yields batches with 'images', 'raw_camera_params', 'voxel_gt', 'model_id', 'category'.
        'raw_camera_params' should be [B, NUM_VIEWS, 5] with (az, el, ipr, dist, fov).
        """
        cache_file_path = (
            self.cache_dir / f"{cache_key}.pkl"
            if self.cache_dir and cache_key
            else None
        )
        if cache_file_path and not force_recompute and cache_file_path.exists():
            with open(cache_file_path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded cached reconstruction data from {cache_file_path}")
            return (
                data["view_data"],
                data["voxels"],
                data["model_ids"],
                data["categories"],
            )

        all_processed_views, all_target_voxels, all_model_ids, all_categories = (
            [],
            [],
            [],
            [],
        )
        for batch in tqdm(source_dataloader, desc="Preparing reconstruction data"):
            images_batch, cam_params_batch, voxels_batch = (
                batch["images"],
                batch["camera_params"],
                batch["voxel_gt"],
            )

            B_models, NUM_VIEWS, C, H, W = images_batch.shape

            img_features_dict = self.image_feature_extractor.extract_features(
                images_batch.view(B_models * NUM_VIEWS, C, H, W),
                layers=image_feat_layers,
                feature_type=image_feat_type,
            )
            img_feats_flat = (
                list(img_features_dict.values())[0].cpu()
                if img_features_dict
                else torch.empty(B_models * NUM_VIEWS, 0)
            )

            img_feats_views = img_feats_flat.view(B_models, NUM_VIEWS, -1)

            proc_cam_params_views = cam_params_batch.cpu()

            combined_view_data = torch.cat(
                (img_feats_views, proc_cam_params_views), dim=-1
            )

            all_processed_views.append(combined_view_data)
            all_target_voxels.append(voxels_batch)
            all_model_ids.extend(batch["model_id"])
            all_categories.extend(batch["category"])

        final_processed_view_data = torch.cat(all_processed_views, dim=0)
        final_target_voxels = torch.cat(all_target_voxels, dim=0)

        if cache_file_path:
            with open(cache_file_path, "wb") as f:
                pickle.dump(
                    {
                        "view_data": final_processed_view_data,
                        "voxels": final_target_voxels,
                        "model_ids": all_model_ids,
                        "categories": all_categories,
                    },
                    f,
                )
            logger.info(f"Cached reconstruction data to {cache_file_path}")
        return (
            final_processed_view_data,
            final_target_voxels,
            all_model_ids,
            all_categories,
        )

    def create_reconstruction_datasets(
        self,
        train_source_loader: DataLoader,
        val_source_loader: DataLoader,
        test_source_loader: DataLoader,
        image_feat_layers: Optional[List[int]] = None,
        img_feat_type: str = "cls_token",
        experiment_name: Optional[str] = None,
        force_recompute_data: bool = False,
    ) -> Tuple[
        ReconstructionInputDataset,
        ReconstructionInputDataset,
        ReconstructionInputDataset,
    ]:
        """Creates train, validation, and test datasets for the reconstruction task."""
        datasets = []
        for split_name, loader_instance in [
            ("train", train_source_loader),
            ("val", val_source_loader),
            ("test", test_source_loader),
        ]:
            cache_key = (
                f"{experiment_name}_{split_name}_reconstruction_prepared_data"
                if experiment_name
                else None
            )
            view_data, voxels, model_ids, categories = self.prepare_reconstruction_data(
                source_dataloader=loader_instance,
                image_feat_layers=image_feat_layers,
                image_feat_type=img_feat_type,
                cache_key=cache_key,
                force_recompute=force_recompute_data,
            )
            dataset = ReconstructionInputDataset(
                view_data=view_data, voxels=voxels, ids=model_ids, cats=categories
            )
            datasets.append(dataset)
            logger.info(
                f"{split_name.upper()} ReconstructionInputDataset created with {len(dataset)} samples."
            )
        return tuple(datasets)


def create_reconstruction_dataloaders(
    train_dataset: ReconstructionInputDataset,
    val_dataset: ReconstructionInputDataset,
    test_dataset: ReconstructionInputDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates DataLoaders for the reconstruction task."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
