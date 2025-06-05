import pickle
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import logging
from typing import Any, Callable, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class BasePipeline:
    """
    Base pipeline for feature extraction or reconstruction processing.
    Handles caching (pickle) and batching over a DataLoader.

    Subclasses must implement _process_batch(batch) -> (features, targets, metadata_dict)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, cache_key: Optional[str]) -> Optional[Path]:
        if self.cache_dir and cache_key:
            return self.cache_dir / f"{cache_key}.pkl"
        return None

    def _load_cache(self, cache_file: Path) -> Tuple[Any, Any, Dict]:
        logger.info(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["features"], data["targets"], data.get("metadata", {})

    def _save_cache(
        self,
        cache_file: Path,
        features: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict,
    ):
        cache_data = {"features": features, "targets": targets, "metadata": metadata}
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"Saved cached data to {cache_file}")

    def extract(
        self,
        dataloader: DataLoader,
        cache_key: Optional[str] = None,
        force_recompute: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Iterate over dataloader and process each batch via _process_batch,
        with optional caching.

        Returns:
            features: [N, ...]
            targets: [N, ...]
            metadata: Dict[str, list]
        """
        cache_file = self._cache_file(cache_key)
        if cache_file and cache_file.exists() and not force_recompute:
            return self._load_cache(cache_file)

        all_features = []
        all_targets = []
        aggregated_meta: Dict[str, list] = {}

        logger.info(f"Processing {len(dataloader)} batches...")
        for batch in dataloader:
            feats, targets, meta = self._process_batch(batch)
            all_features.append(feats.cpu())
            all_targets.append(targets.cpu())
            for k, v in (meta or {}).items():
                aggregated_meta.setdefault(k, []).extend(v)

        features = torch.cat(all_features, dim=0)
        targets = torch.cat(all_targets, dim=0)

        if cache_file:
            self._save_cache(cache_file, features, targets, aggregated_meta)

        return features, targets, aggregated_meta

    def _process_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Given a batch from the dataloader, return a tuple:
        (features: Tensor, targets: Tensor, metadata: Dict[str, list])
        Must be implemented by subclass.
        """
        raise NotImplementedError("_process_batch must be implemented in subclass")
