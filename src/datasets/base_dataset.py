import random
from pathlib import Path
from typing import List, Optional, Dict

from torch.utils.data import Dataset


class BaseSplitDataset(Dataset):
    """
    Base class for datasets that require train/val/test splits via .lst files.
    Subclasses should implement `_gather_all_models()` to return a list of model IDs,
    and `_prepare_samples()` to map model IDs to sample dictionaries.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        categories: Optional[List[str]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_seed: int = 42,
        subset_percentage: Optional[float] = None,
        reset_splits: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.categories = categories
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        # ensure split files exist
        self._ensure_splits(reset_splits)
        # read model IDs from split file
        split_file = self.root / f"{self.split}.lst"
        with open(split_file, "r") as f:
            self.model_list = [line.strip() for line in f]
        # filter categories
        if self.categories:
            self.model_list = [
                m for m in self.model_list if self._model_in_categories(m)
            ]
        # optional subset
        if subset_percentage is not None:
            random.seed(self.split_seed)
            num = int(len(self.model_list) * subset_percentage)
            self.model_list = random.sample(self.model_list, num)
        # prepare samples dicts
        self.samples = self._prepare_samples()

    def _ensure_splits(self, reset: bool = False):
        # create or reload split files
        split_files = [self.root / f"{s}.lst" for s in ["train", "val", "test"]]
        if reset or any(not p.exists() for p in split_files):
            all_models = self._gather_all_models()
            random.seed(self.split_seed)
            random.shuffle(all_models)
            n = len(all_models)
            n_train = int(self.train_ratio * n)
            n_val = int(self.val_ratio * n)
            splits = {
                "train": all_models[:n_train],
                "val": all_models[n_train : n_train + n_val],
                "test": all_models[n_train + n_val :],
            }
            for split_name, mods in splits.items():
                p = self.root / f"{split_name}.lst"
                with open(p, "w") as f:
                    for m in mods:
                        f.write(f"{m}\n")

    def _model_in_categories(self, model_id: str) -> bool:
        cat_id = model_id.split("/")[0]
        from .shapenet_3dr2n2 import ShapeNet3DR2N2

        return ShapeNet3DR2N2.CATEGORIES.get(cat_id) in self.categories

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def _gather_all_models(self) -> List[str]:
        raise NotImplementedError

    def _prepare_samples(self) -> List[Dict]:
        raise NotImplementedError
