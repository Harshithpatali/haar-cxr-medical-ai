import os
import numpy as np
from typing import List, Dict
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from src.models.dual_branch_model import DualBranchModel
from src.training.trainer import Trainer
from src.utils.logger import get_logger


class CrossValidator:

    def __init__(self, dataset, config: dict) -> None:
        self.dataset = dataset
        self.config = config
        self.logger = get_logger("CrossValidator")

        self.folds = config["cross_validation"]["folds"]

    def run(self) -> Dict[str, float]:

        labels = [label for _, _, label in self.dataset]
        labels = np.array(labels)

        skf = StratifiedKFold(
            n_splits=self.folds,
            shuffle=True,
            random_state=self.config["project"]["seed"]
        )

        fold_metrics: List[Dict] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):

            self.logger.info(f"Starting Fold {fold}")

            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=self.config["data"]["batch_size"],
                shuffle=True,
                num_workers=self.config["data"]["num_workers"]
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=self.config["data"]["batch_size"],
                shuffle=False,
                num_workers=self.config["data"]["num_workers"]
            )

            model = DualBranchModel(
                feature_dim=self.config["model"]["feature_dim"],
                dropout=self.config["model"]["dropout"]
            )

            trainer = Trainer(model, self.config, fold=fold)
            metrics = trainer.fit(train_loader, val_loader)

            fold_metrics.append(metrics)

        return self._aggregate_metrics(fold_metrics)

    def _aggregate_metrics(self, fold_metrics: List[Dict]) -> Dict[str, float]:

        aggregated = {}
        keys = fold_metrics[0].keys()

        for key in keys:
            values = [fm[key] for fm in fold_metrics]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))

        self.logger.info(f"Cross-Validation Results: {aggregated}")
        return aggregated
