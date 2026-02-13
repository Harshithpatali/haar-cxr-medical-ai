import os
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow

from src.evaluation.metrics import compute_metrics
from src.evaluation.calibration import (
    expected_calibration_error,
    plot_reliability_diagram
)
from src.training.early_stopping import EarlyStopping
from src.utils.logger import get_logger


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        fold: int = 0
    ) -> None:

        self.model = model
        self.config = config
        self.fold = fold
        self.logger = get_logger(f"Trainer_Fold_{fold}")

        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"]
        )

        self.early_stopping = EarlyStopping(
            patience=config["training"]["early_stopping_patience"]
        )

    def train_one_epoch(self, loader: DataLoader) -> float:

        self.model.train()
        total_loss = 0.0

        for spatial_x, freq_x, labels in loader:
            spatial_x = spatial_x.to(self.device)
            freq_x = freq_x.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            logits = self.model(spatial_x, freq_x)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader: DataLoader) -> Tuple[dict, float]:

        self.model.eval()
        y_true = []
        y_probs = []

        with torch.no_grad():
            for spatial_x, freq_x, labels in loader:
                spatial_x = spatial_x.to(self.device)
                freq_x = freq_x.to(self.device)

                logits = self.model(spatial_x, freq_x)
                probs = torch.sigmoid(logits)

                y_true.extend(labels.numpy())
                y_probs.extend(probs.cpu().numpy().flatten())

        y_true = np.array(y_true)
        y_probs = np.array(y_probs)

        metrics = compute_metrics(y_true, y_probs)
        ece = expected_calibration_error(y_true, y_probs)

        metrics["ece"] = ece

        return metrics, y_probs

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> dict:

        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        with mlflow.start_run(run_name=f"fold_{self.fold}"):

            mlflow.log_params(self.config["training"])

            for epoch in range(self.config["training"]["epochs"]):

                train_loss = self.train_one_epoch(train_loader)
                val_metrics, y_probs = self.validate(val_loader)

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_auc", val_metrics["roc_auc"], step=epoch)

                self.logger.info(
                    f"Epoch {epoch} | "
                    f"Loss: {train_loss:.4f} | "
                    f"AUC: {val_metrics['roc_auc']:.4f}"
                )

                if self.early_stopping(val_metrics["roc_auc"]):
                    self.logger.info("Early stopping triggered.")
                    break

            # Save checkpoint
            checkpoint_path = f"checkpoints/fold_{self.fold}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(self.model.state_dict(), checkpoint_path)

            mlflow.log_artifact(checkpoint_path)

            # Calibration plot
            reliability_path = f"logs/reliability_fold_{self.fold}.png"
            os.makedirs("logs", exist_ok=True)
            plot_reliability_diagram(
                y_true=np.array([]),  # placeholder
                y_probs=np.array([]),  # will re-evaluate properly later
                save_path=reliability_path
            )

            return val_metrics
