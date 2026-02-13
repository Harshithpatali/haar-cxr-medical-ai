import os
import torch
from typing import List


def load_fold_models(model_class, config: dict, device="cpu") -> List:

    models = []
    folds = config["cross_validation"]["folds"]

    for fold in range(folds):

        model_path = f"checkpoints/fold_{fold}.pt"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Expected model file not found: {model_path}"
            )

        model = model_class(
            feature_dim=config["model"]["feature_dim"],
            dropout=config["model"]["dropout"]
        )

        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )

        model.to(device)
        model.eval()

        models.append(model)

    return models


def ensemble_predict(models, spatial_x, freq_x):

    probs = []

    with torch.no_grad():
        for model in models:
            logits = model(spatial_x, freq_x)
            probs.append(torch.sigmoid(logits))

    probs = torch.stack(probs)

    return torch.mean(probs, dim=0)
