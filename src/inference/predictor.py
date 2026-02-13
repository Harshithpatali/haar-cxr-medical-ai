import os
import torch
import numpy as np
from PIL import Image
from typing import Dict

from src.models.dual_branch_model import DualBranchModel
from src.uncertainty.mc_dropout import mc_dropout_predict
from src.transforms.wavelet_transform import (
    haar_wavelet_2d,
    compute_wavelet_energy
)
from src.explainability.gradcam import GradCAM
from src.transforms.spatial_transforms import get_spatial_transforms


class PneumoniaPredictor:
    """
    Streamlit deployment version:
    - Single model loading (no ensemble)
    - MC Dropout uncertainty
    - Grad-CAM explainability
    """

    def __init__(self, config: dict):

        self.config = config
        self.device = torch.device("cpu")

        # 🔴 CHANGE THIS if your filename is different
        model_path = "checkpoints/fold_2.pt"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        # Initialize model
        self.model = DualBranchModel(
            feature_dim=config["model"]["feature_dim"],
            dropout=config["model"]["dropout"]
        )

        # Load trained weights
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

        # Image transforms
        self.transform = get_spatial_transforms(
            config["data"]["image_size"]
        )

        # Grad-CAM target layer (last conv block of spatial branch)
        target_layer = self.model.spatial_branch.feature_extractor[-1]
        self.gradcam = GradCAM(self.model, target_layer)

    def preprocess(self, image: Image.Image):

        image = image.convert("L").resize(
            (
                self.config["data"]["image_size"],
                self.config["data"]["image_size"]
            )
        )

        spatial_tensor = self.transform(image).unsqueeze(0)

        image_np = np.array(image) / 255.0
        wavelet_np = haar_wavelet_2d(image_np)
        wavelet_tensor = torch.tensor(
            wavelet_np,
            dtype=torch.float32
        ).unsqueeze(0)

        return spatial_tensor, wavelet_tensor, image_np

    def predict(self, image: Image.Image) -> Dict:

        spatial_x, freq_x, image_np = self.preprocess(image)

        # Forward pass
        with torch.no_grad():
            logits = self.model(spatial_x, freq_x)
            prob = torch.sigmoid(logits)

        prob_value = prob.item()

        # MC Dropout Uncertainty
        uncertainty = mc_dropout_predict(
            self.model,
            spatial_x,
            freq_x,
            passes=20
        )

        # Grad-CAM
        cam = self.gradcam.generate(spatial_x, freq_x)

        os.makedirs("logs", exist_ok=True)
        cam_path = "logs/gradcam_output.png"

        GradCAM.overlay_heatmap(
            image_np,
            cam,
            cam_path
        )

        # Wavelet Energy
        energies = compute_wavelet_energy(
            freq_x.squeeze().numpy()
        )

        return {
            "probability": prob_value,
            "prediction": (
                "PNEUMONIA"
                if prob_value >= 0.5
                else "NORMAL"
            ),
            "uncertainty": uncertainty,
            "gradcam_path": cam_path,
            "wavelet_energy": energies
        }
