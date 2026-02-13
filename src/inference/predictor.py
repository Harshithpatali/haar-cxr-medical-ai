import os
import torch
import numpy as np
from PIL import Image
from typing import Dict

from src.models.dual_branch_model import DualBranchModel
from src.evaluation.ensemble import load_fold_models, ensemble_predict
from src.uncertainty.mc_dropout import mc_dropout_predict
from src.transforms.wavelet_transform import haar_wavelet_2d, compute_wavelet_energy
from src.explainability.gradcam import GradCAM
from src.transforms.spatial_transforms import get_spatial_transforms


class PneumoniaPredictor:

    def __init__(self, config: dict):

        self.config = config
        self.device = torch.device("cpu")

        self.models = load_fold_models(
            DualBranchModel,
            config,
            device="cpu"
        )

        self.transform = get_spatial_transforms(
            config["data"]["image_size"]
        )

        # GradCAM uses first fold model
        target_layer = self.models[0].spatial_branch.feature_extractor[-1]
        self.gradcam = GradCAM(self.models[0], target_layer)

    def preprocess(self, image: Image.Image):

        image = image.convert("L").resize(
            (self.config["data"]["image_size"],
             self.config["data"]["image_size"])
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

        # Ensemble prediction
        prob = ensemble_predict(self.models, spatial_x, freq_x)
        prob_value = prob.item()

        # Uncertainty from first model
        uncertainty = mc_dropout_predict(
            self.models[0],
            spatial_x,
            freq_x,
            passes=20
        )

        # GradCAM
        cam = self.gradcam.generate(spatial_x, freq_x)

        os.makedirs("logs", exist_ok=True)
        cam_path = "logs/gradcam_output.png"
        GradCAM.overlay_heatmap(image_np, cam, cam_path)

        # Wavelet energy
        energies = compute_wavelet_energy(freq_x.squeeze().numpy())

        return {
            "probability": prob_value,
            "prediction": "PNEUMONIA" if prob_value >= 0.5 else "NORMAL",
            "uncertainty": uncertainty,
            "gradcam_path": cam_path,
            "wavelet_energy": energies
        }
