import torch
import numpy as np
import cv2
from typing import Tuple


class GradCAM:
    """
    Grad-CAM for Spatial Branch (ResNet).
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(
        self,
        spatial_input: torch.Tensor,
        freq_input: torch.Tensor
    ) -> np.ndarray:

        self.model.zero_grad()
        output = self.model(spatial_input, freq_input)

        output.backward(torch.ones_like(output))

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        cam: np.ndarray,
        save_path: str
    ) -> None:

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )

        overlay = heatmap * 0.4 + np.stack([image]*3, axis=-1)
        overlay = np.uint8(overlay)

        cv2.imwrite(save_path, overlay)
