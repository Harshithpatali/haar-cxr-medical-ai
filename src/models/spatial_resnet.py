import torch
import torch.nn as nn
import torchvision.models as models


class SpatialResNet(nn.Module):
    """
    Modified ResNet18 for single-channel medical images.
    Outputs configurable feature dimension.
    """

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()

        backbone = models.resnet18(weights=None)

        # Modify first convolution for grayscale input
        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        in_features = backbone.fc.in_features

        self.projection = nn.Linear(in_features, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 224, 224)
        returns: (B, output_dim)
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.projection(features)
