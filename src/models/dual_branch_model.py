import torch
import torch.nn as nn
from src.models.spatial_resnet import SpatialResNet
from src.models.frequency_cnn import FrequencyCNN


class DualBranchModel(nn.Module):
    """
    Hybrid model combining spatial CNN and Haar frequency CNN.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.spatial_branch = SpatialResNet(output_dim=feature_dim)
        self.frequency_branch = FrequencyCNN(
            output_dim=feature_dim,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        spatial_x: torch.Tensor,
        frequency_x: torch.Tensor
    ) -> torch.Tensor:
        """
        spatial_x: (B,1,224,224)
        frequency_x: (B,4,112,112)
        returns logits (B,1)
        """

        spatial_features = self.spatial_branch(spatial_x)
        frequency_features = self.frequency_branch(frequency_x)

        fused = torch.cat([spatial_features, frequency_features], dim=1)

        logits = self.classifier(fused)

        return logits
