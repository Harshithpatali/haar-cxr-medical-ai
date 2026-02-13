import torch
import torch.nn as nn


class FrequencyCNN(nn.Module):
    """
    CNN branch for 4-channel Haar wavelet coefficients.
    Input: (B, 4, 112, 112)
    """

    def __init__(self, output_dim: int = 256, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, 112, 112)
        returns: (B, output_dim)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
