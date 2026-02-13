import os
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from src.transforms.wavelet_transform import haar_wavelet_2d


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


class CXRPneumoniaDataset(Dataset):
    """
    Production-grade dataset loader with file validation.
    """

    def __init__(
        self,
        root_dir: str,
        transform=None
    ) -> None:

        self.samples: List[Tuple[str, int]] = []
        self.transform = transform

        classes = ["NORMAL", "PNEUMONIA"]

        for label, cls in enumerate(classes):

            class_dir = os.path.join(root_dir, cls)

            if not os.path.exists(class_dir):
                raise ValueError(f"Directory not found: {class_dir}")

            for img_name in os.listdir(class_dir):

                # Skip hidden/system files
                if img_name.startswith("."):
                    continue

                # Validate extension
                if not img_name.lower().endswith(VALID_EXTENSIONS):
                    continue

                img_path = os.path.join(class_dir, img_name)

                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise ValueError("No valid image files found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("L")
        except UnidentifiedImageError:
            raise RuntimeError(f"Corrupted image file: {img_path}")

        if self.transform is None:
            raise ValueError("Transform must be provided")

        spatial_tensor = self.transform(image)

        image_np = np.array(image.resize((224, 224))) / 255.0
        wavelet_np = haar_wavelet_2d(image_np)

        wavelet_tensor = torch.tensor(wavelet_np, dtype=torch.float32)

        return spatial_tensor, wavelet_tensor, torch.tensor(label, dtype=torch.float32)
