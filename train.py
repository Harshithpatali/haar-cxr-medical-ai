import os
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.transforms.spatial_transforms import get_spatial_transforms
from src.datasets.cxr_dataset import CXRPneumoniaDataset
import torch
from src.models.dual_branch_model import DualBranchModel

model = DualBranchModel(feature_dim=256)
spatial = torch.randn(4, 1, 224, 224)
freq = torch.randn(4, 4, 112, 112)

out = model(spatial, freq)
print(out.shape)  # Should be [4,1]


def main():

    config = load_config("configs/config.yaml")
    set_seed(config["project"]["seed"])
    logger = get_logger("train")

    train_dir = os.path.join(config["data"]["root_dir"], "train")

    transform = get_spatial_transforms(config["data"]["image_size"])

    dataset = CXRPneumoniaDataset(train_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"]
    )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info("Block 1 infrastructure working.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

