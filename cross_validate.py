import os
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.transforms.spatial_transforms import get_spatial_transforms
from src.datasets.cxr_dataset import CXRPneumoniaDataset
from src.training.cross_validator import CrossValidator


def main():

    config = load_config("configs/config.yaml")
    set_seed(config["project"]["seed"])

    transform = get_spatial_transforms(config["data"]["image_size"])

    dataset = CXRPneumoniaDataset(
        root_dir=os.path.join(config["data"]["root_dir"], "train"),
        transform=transform
    )

    cv = CrossValidator(dataset, config)
    results = cv.run()

    print("Final Cross-Validation Results:")
    print(results)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

