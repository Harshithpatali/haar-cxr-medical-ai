from torchvision import transforms


def get_spatial_transforms(image_size: int):
    """
    Standard medical image preprocessing.
    """

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
