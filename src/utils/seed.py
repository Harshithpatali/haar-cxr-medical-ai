import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
