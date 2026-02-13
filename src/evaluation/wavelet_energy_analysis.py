import numpy as np
from typing import Dict, List
from scipy.stats import ttest_ind
from src.transforms.wavelet_transform import compute_wavelet_energy


def compute_dataset_energy(dataset) -> Dict[str, List[float]]:

    energy_stats = {
        "LL": [],
        "LH": [],
        "HL": [],
        "HH": [],
        "labels": []
    }

    for spatial_x, wavelet_x, label in dataset:

        energies = compute_wavelet_energy(wavelet_x.numpy())

        for band in ["LL", "LH", "HL", "HH"]:
            energy_stats[band].append(energies[band])

        energy_stats["labels"].append(int(label))

    return energy_stats


def compare_energy_distributions(energy_stats: Dict) -> Dict:

    results = {}

    labels = np.array(energy_stats["labels"])

    for band in ["LL", "LH", "HL", "HH"]:

        values = np.array(energy_stats[band])

        normal = values[labels == 0]
        pneumonia = values[labels == 1]

        t_stat, p_val = ttest_ind(normal, pneumonia, equal_var=False)

        results[band] = {
            "normal_mean": float(np.mean(normal)),
            "pneumonia_mean": float(np.mean(pneumonia)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val)
        }

    return results
