import pywt
import numpy as np


def haar_wavelet_2d(image: np.ndarray) -> np.ndarray:
    """
    1-level Haar wavelet transform.
    Returns 4-channel array: LL, LH, HL, HH
    """

    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    return np.stack([LL, LH, HL, HH], axis=0)


def compute_wavelet_energy(wavelet_tensor: np.ndarray) -> dict:
    """
    Compute energy per subband.
    E = sum(x^2)
    """
    energies = {}

    bands = ["LL", "LH", "HL", "HH"]

    for i, band in enumerate(bands):
        energies[band] = float(np.sum(wavelet_tensor[i] ** 2))

    return energies
