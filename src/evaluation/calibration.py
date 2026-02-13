import numpy as np
import matplotlib.pyplot as plt


def expected_calibration_error(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10
) -> float:

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        bin_mask = (y_probs >= bins[i]) & (y_probs < bins[i + 1])

        if np.sum(bin_mask) > 0:
            acc = np.mean(y_true[bin_mask] == (y_probs[bin_mask] >= 0.5))
            conf = np.mean(y_probs[bin_mask])
            ece += (np.sum(bin_mask) / total) * abs(acc - conf)

    return float(ece)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: str
) -> None:

    bins = np.linspace(0.0, 1.0, 11)
    bin_centers = []
    accuracies = []

    for i in range(10):
        bin_mask = (y_probs >= bins[i]) & (y_probs < bins[i + 1])

        if np.sum(bin_mask) > 0:
            acc = np.mean(y_true[bin_mask] == (y_probs[bin_mask] >= 0.5))
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            accuracies.append(acc)

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(bin_centers, accuracies, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.savefig(save_path)
    plt.close()
