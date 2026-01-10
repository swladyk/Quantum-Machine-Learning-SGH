from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Oblicza trafność kierunku (góra/dół)."""
    if len(y_true) < 2 or len(y_pred) < 2:
        return float("nan")
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    correct = (true_dir == pred_dir).sum()
    return correct / len(true_dir)


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_loss: List[float],
    test_loss: List[float],
    split_idx: int,
):
    """Wykres predykcji oraz strat z wyraźnym podziałem Train/Test."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # Predykcje vs prawdziwe wartości
    axes[0].plot(y_true, label="Prawdziwe ceny", color="black")
    axes[0].plot(y_pred, label="Predykcja", color="tab:orange", linestyle="--")
    axes[0].axvline(split_idx, color="red", linestyle=":", label="Train/Test split")
    axes[0].set_ylabel("Cena")
    axes[0].legend()
    axes[0].set_title("Prawdziwe vs przewidziane ceny")

    da = directional_accuracy(y_true[split_idx:], y_pred[split_idx:])
    axes[0].text(
        0.02,
        0.95,
        f"Directional Accuracy (test): {da:.2%}",
        transform=axes[0].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    # Historia straty
    axes[1].plot(train_loss, label="Train loss (MSE)")
    axes[1].plot(test_loss, label="Test loss (MSE)")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].set_title("Przebieg funkcji kosztu")

    plt.tight_layout()
    plt.show()
