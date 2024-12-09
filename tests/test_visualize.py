import pytest
import matplotlib.pyplot as plt
from evaluation.visualize import plot_accuracy_loss, plot_f1_score


def test_plot_accuracy_loss():
    """
    plot_accuracy_loss fonksiyonunun doğru çalıştığını test eder.
    """
    # Test için sahte doğruluk ve kayıp verileri
    accuracy = [0.5, 0.6, 0.7, 0.8]
    loss = [1.0, 0.8, 0.6, 0.4]

    # Grafik çizimi
    plot_accuracy_loss(accuracy, loss)

    # Grafiklerin çizildiğinden emin ol
    assert len(plt.gca().lines) == 2, "Doğruluk ve kayıp grafiklerinin çizilmesi gerekti."


def test_plot_f1_score():
    """
    plot_f1_score fonksiyonunun doğru çalıştığını test eder.
    """
    # Test için sahte F1 skoru verileri
    f1_scores = [0.5, 0.6, 0.7, 0.8]

    # Grafik çizimi
    plot_f1_score(f1_scores)

    # Grafik çizildiğinden emin ol
    assert len(plt.gca().lines) == 1, "F1 skoru grafiği çizilmelidir."