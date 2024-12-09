import pytest
from src.evaluation.metrics import calculate_accuracy, calculate_f1_score


def test_calculate_accuracy():
    """
    calculate_accuracy fonksiyonunun doğru çalıştığını test eder.
    """
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]

    accuracy = calculate_accuracy(y_true, y_pred)

    # Beklenen doğruluk: 3 doğru tahmin / 4 toplam örnek = 0.75
    assert accuracy == 0.75, f"Beklenen doğruluk: 0.75, ancak gelen: {accuracy}"


def test_calculate_f1_score():
    """
    calculate_f1_score fonksiyonunun doğru çalıştığını test eder.
    """
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]

    f1_score = calculate_f1_score(y_true, y_pred)

    # Beklenen F1 skoru, precision ve recall hesaplamalarına bağlıdır.
    assert f1_score > 0.5, f"F1 skoru çok düşük: {f1_score}"
