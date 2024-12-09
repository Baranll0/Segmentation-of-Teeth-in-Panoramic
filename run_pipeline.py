import subprocess

def run():
    """
    Pipeline'ı başlatan ana fonksiyon.
    """
    print("Pipeline çalıştırılıyor...")

    # 1. Preprocessing (Veri Ön İşleme)
    print("Veri ön işleme başlatılıyor...")
    subprocess.run(["python", "preprocessing/preprocess.py"], check=True)

    # 2. Augmentation (Veri Artırma)
    print("Veri artırma başlatılıyor...")
    subprocess.run(["python", "preprocessing/augment.py"], check=True)

    # 3. Inference (Tahmin)
    print("Tahmin başlatılıyor...")
    subprocess.run(["python", "inference/predict.py"], check=True)

    # 4. Evaluation (Değerlendirme)
    print("Model değerlendirmesi başlatılıyor...")
    subprocess.run(["python", "evaluation/metrics.py"], check=True)
    subprocess.run(["python", "evaluation/visualize.py"], check=True)

    # 5. Testler (Unit Testleri)
    print("Testler çalıştırılıyor...")
    subprocess.run(["pytest", "tests"], check=True)

    print("Pipeline tamamlandı!")

if __name__ == "__main__":
    run()
