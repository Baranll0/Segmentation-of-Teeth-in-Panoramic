from setuptools import setup, find_packages
import os

# Dependencies listesi
REQUIRED_PACKAGES = [
    'numpy',
    'torch',
    'torchvision',
    'pandas',
    'scikit-learn',
    'matplotlib',
    'Pillow',
    'pytest',
    'tqdm',
    'seaborn',
    'scipy',
    'opencv-python',
    'gradio',  # Eğer Gradio kullanıyorsanız
]

# setup.py'nin bulunduğu dizinde tests klasörünü bul
TESTS_DIR = os.path.join(os.path.dirname(__file__), 'tests')

# Setup fonksiyonu
setup(
    name="car-classification-pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        'console_scripts': [
            'run_pipeline = pipeline.run_pipeline:main',  # Pipeline'ı çalıştıracak ana fonksiyon
        ],
    },
    test_suite='pytest',  # Testleri çalıştırmak için pytest kullanıyoruz
    tests_require=REQUIRED_PACKAGES,
)
