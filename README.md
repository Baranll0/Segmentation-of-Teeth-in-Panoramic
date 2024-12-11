# Segmentation of Teeth in Panoramic X-rays

This project implements a deep learning-based approach to segment teeth from panoramic X-ray images using a custom UNet model. The training process is designed to maximize segmentation accuracy while monitoring the performance through various metrics and visualizations.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Results](#results)
6. [Visualizations](#visualizations)
7. [How to Run](#how-to-run)
8. [Directory Structure](#directory-structure)

---

## Overview

This project focuses on the semantic segmentation of teeth in panoramic X-rays using a UNet model. The pipeline includes:
- Preprocessing the dataset (augmentation and normalization).
- Training the UNet model.
- Early stopping for efficient training.
- Validation through metrics such as Dice Coefficient and loss.
- Visualization of training performance and predictions.

---

## Dataset

The dataset consists of X-ray images and corresponding masks stored in the following directory structure:

"/dataset/DentalPanoramicXrays/images/masks/"

- **Images**: Grayscale X-ray images.
- **Masks**: Binary masks where the teeth are segmented.
---

## Model Architecture

The model is a customized implementation of UNet with the following features:
- **Encoder**: Stacked convolutional layers followed by max pooling.
- **Bottleneck**: High-level feature extraction.
- **Decoder**: Transpose convolution layers for upsampling and skip connections to preserve spatial information.

UNet is chosen for its proven efficiency in semantic segmentation tasks.

---

## Training

### Training Details
- **Batch Size**: 2
- **Optimizer**: Adam
- **Learning Rate**: \(1e-4\)
- **Loss Function**: Dice Loss
- **Metrics**: Dice Coefficient and Accuracy
- **Early Stopping**: Triggered after 10 epochs with no improvement in validation loss.

### Training Progress
The model was trained for up to 100 epochs but stopped early at epoch 43 due to early stopping.

---

## Results

### Performance Metrics
- **Final Train Accuracy**: 93.40%
- **Final Validation Accuracy**: 90.01%
- **Final Train Dice Coefficient**: 0.934
- **Final Validation Dice Coefficient**: 0.9001
- **Final Validation Loss**:0.099
- **Final Train Loss**:0.066

---

## Visualizations

### Dice Coefficient Plot
![Dice Coefficient Plot](https://i.imgur.com/QGEGU35.png)

### Loss Plot
![Loss Plot](https://i.imgur.com/riLItp6.png)

### Train Model

![Train Model](https://i.imgur.com/b7KjpKm.png)

### Prediction Examples
![Predict 1](https://i.imgur.com/JukrBmG.png)
![Predict 2](https://i.imgur.com/vS1SRZ1.png)
![Predict 3](https://i.imgur.com/mpUdhRG.png)

These plots show the convergence of the model during training and validation.
---

1. Clone this repository:
   ```bash
   git clone git@github.com:Baranll0/Segmentation-of-Teeth-in-Panoramic.git
   cd Segmentation-of-Teeth-in-Panoramic
   
2. Install the required dependencies:
```bash
  pip install -r requirements.txt
 ```
3. Run the pipeline:
```bash
python run_pipeline.py
```
4. Outputs will be saved in:
```bash
/outputs/models/       # Saved models
/outputs/visualizations/  # Visualizations
```
## Future Improvements and Experiments

This project lays the foundation for the segmentation of teeth in panoramic X-rays using a UNet architecture. However, there are several avenues for further improvements and experiments to enhance the model's performance and robustness:

1. Transfer Learning
Leveraging pretrained models (e.g., ResNet, EfficientNet) as encoders in the UNet architecture can potentially improve the model's learning efficiency and accuracy. Pretrained weights from large datasets such as ImageNet provide a strong starting point for feature extraction, particularly in medical imaging tasks.

2. Ensemble Learning
Implementing ensemble learning methods, such as averaging the predictions of multiple UNet variants or other architectures, can increase prediction stability and accuracy. Comparison of ensemble results with the current standalone model would offer deeper insights.

3. Hyperparameter Optimization
Experimenting with advanced hyperparameter tuning techniques like Grid Search, Random Search, or Bayesian Optimization can yield further improvements. Parameters such as learning rate, dropout rate, and batch size could be fine-tuned for better convergence and generalization.

4. Comparison with Other Architectures
Testing and comparing other segmentation architectures like DeepLabV3+, PSPNet, or Transformer-based architectures with the current UNet implementation can provide insights into the most suitable approach for this specific task.

5. Augmentation Strategies
Extending data augmentation with techniques like CutMix, MixUp, or GAN-generated synthetic data can help mitigate overfitting and improve generalization.

6. Post-Processing Improvements
Exploring advanced post-processing techniques, such as Conditional Random Fields (CRFs) or morphological operations, can refine the segmented mask outputs and address any noise or inaccuracies.

7. Performance Evaluation on Additional Metrics
While Dice Coefficient and Accuracy are used as primary metrics, incorporating additional metrics such as Jaccard Index (IoU), Sensitivity, and Specificity can provide a more comprehensive evaluation of the model's performance.

8. Deployment and Scalability

   *   Converting the model to an optimized format (e.g., ONNX, TensorRT) for real-time deployment.
   * Scaling the pipeline to handle larger datasets or integrate into a clinical setting.
---
By incorporating these improvements and experiments, this project aims to evolve into a comprehensive solution for teeth segmentation in panoramic X-rays. Future updates to this repository will include these experiments with detailed comparisons and visualizations.