# Segmentation of Teeth in Panoramic X-rays

This project implements a deep learning-based approach to segment teeth from panoramic X-ray images using a Nested U-Net (U-Net++) architecture. The training pipeline has been updated to include a combined loss function (CrossEntropyLoss + DiceLoss), enhanced preprocessing, and extensive training over 200 epochs.



---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Results](#results)

---

## Overview

This project focuses on segmenting individual teeth from panoramic X-rays using deep learning techniques.
The updated pipeline includes:
- Preprocessing: Image resizing, normalization, and format conversion.
- Model: Nested U-Net (U-Net++) architecture.
- Loss Function: Combination of CrossEntropyLoss and DiceLoss.
- Post-processing: Connected Component Analysis (CCA) for bounding box generation.
- Evaluation Metrics: Dice Coefficient, Combined Loss, and Accuracy.

**Note: The developed project will be improved and republished.**


---

## Dataset

The dataset consists of panoramic X-ray images and corresponding multi-class masks. Each tooth is labeled with a unique class ID to allow individual segmentation.

```bash
/data/
│
├── images/                # Input X-ray images (JPG format)
├── masks/                 # Corresponding segmentation masks
└── processed/             # Preprocessed data (resized, converted)


```
Image Format: Resized to 512x512. Converted from PNG to JPG.
Classes: 33 unique classes (teeth). Background is labeled as 0.

---

## Model Architecture

The Nested U-Net (U-Net++) model was used due to its strong performance in semantic segmentation tasks.
- **Encoder**: Feature extraction using convolutional blocks.
- **Bottleneck**: High-level representation of features.
- **Decoder**: Upsampling with skip connections to preserve spatial information.
- **Loss Function**: Combined CrossEntropyLoss and DiceLoss.

UNet is chosen for its proven efficiency in semantic segmentation tasks.

---

## Training

### Training Details
- **Batch Size**: 2
- **Optimizer**: Adam
- **Learning Rate**: \(1e-4\)
- **Loss Function**: 	CrossEntropyLoss + DiceLoss
- **Metrics**: Dice Coefficient and Accuracy
- **Training Duration:** Model trained for 200 epochs.
- **Best Model: best.pt** (lowest validation loss).
- **Last Model: last.pt** (final model after all epochs).

---

## Results

### Performance Metrics
- **Final Combined Loss**: 	0.199
- **Best Model**:	best.pt
- **Final Model**:	last.pt

---

### Prediction Examples
![Predict 1](https://i.imgur.com/oXTtScb.png)
![Predict 2](https://i.imgur.com/YN9kUpU.png)
![Predict 3](https://i.imgur.com/ArtC61U.png)

These plots show the convergence of the model during training and validation.
---

By incorporating these improvements, this project aims to deliver a robust and scalable solution for teeth segmentation in panoramic X-rays.